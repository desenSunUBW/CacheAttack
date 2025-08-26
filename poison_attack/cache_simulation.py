import torch
import numpy as np
from collections import deque
import heapq
import pandas as pd
import clip
from datasets import load_dataset
import glob
from functools import cmp_to_key
from utils import get_all_diffusiondb_prompts
import sys

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using CUDA device.")
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU device.")

def compare_cache_key(a, b):
    return a.last_accessed - b.last_accessed

def compare_cache_value(a, b):
    return a.access_count  * a.weight - b.access_count  * b.weight

class Data:
    def __init__(self, id, vector, weight=10000, is_key=False):
        self.id = id
        self.vector = vector
        self.access_count = 0
        self.last_accessed = 0 # 用于模拟时间，越小表示越久未访问
        self.weight = weight
        self.is_key = is_key # 标记是否是Key Data，辅助区分

    def __repr__(self):
        return f"Data(ID: {self.id}, Key:{self.is_key}, Access:{self.access_count}, LastAccess:{self.last_accessed})"

    # 使Data对象可哈希，以便作为字典的key
    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Data):
            return NotImplemented
        return self.id == other.id

class StoragePool:
    def __init__(self, max_total_value_size, max_value_per_key=5):
        self.max_total_value_size = max_total_value_size # 存储池中Value Data的总上限
        self.max_value_per_key = max_value_per_key # 每个Key最多存储多少条Value数据
        self.pool = {}  # key: Key Data object, value: list of Value Data objects
        self.current_time = 0 # 模拟时间
        self.total_value_data_count = 0 # 当前池中Value Data的总数量
        self.global_evict_candidates = []
        self.similarity_threshold = 0.65
        self.evict_key_ids = []

    def cosine_similarity(self, vec1, vec2):
        dot_product = torch.dot(vec1, vec2)
        norm_a = torch.linalg.norm(vec1)
        norm_b = torch.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return (dot_product / (norm_a * norm_b)).item()

    def _update_data_access(self, data_obj, similarity, replacement_policy):
        if replacement_policy == "FIFO":
            return True
        elif replacement_policy == "LRU":
            # data_obj.access_count += 1
            data_obj.last_accessed = self.current_time
            return True
        else:
            value_data_list = self.pool[data_obj]
            weight = self.skip_level(similarity)
            length = len(value_data_list)
            for index in range(len(value_data_list)):
                value_data = value_data_list[length - index - 1]
                if value_data.weight <= weight:
                    value_data.access_count += 1
                    value_data.last_accessed = self.current_time
                    return True
            return False

    def _evict_value_data_global(self, num_to_evict, replacement_policy):
        if num_to_evict <= 0:
            return []

        global_evict_candidates = []
        if replacement_policy != "LCBFU":
            global_evict_candidates = [key_data for key_data, value_data_list in self.pool.items()]
            final_topk = heapq.nsmallest(1, global_evict_candidates, key=lambda x: x.last_accessed)
        else:
            global_evict_candidates = [
                (val_data.id, key_data, val_data)
                for key_data, value_data_list in self.pool.items()
                for val_data in value_data_list
            ]            
            final_topk = heapq.nsmallest(5, global_evict_candidates, key=lambda x: x[2].access_count * x[2].weight)
        '''
        for key_data, value_data_list in self.pool.items():
            if replacement_policy != "LCBFU":
                eviction_score = key_data.last_accessed
                heapq.heappush(global_evict_candidates, (eviction_score, key_data))
            else:
                for val_data in value_data_list:
                    # 淘汰分数：越小越容易淘汰。访问次数少，最近访问时间小（久未访问）则分数低
                    # 这里可以引入Key的权重、Value在列表中的权重等
                    eviction_score = val_data.access_count  * val_data.weight
                    heapq.heappush(global_evict_candidates, (eviction_score, val_data.id, key_data, val_data)) # 存储key_data以便从池中删除
        '''
        evicted_items = []
        for _ in range(min(num_to_evict, len(final_topk))):
            if not final_topk:
                break
            if replacement_policy != "LCBFU":
                # '''
                key = heapq.heappop(final_topk)
                evicted_items.append(key)
                del self.pool[key]
                self.evict_key_ids.append(key.id)
                # '''
            else:
                _, parent_key, val_data_to_evict = heapq.heappop(final_topk)
                
                # 从父Key的Value列表中移除
                if parent_key in self.pool and val_data_to_evict in self.pool[parent_key]:
                    self.pool[parent_key].remove(val_data_to_evict)
                    self.total_value_data_count -= 1
                    evicted_items.append(val_data_to_evict)
                    # 如果一个Key下的所有Value都被淘汰完了，可以考虑淘汰这个Key
                    if not self.pool[parent_key]:
                        del self.pool[parent_key]
                        self.evict_key_ids.append(parent_key.id)
                    # print(f"  Key {parent_key.id} 因所有Value被淘汰而移除。")
        return evicted_items


    def skip_level(self, similarity):
        if similarity >= 0.65 and similarity < 0.75:
            return 1
        elif similarity >= 0.75 and similarity < 0.85:
            return 2
        elif similarity >= 0.85 and similarity < 0.9:
            return 3
        elif similarity >= 0.9 and similarity < 0.95:
            return 4
        elif similarity >= 0.95:
            return 5
        else:
            return 0

    def process_incoming_data(self, new_data, replacement_policy="LCBFU", replace_cache=False):
        self.current_time += 1
        
        # 记录操作前池中所有数据的ID，用于判断淘汰
        all_data_ids_before = self._get_all_current_data_ids()

        best_match_key_data = None
        best_key_similarity = -1

        all_key_data_list = list(self.pool.keys())
        # 1. 查找相似Key
        if all_key_data_list:
            # Stack all key vectors into a single tensor for batch processing
            # Shape will be (num_keys, vector_dim)
            all_keys_vectors = torch.stack([kd.vector for kd in all_key_data_list])
            
            # Unsqueeze new_data.vector to (1, vector_dim) to enable batch comparison
            new_data_vector_batch = new_data.vector

            # Compute cosine similarity between new_data_vector_batch and all_keys_vectors
            # Output `similarities` will be a tensor of shape (num_keys,)
            # print(new_data_vector_batch.shape)
            # print(all_keys_vectors.shape)
            similarities = torch.matmul(all_keys_vectors, new_data_vector_batch.T)
            # similarities = torch.nn.functional.cosine_similarity(new_data_vector_batch, all_keys_vectors, dim=-1)
            # print(similarities.shape)
            # Find the maximum similarity and its index
            best_similarity_tensor, best_idx_tensor = torch.max(similarities, dim=0)
            
            # Get the scalar values
            best_key_similarity = best_similarity_tensor
            
            # Check if the best similarity meets the threshold
            if best_key_similarity >= self.similarity_threshold:
                best_match_key_data = all_key_data_list[best_idx_tensor.item()]

        if best_match_key_data and not self._update_data_access(best_match_key_data, best_key_similarity, replacement_policy):
            best_match_key_data = None

        if best_match_key_data and replace_cache:
            self.evict_key_ids.append(best_match_key_data.id)
            del self.pool[best_match_key_data]
            best_match_key_data = None

        if best_match_key_data:
            # value_data_list = self.pool[best_match_key_data]
            #     # 检查淘汰情况并返回
            # newly_evicted_ids = all_data_ids_before - self._get_all_current_data_ids()
            return best_match_key_data.id, self.evict_key_ids

        else:
            # 3. Key未命中: 插入新Key及其5个Value
            # 预计需要增加5个Value Data
            required_space = self.max_value_per_key # 假设新Key总是带来5个Value

            if self.total_value_data_count + required_space > self.max_total_value_size:
                # 需要全局淘汰以腾出空间
                
                if replacement_policy == "LCBFU":
                    num_to_evict = (self.total_value_data_count + required_space) - self.max_total_value_size
                    
                else:
                    num_to_evict = 1
                evicted_global_values = self._evict_value_data_global(num_to_evict, replacement_policy)
                if len(evicted_global_values) < num_to_evict:
                    print(f"  WARN: 无法淘汰足够的Value数据，目标 ，实际淘汰 {len(evicted_global_values)}。")
                    return 'failed_global_eviction', set()

            # 插入新Key和其5个Value
            new_key = Data(new_data.id, new_data.vector, is_key=True) # 确保作为Key
            new_key.last_accessed = self.current_time
            # self._update_data_access(new_key) # Key自身也要更新访问信息
            self.pool[new_key] = []

            # 生成并插入剩余的4条新的Value Data
            for j in range(self.max_value_per_key):
                # 假设新的Value数据ID是基于new_data.id递增的
                # 注意：这里需要确保ID唯一性，在实际应用中更应有全局ID生成器
                new_value_id = f"{new_data.id}_val{j}" 
                # 这些Value的向量可以与new_data相似，或完全随机
                new_val = Data(new_value_id, None, weight=j + 1, is_key=False)
                if "insert_cache" in new_value_id:
                    new_val.access_count += 1000
                new_val.last_accessed = self.current_time
                self.pool[new_key].append(new_val)
                self.total_value_data_count += 1
            # newly_evicted_ids = all_data_ids_before - self._get_all_current_data_ids()
            return '', self.evict_key_ids
        
    def _get_all_current_data_ids(self):
        """Helper to get all currently present data IDs (Keys and Values)."""
        ids = set()
        for key_data in self.pool.keys():
            ids.add(key_data.id)
            # for val_data in self.pool[key_data]:
            #     ids.add(val_data.id)
        return ids


clip_model, preprocess = clip.load("ViT-L/14", device="cuda")
def get_embeddings(text):
    # Embed special texts A
    
    with torch.no_grad():
        text = clip.tokenize(text, truncate=True).cuda()
        embedding = clip_model.encode_text(text)
        embedding = embedding / torch.norm(embedding, dim=-1, keepdim=True)
        return embedding

def simulate_data_retention(max_pool_size, 
                            prompts=None, 
                            replacement_policy=None,
                            target_prompts=None,
                            insert_prompts=None,
                            write_file=None,):
    storage_pool = StoragePool(max_pool_size)
    log_file = open(write_file, "w")
    # 记录所有进入过系统的数据的引入时间，无论是否被淘汰
    data_introduction_time = {}
    # 记录数据被淘汰的时间，用于计算保留时间
    data_eviction_time = {}

    # 初始ID计数器
    global_data_id_counter = 0
    
    target_emb = get_embeddings(target_prompts)
    target_number = len(target_prompts)
    target_add_index = 0
    insert_emb = get_embeddings(insert_prompts)
    insert_emb_backup = insert_emb.clone()
    insert_time = 0
    insert_add_index = 0
    insert = False
    length = len(prompts)
    batch_size = 1024
    target_add_gap = max_pool_size // target_number
    replace_insert_cache = False
    prepared_insert_emb = None
    start_counting_time = 0
    end_counting_time = 0
    hit_number = 0
    for i in range(length // batch_size):
        cur_prompt_emb = get_embeddings(prompts[i * batch_size : (i + 1) * batch_size])
        for b in range(batch_size):
            current_sim_cycle = i * batch_size + b + 1
            replace_cache = False
            if storage_pool.total_value_data_count > (target_add_index + 1) * target_add_gap and target_add_index < target_number:
                new_data_id = f"target_cache_{target_add_index}"
                # print(f"insert target cache {global_data_id_counter}")
                new_incoming_data = Data(new_data_id, target_emb[target_add_index].unsqueeze(0))
                target_add_index += 1
                replace_cache = True
            elif replace_insert_cache:
                # print(f"insert designed prompt with index {global_data_id_counter}")
                new_data_id = f"insert_cache_{insert_add_index}"
                new_incoming_data = Data(new_data_id, prepared_insert_emb.unsqueeze(0))
                replace_insert_cache = False
                replace_cache = True
                
            else:

            # if storage_pool.total_value_data_count + storage_pool.max_value_per_key > max_pool_size and not insert:
            # if storage_pool.total_value_data_count + 1 > max_pool_size and not insert:
            #     insert = True
            #     new_data_id = f"convert_cache_{global_data_id_counter}"
            #     new_incoming_data = Data(new_data_id, get_embeddings(convert_prompt))
            #     insert_time = storage_pool.current_time
            # else:
                new_data_id = f"Incoming_{global_data_id_counter}"
                new_incoming_data = Data(new_data_id, cur_prompt_emb[b].unsqueeze(0))
            global_data_id_counter += 1
                    
            # 记录新传入数据的时间，它可能是Key或Value
            data_introduction_time[new_incoming_data.id] = storage_pool.current_time

            # ProcessIncomingData现在会返回操作类型和被淘汰的数据ID
            hit_id, newly_evicted_ids = storage_pool.process_incoming_data(new_incoming_data, replacement_policy, replace_cache=replace_cache)
            storage_pool.evict_key_ids = []

            for evicted_id in newly_evicted_ids:
                if "target_cache" in evicted_id:
                    # print(f"LRU convert cache existing duration is {storage_pool.current_time - insert_time}")
                    # return storage_pool.current_time - insert_time
                    target_index = int(evicted_id.split("_")[-1])
                    evict_emb = target_emb[target_index].unsqueeze(0)
                    insert_similarities = torch.matmul(insert_emb, evict_emb.T)
                    insert_add_index = torch.argmax(insert_similarities).item()
                    prepared_insert_emb = insert_emb[insert_add_index].clone()
                    insert_emb[insert_add_index] = insert_emb[insert_add_index] * 0
                    replace_insert_cache = True
                elif "insert_cache" in evicted_id:
                    # print(f"evict designed cache {evicted_id}")
                    replace_insert_cache = True
                    insert_add_index = int(evicted_id.split("_")[-1])
                    prepared_insert_emb = insert_emb_backup[insert_add_index]

            
            if "insert_cache" in hit_id:
                log_file.write(f"{hit_id}, {prompts[i * batch_size + b]}\n")
                hit_number += 1
                if start_counting_time == 0:
                    start_counting_time = storage_pool.current_time
                if hit_number % 100 == 99:
                    log_file.write(f"hit {hit_number} times, current hit rate is {hit_number / (storage_pool.current_time - start_counting_time)}\n")
                
                if hit_number % 1000 == 999:
                    log_file.write(f"hit {hit_number} times, current hit rate is {hit_number / (storage_pool.current_time - start_counting_time)}\n")
                    return hit_number / (storage_pool.current_time - start_counting_time)
    
    log_file.write(f"hit {hit_number} times, current hit rate is {hit_number / (storage_pool.current_time - start_counting_time)}\n")
    print(f"total prompt number is {global_data_id_counter}")
    return hit_number / (storage_pool.current_time - start_counting_time)

def read_prompts_from_file(file_path):
    prompts = []
    with open(file_path) as f:
        prompt = f.readline()
        while prompt:
            prompts.append(prompt)
            prompt = f.readline()
    return prompts

# --- 运行模拟 ---
if __name__ == "__main__":
    items_per_gb = 2048
    base_dir = sys.argv[1]
    datasets = ["diffusiondb", "lexica"]
    models = ["flux"]
    replacement_policy = "LCBFU"
    logos = ["blue moon sign", "Mcdonald sign", "Apple sign", "Chanel symbol", "circled triangle symbol", "circled Nike symbol"]
    logo_index = int(sys.argv[2])
    logo = logos[logo_index]

    insert_method = "insert_emb_space"
    # insert_method = "insert_direct"
    # insert_method = "recover_prompts"

    for dataset in datasets:
        for model in models:
            if dataset == "lexica":
                cache_size = 1
                trainset  = load_dataset('vera365/lexica_dataset', split='train')
                testset  = load_dataset('vera365/lexica_dataset', split='test')
                prompts = trainset[:]["prompt"] + testset[:]["prompt"]
            else:
                cache_size = 100
                prompts = list(map(str, get_all_diffusiondb_prompts()))

            
            target_file = f"{dataset}/{model}_original.log"
            if insert_method == "recover_prompts":
                insert_file = f"{dataset}/{model}.log"
                # insert_file = target_file
                write_file = f"{dataset}/{model}_{logo}_{replacement_policy}_{cache_size}_hit_rate.csv"
            elif insert_method == "insert_direct":
                insert_file = f"{base_dir}/{logo}/{dataset}-{model}-prompts.log"
                write_file = f"{base_dir}/{logo}/{dataset}-{model}_{replacement_policy}_{cache_size}_prompts_hit_rate.csv"
            else:
                insert_file = f"{base_dir}/{logo}/{dataset}-{model}_cachetox.log"
                write_file = f"{base_dir}/{logo}/{dataset}-{model}_{replacement_policy}_{cache_size}_cachetox_hit_rate.csv"
                
            target_prompts = read_prompts_from_file(target_file)
            insert_prompts = read_prompts_from_file(insert_file)

            
            MAX_POOL_SIZE = int(items_per_gb * cache_size)  # 存储池最大容量

            hit_rate = simulate_data_retention(
                max_pool_size=MAX_POOL_SIZE,
                prompts=prompts,
                replacement_policy=replacement_policy,
                target_prompts=target_prompts,
                insert_prompts=insert_prompts,
                write_file=write_file,
            )
            print(f"{replacement_policy} with logo {logo} under model {model} with insert method as {insert_method}'s hit rate is {hit_rate}")
