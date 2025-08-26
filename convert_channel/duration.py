import torch
import numpy as np
from collections import deque
import heapq
import pandas as pd
import clip
from datasets import load_dataset
import glob
from utils import get_all_diffusiondb_prompts
# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using CUDA device.")
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU device.")

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

    def cosine_similarity(self, vec1, vec2):
        dot_product = torch.dot(vec1, vec2)
        norm_a = torch.linalg.norm(vec1)
        norm_b = torch.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return (dot_product / (norm_a * norm_b)).item()

    def _update_data_access(self, data_obj, similarity):
        # data_obj.access_count += 1
        # data_obj.last_accessed = self.current_time
        # return True
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


    def _evict_value_data_local(self, value_data_list, num_to_evict=1):
        """
        在单个Key的Value列表中进行局部淘汰。
        选择访问次数最少且最久未访问的Value数据。
        """
        evict_candidates = []
        for val_data in value_data_list:
            # 淘汰分数：越小越容易淘汰。访问次数少，最近访问时间小（久未访问）则分数低
            eviction_score = val_data.access_count * val_data.weight
            heapq.heappush(evict_candidates, (eviction_score, val_data.id, val_data)) # id用于打破平局，实际上data本身可以区分

        evicted_items = []
        for _ in range(min(num_to_evict, len(evict_candidates))):
            score, _, val_data_to_evict = heapq.heappop(evict_candidates)
            value_data_list.remove(val_data_to_evict)
            self.total_value_data_count -= 1
            evicted_items.append(val_data_to_evict)
            # print(f"    局部淘汰Value数据: {val_data_to_evict.id} (分数: {score:.2f})")
        return evicted_items

    def _evict_value_data_global(self, num_to_evict):
        if num_to_evict <= 0:
            return []

        global_evict_candidates = []
        for key_data, value_data_list in self.pool.items():
            '''
            eviction_score = key_data.last_accessed
            heapq.heappush(global_evict_candidates, (eviction_score, key_data))
            '''
            for val_data in value_data_list:
                # 淘汰分数：越小越容易淘汰。访问次数少，最近访问时间小（久未访问）则分数低
                # 这里可以引入Key的权重、Value在列表中的权重等
                eviction_score = val_data.access_count  * val_data.weight
                heapq.heappush(global_evict_candidates, (eviction_score, val_data.id, key_data, val_data)) # 存储key_data以便从池中删除
            

        evicted_items = []
        for _ in range(min(num_to_evict, len(global_evict_candidates))):
            if not global_evict_candidates:
                break
            score, _, parent_key, val_data_to_evict = heapq.heappop(global_evict_candidates)
            '''
            score, key = heapq.heappop(global_evict_candidates)
            evicted_items.append(key)
            del self.pool[key]
            '''
            # 从父Key的Value列表中移除
            if parent_key in self.pool and val_data_to_evict in self.pool[parent_key]:
                self.pool[parent_key].remove(val_data_to_evict)
                self.total_value_data_count -= 1
                evicted_items.append(val_data_to_evict)
                # 如果一个Key下的所有Value都被淘汰完了，可以考虑淘汰这个Key
                if not self.pool[parent_key]:
                    del self.pool[parent_key]
                    # print(f"  Key {parent_key.id} 因所有Value被淘汰而移除。")
            # print(f"  全局淘汰Value数据: {val_data_to_evict.id} (分数: {score:.2f})")
            
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

    def process_incoming_data(self, new_data, similarity_threshold=0.65):
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
            similarities = torch.nn.functional.cosine_similarity(new_data_vector_batch, all_keys_vectors, dim=-1)
            # print(similarities.shape)
            # Find the maximum similarity and its index
            best_similarity_tensor, best_idx_tensor = torch.max(similarities, dim=0)
            
            # Get the scalar values
            best_key_similarity = best_similarity_tensor
            
            # Check if the best similarity meets the threshold
            if best_key_similarity >= similarity_threshold:
                best_match_key_data = all_key_data_list[best_idx_tensor.item()]

        if best_match_key_data and not self._update_data_access(best_match_key_data, best_key_similarity):
            best_match_key_data = None

        if best_match_key_data:
            # value_data_list = self.pool[best_match_key_data]
            
            # # 尝试在当前Key的Value列表中找到相似项进行重用
            # best_match_value_data = None
            # best_value_similarity = -1
            # for value_data in value_data_list:
            #     val_sim = self.cosine_similarity(new_data.vector, value_data.vector)
            #     if val_sim >= similarity_threshold and val_sim > best_value_similarity:
            #         best_value_similarity = val_sim
            #         best_match_value_data = value_data

            # if best_match_value_data:
            #     # 2a. Value命中: 重用Value
            #     self._update_data_access(best_match_value_data)
            #     print(f"    Value {best_match_value_data.id} 命中并重用 (相似度: {best_value_similarity:.2f})")
                
            #     # 检查淘汰情况并返回
            newly_evicted_ids = all_data_ids_before - self._get_all_current_data_ids()
            return 'reused_value', newly_evicted_ids
            '''
            else:
                # 2b. Value未命中: 尝试插入新数据作为Value
                if len(value_data_list) < self.max_value_per_key:
                    # Value列表未满，直接添加
                    new_data.last_accessed = self.current_time # 设置新数据时间
                    value_data_list.append(new_data)
                    self.total_value_data_count += 1
                    print(f"    Value {new_data.id} 作为新Value插入Key {best_match_key_data.id} (当前Value数量: {len(value_data_list)})")
                    
                    newly_evicted_ids = all_data_ids_before - self._get_all_current_data_ids()
                    return 'inserted_value', newly_evicted_ids
                else:
                    # Value列表已满，需要局部淘汰一个Value
                    print(f"    Key {best_match_key_data.id} 的Value列表已满，尝试局部淘汰...")
                    evicted_local_values = self._evict_value_data_local(value_data_list, num_to_evict=1)
                    if evicted_local_values:
                        new_data.last_accessed = self.current_time
                        value_data_list.append(new_data)
                        self.total_value_data_count += 1
                        print(f"    Value {new_data.id} 替换 {evicted_local_values[0].id} 插入Key {best_match_key_data.id}")
                        
                        newly_evicted_ids = all_data_ids_before - self._get_all_current_data_ids()
                        return 'inserted_value_after_local_eviction', newly_evicted_ids
                    else:
                        print(f"    无法在Key {best_match_key_data.id} 下淘汰Value。")
                        return 'no_space_in_key', set()
            '''
        
        else:
            # 3. Key未命中: 插入新Key及其5个Value
            # 预计需要增加5个Value Data
            required_space = self.max_value_per_key # 假设新Key总是带来5个Value

            if self.total_value_data_count + required_space > self.max_total_value_size:
                # 需要全局淘汰以腾出空间
                
                num_to_evict = (self.total_value_data_count + required_space) - self.max_total_value_size
                evicted_global_values = self._evict_value_data_global(num_to_evict)
                '''
                evicted_global_values = self._evict_value_data_global(1)
                '''
                if len(evicted_global_values) < num_to_evict:
                    print(f"  WARN: 无法淘汰足够的Value数据，目标 ，实际淘汰 {len(evicted_global_values)}。")
                    return 'failed_global_eviction', set()

            # 插入新Key和其5个Value
            new_key = Data(new_data.id, new_data.vector, is_key=True) # 确保作为Key
            new_key.last_accessed = self.current_time
            # self._update_data_access(new_key) # Key自身也要更新访问信息
            self.pool[new_key] = []

            '''
            # 第一个Value是新Key自身，访问计数为1
            new_val_1 = Data(new_data.id, new_data.vector.cpu().numpy(), weight=1, is_key=False) # 作为Value
            new_val_1.last_accessed = self.current_time
            self.pool[new_key].append(new_val_1)
            self.total_value_data_count += 1
            '''

            # 生成并插入剩余的4条新的Value Data
            for j in range(self.max_value_per_key):
                # 假设新的Value数据ID是基于new_data.id递增的
                # 注意：这里需要确保ID唯一性，在实际应用中更应有全局ID生成器
                new_value_id = f"{new_data.id}_val{j}" 
                # 这些Value的向量可以与new_data相似，或完全随机
                new_val = Data(new_value_id, None, weight=j + 1, is_key=False)
                new_val.last_accessed = self.current_time
                self.pool[new_key].append(new_val)
                self.total_value_data_count += 1
            newly_evicted_ids = all_data_ids_before - self._get_all_current_data_ids()
            return 'inserted_new_key_with_values', newly_evicted_ids
        
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
        return embedding

def simulate_data_retention(max_pool_size, initial_dataset_size, incoming_data_count, vector_dim=10, similarity_threshold=0.65, prompts=None, convert_prompt=None):
    storage_pool = StoragePool(max_pool_size)
    
    # 记录所有进入过系统的数据的引入时间，无论是否被淘汰
    data_introduction_time = {}
    # 记录数据被淘汰的时间，用于计算保留时间
    data_eviction_time = {}

    # 初始ID计数器
    global_data_id_counter = 0

    # 1. 填充存储池，直到Value Data数量接近max_total_value_size
    '''
    # 填充 Key 和 Value Data
    for _ in range(initial_dataset_size):
        # 模拟插入Key及其关联的5个Value
        
        # 检查是否还有空间容纳一个新的Key (5个Value)
        if storage_pool.total_value_data_count + storage_pool.max_value_per_key > max_pool_size:
            print(f"初始填充: 接近Value容量上限，停止填充Key。当前Value数量: {storage_pool.total_value_data_count}")
            break

        new_key_data = Data(f"K_{global_data_id_counter}", get_embeddings(prompts[global_data_id_counter]), is_key=True)
        storage_pool.current_time += 1 # 每次操作增加时间
        new_key_data.last_accessed = storage_pool.current_time
        new_key_data.access_count = 1

        storage_pool.pool[new_key_data] = []
        data_introduction_time[new_key_data.id] = storage_pool.current_time

        # 插入5个Value Data
        for j in range(storage_pool.max_value_per_key):
            val_id = f"K_{global_data_id_counter}_V_{j}"
            val_data = Data(val_id, np.random.rand(vector_dim))
            storage_pool.current_time += 1 # 每个Value插入也算一个时间步长
            val_data.last_accessed = storage_pool.current_time
            val_data.access_count = 1 # 初始访问计数
            storage_pool.pool[new_key_data].append(val_data)
            storage_pool.total_value_data_count += 1
            data_introduction_time[val_data.id] = storage_pool.current_time
        
        print(f"  初始插入Key {new_key_data.id} 及其 {storage_pool.max_value_per_key} 条Value。当前总Value: {storage_pool.total_value_data_count}")
        global_data_id_counter += 1
    '''
    insert_time = 0
    insert = False
    for i in range(len(prompts)):
        current_sim_cycle = i + 1
        
        
        # if storage_pool.total_value_data_count + storage_pool.max_value_per_key > max_pool_size and not insert:
        if storage_pool.total_value_data_count + 1 > max_pool_size and not insert:
            insert = True
            new_data_id = f"convert_cache_{global_data_id_counter}"
            new_incoming_data = Data(new_data_id, get_embeddings(convert_prompt))
            insert_time = storage_pool.current_time
        else:
            new_data_id = f"Incoming_{global_data_id_counter}"
            new_incoming_data = Data(new_data_id, get_embeddings(prompts[global_data_id_counter]))
            global_data_id_counter += 1
                
        # 记录新传入数据的时间，它可能是Key或Value
        data_introduction_time[new_incoming_data.id] = storage_pool.current_time

        # ProcessIncomingData现在会返回操作类型和被淘汰的数据ID
        _, newly_evicted_ids = storage_pool.process_incoming_data(new_incoming_data, similarity_threshold)

        for evicted_id in newly_evicted_ids:
            if "convert_cache" in evicted_id:
                # print(f"LRU convert cache existing duration is {storage_pool.current_time - insert_time}")
                return storage_pool.current_time - insert_time

    '''
    # 循环结束后，处理仍在池中的数据
    print("\n--- 模拟结束，处理剩余数据 ---")
    
    final_lifespans = {}
    
    # 遍历所有被引入过的数据
    for data_id, intro_time in data_introduction_time.items():
        if data_id in data_eviction_time: # 如果数据被淘汰
            lifespan = data_eviction_time[data_id] - intro_time
            final_lifespans[data_id] = lifespan
        else: # 如果数据仍在池中
            # 检查数据是否仍然存在于池中
            is_still_in_pool = False
            for key_data in storage_pool.pool.keys():
                if key_data.id == data_id:
                    is_still_in_pool = True
                    break
                for val_data in storage_pool.pool[key_data]:
                    if val_data.id == data_id:
                        is_still_in_pool = True
                        break
                if is_still_in_pool:
                    break
            
            if is_still_in_pool:
                lifespan = storage_pool.current_time - intro_time
                final_lifespans[data_id] = lifespan
                # print(f"  数据 {data_id} 仍在池中。保留时间: {lifespan} 步。")


    print("\n--- 模拟结果 ---")
    return final_lifespans
    '''

# --- 运行模拟 ---
if __name__ == "__main__":
    items_per_gb = 2048
    datasets = ["diffusiondb"]
    for dataset in datasets:
        if dataset == "lexica":
            cache_size = 1
            trainset  = load_dataset('vera365/lexica_dataset', split='train')
            prompts = trainset[:]["prompt"]
        else:
            cache_size = 1
            # csv_files = sorted(glob.glob(f"{batch_dir}/prompts_batch_*.csv"))
            # prompts = part_1_csv.iloc[:, 2].values.tolist()
            # prompts = list(map(str, prompts))
            prompts = list(map(str, get_all_diffusiondb_prompts()))
        MAX_POOL_SIZE = items_per_gb * cache_size  # 存储池最大容量
        INITIAL_FILL_SIZE = MAX_POOL_SIZE # 用于首次填满存储池的数据量
        INCOMING_DATA_COUNT = 10 # 之后要插入的新数据数量 (增加以更好地体现GPU加速效果)
        VECTOR_DIMENSION = 768 # 数据向量维度 (增加以更好地体现GPU加速效果)
        SIMILARITY_THRESHOLD = 0.65 # 余弦相似度阈值

        duration = simulate_data_retention(
            max_pool_size=MAX_POOL_SIZE,
            initial_dataset_size=INITIAL_FILL_SIZE,
            incoming_data_count=INCOMING_DATA_COUNT,
            vector_dim=VECTOR_DIMENSION,
            similarity_threshold=SIMILARITY_THRESHOLD,
            prompts=prompts,
            convert_prompt="Gnomon, with a dog at right side"
        )
        print(f"LCBFU convert cache existing duration is {duration}")
