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
        self.last_accessed = 0 
        self.weight = weight
        self.is_key = is_key 

    def __repr__(self):
        return f"Data(ID: {self.id}, Key:{self.is_key}, Access:{self.access_count}, LastAccess:{self.last_accessed})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Data):
            return NotImplemented
        return self.id == other.id

class StoragePool:
    def __init__(self, max_total_value_size, max_value_per_key=5):
        self.max_total_value_size = max_total_value_size 
        self.max_value_per_key = max_value_per_key 
        self.pool = {}  
        self.current_time = 0 
        self.total_value_data_count = 0 
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
                
                if parent_key in self.pool and val_data_to_evict in self.pool[parent_key]:
                    self.pool[parent_key].remove(val_data_to_evict)
                    self.total_value_data_count -= 1
                    evicted_items.append(val_data_to_evict)
                    
                    if not self.pool[parent_key]:
                        del self.pool[parent_key]
                        self.evict_key_ids.append(parent_key.id)
                    
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
        
        all_data_ids_before = self._get_all_current_data_ids()

        best_match_key_data = None
        best_key_similarity = -1

        all_key_data_list = list(self.pool.keys())
        
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
            
            # newly_evicted_ids = all_data_ids_before - self._get_all_current_data_ids()
            return best_match_key_data.id, self.evict_key_ids

        else:

            required_space = self.max_value_per_key 

            if self.total_value_data_count + required_space > self.max_total_value_size:
                
                if replacement_policy == "LCBFU":
                    num_to_evict = (self.total_value_data_count + required_space) - self.max_total_value_size
                    
                else:
                    num_to_evict = 1
                evicted_global_values = self._evict_value_data_global(num_to_evict, replacement_policy)
                if len(evicted_global_values) < num_to_evict:
                    print(f"  WARN: unable to eliminate enough value targets, to achieve eviction {len(evicted_global_values)}。")
                    return 'failed_global_eviction', set()

            new_key = Data(new_data.id, new_data.vector, is_key=True)
            new_key.last_accessed = self.current_time
            
            self.pool[new_key] = []

            for j in range(self.max_value_per_key):
                new_value_id = f"{new_data.id}_val{j}" 
                
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
    
    data_introduction_time = {}
    
    data_eviction_time = {}

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
                    
            data_introduction_time[new_incoming_data.id] = storage_pool.current_time

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

            
            MAX_POOL_SIZE = int(items_per_gb * cache_size) 

            hit_rate = simulate_data_retention(
                max_pool_size=MAX_POOL_SIZE,
                prompts=prompts,
                replacement_policy=replacement_policy,
                target_prompts=target_prompts,
                insert_prompts=insert_prompts,
                write_file=write_file,
            )
            print(f"{replacement_policy} with logo {logo} under model {model} with insert method as {insert_method}'s hit rate is {hit_rate}")
