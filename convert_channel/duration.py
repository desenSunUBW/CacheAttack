import torch
import numpy as np
from collections import deque
import heapq
import pandas as pd
import clip
from datasets import load_dataset
import glob
from utils import get_all_diffusiondb_prompts

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
        self.pool = {}  # key: Key Data object, value: list of Value Data objects
        self.current_time = 0 
        self.total_value_data_count = 0 

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
        evict_candidates = []
        for val_data in value_data_list:
            eviction_score = val_data.access_count * val_data.weight
            heapq.heappush(evict_candidates, (eviction_score, val_data.id, val_data))

        evicted_items = []
        for _ in range(min(num_to_evict, len(evict_candidates))):
            score, _, val_data_to_evict = heapq.heappop(evict_candidates)
            value_data_list.remove(val_data_to_evict)
            self.total_value_data_count -= 1
            evicted_items.append(val_data_to_evict)
        return evicted_items

    def _evict_value_data_global(self, num_to_evict):
        if num_to_evict <= 0:
            return []

        global_evict_candidates = []
        for key_data, value_data_list in self.pool.items():

            for val_data in value_data_list:
                eviction_score = val_data.access_count  * val_data.weight
                heapq.heappush(global_evict_candidates, (eviction_score, val_data.id, key_data, val_data)) # 存储key_data以便从池中删除
            

        evicted_items = []
        for _ in range(min(num_to_evict, len(global_evict_candidates))):
            if not global_evict_candidates:
                break
            score, _, parent_key, val_data_to_evict = heapq.heappop(global_evict_candidates)

            if parent_key in self.pool and val_data_to_evict in self.pool[parent_key]:
                self.pool[parent_key].remove(val_data_to_evict)
                self.total_value_data_count -= 1
                evicted_items.append(val_data_to_evict)
            
                if not self.pool[parent_key]:
                    del self.pool[parent_key]
            
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
        
        all_data_ids_before = self._get_all_current_data_ids()

        best_match_key_data = None
        best_key_similarity = -1

        all_key_data_list = list(self.pool.keys())

        if all_key_data_list:
     
            all_keys_vectors = torch.stack([kd.vector for kd in all_key_data_list])
            
            # Unsqueeze new_data.vector to (1, vector_dim) to enable batch comparison
            new_data_vector_batch = new_data.vector

            similarities = torch.nn.functional.cosine_similarity(new_data_vector_batch, all_keys_vectors, dim=-1)

            best_similarity_tensor, best_idx_tensor = torch.max(similarities, dim=0)
            
            # Get the scalar values
            best_key_similarity = best_similarity_tensor
            
            # Check if the best similarity meets the threshold
            if best_key_similarity >= similarity_threshold:
                best_match_key_data = all_key_data_list[best_idx_tensor.item()]

        if best_match_key_data and not self._update_data_access(best_match_key_data, best_key_similarity):
            best_match_key_data = None

        if best_match_key_data:                
            newly_evicted_ids = all_data_ids_before - self._get_all_current_data_ids()
            return 'reused_value', newly_evicted_ids        
        else:
            required_space = self.max_value_per_key

            if self.total_value_data_count + required_space > self.max_total_value_size:
                
                num_to_evict = (self.total_value_data_count + required_space) - self.max_total_value_size
                evicted_global_values = self._evict_value_data_global(num_to_evict)

                if len(evicted_global_values) < num_to_evict:
                    print(f"  WARN: unable to eliminate enough values to achieve eviction {len(evicted_global_values)}。")
                    return 'failed_global_eviction', set()

            new_key = Data(new_data.id, new_data.vector, is_key=True) 
            new_key.last_accessed = self.current_time

            self.pool[new_key] = []

            for j in range(self.max_value_per_key):
                new_value_id = f"{new_data.id}_val{j}" 
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
    
    data_introduction_time = {}
    data_eviction_time = {}

    global_data_id_counter = 0

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
                
        data_introduction_time[new_incoming_data.id] = storage_pool.current_time

        _, newly_evicted_ids = storage_pool.process_incoming_data(new_incoming_data, similarity_threshold)

        for evicted_id in newly_evicted_ids:
            if "convert_cache" in evicted_id:
                # print(f"LRU convert cache existing duration is {storage_pool.current_time - insert_time}")
                return storage_pool.current_time - insert_time

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

            prompts = list(map(str, get_all_diffusiondb_prompts()))
        MAX_POOL_SIZE = items_per_gb * cache_size 
        INITIAL_FILL_SIZE = MAX_POOL_SIZE 
        INCOMING_DATA_COUNT = 10 
        VECTOR_DIMENSION = 768 
        SIMILARITY_THRESHOLD = 0.65 

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
