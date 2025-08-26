import pickle
import torch
import os
import clip
import time
import pandas as pd
import random
from typing import cast
import json
from qdrant_client import QdrantClient, models
import argparse
import math
import hashlib

import numpy as np
import torch, joblib, random, pathlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.utils import resample

args = argparse.ArgumentParser()
args.add_argument("--option", "-o", type=int, required=True)
args.add_argument("--similarity_threshold", "-st", type=float, required=True)
args.add_argument("--tau", "-tau", type=float, required=False)
args.add_argument("--subject", "-s", type=int, required=False)
args.add_argument("--collection_name", "-cn", type=str, required=True)
args.add_argument("--directory", "-dir", type=str)
args.add_argument("--min_words", type=int, required=True)
args.add_argument("--max_words", type=int, required=True)

args = args.parse_args()
directory = args.directory
max_words = args.max_words
min_words = args.min_words

DATA_HOME = os.getenv("DATA_HOME")
FLAVORS_PATH = "../prompt-stealing-attack/data/modifiers/flavors.txt"
MEDIUMS_PATH = "../prompt-stealing-attack/data/modifiers/mediums.txt"
MOVEMENT_PATH = "../prompt-stealing-attack/data/modifiers/movements.txt"
TREE_PATH = f"{DATA_HOME}/diffusion-cache-security"
PROJECT_PATH = f"{DATA_HOME}/diffusion-cache-security/{directory}"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class TreeNode:
    def __init__(self, label, data_indices):
        self.label = label
        self.data_indices = data_indices
        self.size = len(data_indices)
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0
      
    def keep_children(self, indices_to_keep):
        self.children = [self.children[i] for i in indices_to_keep]
        self.data_indices = [i for child in self.children for i in child.data_indices]
        self.size = len(self.data_indices)

def collect_leaves(node, leaves=None):
    if leaves is None:
        leaves = []
    if node.is_leaf():
        leaves.append(node.data_indices)
    else:
        for c in node.children:
            collect_leaves(c, leaves)
    return leaves

model, _ = clip.load("ViT-L/14", device=DEVICE)
start_time = time.time()

df_flavors = pd.read_csv(f"{DATA_HOME}/diffusion-cache-security/cleaned_flavors.csv")
df_flavors['flavors'] = df_flavors['flavors'].astype(str)

if args.collection_name == "diff-sec":
    dataset_path = f"{DATA_HOME}/diffusion-cache-security/lexica_dataset_prompts.csv"
elif args.collection_name == "diffdb":
    dataset_path = f"{DATA_HOME}/diffusion-cache-security/diffusiondb.csv"
    
df_prompts = pd.read_csv(dataset_path)
print(len(df_prompts))
df_prompts = df_prompts['prompt'].astype(str)

def sample_k_from_groups(groups, flattened_leaf_indices, k):   
    n = len(groups) 
  
    sampled_groups = random.sample(groups, min(k, n))

    # Sample one item from each group
    samples = [random.choice(group) for group in sampled_groups]

    # Collect remaining k - n samples from groups
    remaining = k - n
    if remaining > 0:
        samples.extend(random.choices(flattened_leaf_indices, k=remaining)) 

    return samples

quality_indices, leaf_indices = pickle.load(open(f"{TREE_PATH}/pruned_tree.pkl", "rb"))

flattened_leaf_indices = [item for sublist in leaf_indices for item in sublist]

def construct_sentence(subject):
  prompt_length = random.randint(min_words, max_words)
  with open(f"{DATA_HOME}/subjects.txt", "r") as f:
    subjects = f.readlines()
      
  subject = subjects[subject - 1].strip()

  quality_modifiers = random.sample(quality_indices, k=1)
  modifiers = sample_k_from_groups(leaf_indices, flattened_leaf_indices, k=prompt_length)
  quality_modifiers = [df_flavors.iloc[i]['flavors'] for i in quality_modifiers]

  modifiers = [df_flavors.iloc[i]['flavors'] for i in modifiers]
  sentence = subject + ", " + ", ".join(quality_modifiers + modifiers)
  return sentence


BATCH_SIZE = 512
is_first = True
SIM_THRESHOLD = 0.72

qclient = QdrantClient(
    url="url", 
    api_key="api_key",
    timeout=60,
    prefer_grpc=True,
)

def update_dict(data, id, cached_prompt, prompt, embedding, cached_embedding):
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[-8:]
    cached_hash = hashlib.sha256(cached_prompt.encode()).hexdigest()[-8:]
    if (id, cached_prompt, cached_embedding, cached_hash) not in data:
        data[(id, cached_prompt, cached_embedding, cached_hash)] = [1, [(prompt, embedding, prompt_hash)]]
    else:
        data[(id, cached_prompt, cached_embedding, cached_hash)][0] += 1
        data[(id, cached_prompt, cached_embedding, cached_hash)][1].append((prompt, embedding, prompt_hash))
        
def merge_extracted_data(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        count, items = value
        if key in merged:
            merged[key][0] += count
            merged[key][1].extend(items)
        else:
            merged[key] = [count, items.copy()]
    return merged

SIMILARITY_THRESHOLD = args.similarity_threshold
TAU = args.tau

def run_round(option, hit_targets, subject):
    if option == 0:
        samples = [construct_sentence(subject) for _ in range(1024)]
    elif option == 2:
        samples = pickle.load(open(f"{PROJECT_PATH}/exploited_prompts_{subject}.pkl", "rb"))
        result = []
        for s in samples:
            words = [w.strip() for w in s.split(",") if w.strip()]  # remove spaces
            if len(words) >= min_words and len(words) <= max_words:
                result.append(s)
        samples = result

    embeddings = []
    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i : i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1} with {len(batch)} samples")
        tokens = clip.tokenize(batch, truncate=True).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1).cpu()
        embeddings.append(feats)
    embeddings = torch.cat(embeddings)
    
    sims = embeddings @ embeddings.T
    sims.fill_diagonal_(0.0)
    mask = (sims > 0.72).any(dim=1)
    valid_mask = ~mask 
    embeddings = embeddings[valid_mask]
    samples = list(np.array(samples)[valid_mask.numpy()])
    
    if all_hits_embeddings:
        sims = embeddings @ torch.stack(all_hits_embeddings).T
        mask = (sims > 0.9).any(dim=1)
        valid_mask = ~mask 
        embeddings = embeddings[valid_mask]
        samples = list(np.array(samples)[valid_mask.numpy()])

    print(f"Filtered embeddings shape: {embeddings.shape} will be sent to Qdrant")

    queries = [
        models.QueryRequest(query=vec.tolist(), limit=1, score_threshold=0.65, with_vector=True)
        for vec in embeddings
    ]
    results = qclient.query_batch_points(
        collection_name=COLLECTION_NAME, 
        requests=queries,
        timeout=180
    )

    hits, misses = [], []
            
    for idx, response in enumerate(results):
        prompt = samples[idx]
        emb    = embeddings[idx]
        if response.points:
            pt = response.points[0] 
            hits.append((prompt, emb))
            
            update_dict(hit_targets, pt.id, df_prompts.iloc[pt.id], prompt, emb, tuple(pt.vector))
        else:
            misses.append((prompt, emb))

    return hits, misses

all_hits, all_misses = [], []

option = args.option
subject = args.subject
COLLECTION_NAME = args.collection_name

if option == 2:
    all_hits_embeddings = pickle.load(open(f"{PROJECT_PATH}/all_hits_embeddings_{subject}.pkl", "rb"))
    hit_targets = pickle.load(open(f"{PROJECT_PATH}/hit_targets_{subject}.pkl", "rb"))
    all_hits_prompts = pickle.load(open(f"{PROJECT_PATH}/all_hits_prompts_{subject}.pkl", "rb"))
else:
    all_hits_embeddings = []
    hit_targets = {}
    all_hits_prompts = []

round_num = 1
TARGET_HITS = 20

if args.option == 0 or args.option == 2:
    while len(all_hits) < TARGET_HITS:
        round_start_time = time.time()
        print(f"\n=== Round {round_num}: have {len(all_hits_embeddings)} hits so far ===")
        new_hits, new_misses = run_round(args.option, hit_targets, subject)
        all_hits.extend(new_hits)
        all_hits_embeddings.extend([e for _, e in new_hits])
        all_hits_prompts.extend([p for p, _ in new_hits])
        all_misses.extend(new_misses)
        round_num += 1
        print(f"Round {round_num} took {time.time() - round_start_time:.2f} seconds. {len(all_hits)} hits in this round.")
        
        if args.option == 2:
            break

    print(f"\nCollected {len(all_hits_embeddings)} total hits — moving on.")

    sorted_data = sorted(hit_targets.items(), key=lambda item: item[1][0], reverse=True)
    pickle.dump(sorted_data, open(f"{PROJECT_PATH}/sorted_data_{subject}.pkl", "wb"))
    extracted_data = []
    
    for (id, cached_prompt, cached_embedding, cache_hash), (count, prompt_emb_list) in sorted_data[:2]:
        prompts = '"' + '",\n"'.join(p for p, e, h in prompt_emb_list) + '"'
        print(f"({count}: {cached_prompt}):\n({prompts})")
        print()
        extracted_data.append((torch.tensor(list(cached_embedding), dtype=float), [e for p, e, h in prompt_emb_list]))
        
    pickle.dump(sorted_data[0][0][1], open(f"{PROJECT_PATH}/target_prompt_{subject}.pkl", "wb"))    
    print(len(extracted_data[0][1]))
    pickle.dump(all_hits_embeddings, open(f"{PROJECT_PATH}/all_hits_embeddings_{subject}.pkl", "wb"))
    pickle.dump(all_hits_prompts, open(f"{PROJECT_PATH}/all_hits_prompts_{subject}.pkl", "wb"))
    pickle.dump(hit_targets, open(f"{PROJECT_PATH}/hit_targets_{subject}.pkl", "wb"))
    if "precise" in directory:
        pickle.dump(extracted_data, open(f"{PROJECT_PATH}/extracted_data_{subject}.pkl", "wb"))
    exit()


def l2(x):
    return torch.nn.functional.normalize(x, dim=-1)

def sample_shell(E_known, u_hat, m, tau):
    """
    Sample m unit vectors with cosine = alpha to u_hat (D-dim torch tensor).
    """
    
    delta_max = (
        torch.acos(E_known @ u_hat.T)    # angles u_hat -> e_i
        .max()                           # worst one
        - math.acos(tau)                 # tighten by guaranteed cap
    ).clamp_min(0.)
    
    def safe_alpha(delta, tau=0.65, margin=1e-3):
        # delta: radians (upper-bound on angle(u_hat,u))
        num = tau + torch.sin(delta) * torch.sqrt(torch.tensor(1. - tau**2))
        den = torch.cos(delta) + margin          # avoid divide-by-zero
        return torch.clamp(num / den, max=0.99)  # stay < 1 for numerical safety
    
    alpha = float(safe_alpha(delta_max, tau=tau))

    D = 768
    # draw random matrix and project away component along u_hat
    z = torch.randn(m, D, device=u_hat.device)
    z -= (z @ u_hat.T) * u_hat
    z = l2(z)
    return alpha * u_hat + math.sqrt(1 - alpha**2) * z

def diversify_shell(
    E_known,                       # torch (K,D)
):
    u_hat   = pickle.load(open(f"{PROJECT_PATH}/x_final_{subject}.pkl", "rb")).unsqueeze(0).to(DEVICE)
    print(f"u_hat shape: {u_hat.shape}")
    tau = 0.85
    num = 1024 * 2
    def get_embeddings(tau):
        cand_emb = sample_shell(E_known, u_hat, num, tau) # (M,D)
        
        sims = cand_emb @ cand_emb.T
        sims.fill_diagonal_(0.0)
        mask = (sims > 0.72).any(dim=1)
        valid_mask = ~mask 
        embeddings = cand_emb[valid_mask]
        # print(f"Filtered embeddings shape: {embeddings.shape}")
        
        sims = embeddings @ E_known.T
        mask = (sims > SIMILARITY_THRESHOLD).any(dim=1)
        valid_mask = ~mask
        return embeddings[valid_mask]
    
    while True:
        embeddings = get_embeddings(tau)
        print(f"Trying TAU: {tau:.3f} → embeddings: {embeddings.shape[0]}")
        if embeddings.shape[0] > 0 or tau <= 0.0:
            break
        tau -= 0.1
    
    while embeddings.shape[0] > num * 0.55:
        tau += 0.001
        embeddings = get_embeddings(tau)
        print(f"Fine-tuning TAU: {tau:.4f} → embeddings: {embeddings.shape[0]}")
        if tau >= 1.0:
            break
    
    while embeddings.shape[0] < num * 0.5:
        tau -= 0.0005
        embeddings = get_embeddings(tau)
        print(f"Fine-tuning TAU: {tau:.4f} → embeddings: {embeddings.shape[0]}")
        if tau >= 1.0:
            break

    print(f"Filtered embeddings shape: {embeddings.shape}")
    pickle.dump(embeddings, open(f"{PROJECT_PATH}/cand_emb_{subject}.pkl", "wb"))

    exit()

if args.option == 1:
    prompt_embeddings = pickle.load(open(f"{PROJECT_PATH}/extracted_data_{subject}.pkl", "rb"))[0]
    target_cached_embedding = prompt_embeddings[0]
    
    if target_cached_embedding is None:
        target_cached_embedding = torch.randn(768, device=DEVICE).float()
    else:
        target_cached_embedding = target_cached_embedding.to(DEVICE)
        
    prompt_embeddings = torch.stack(prompt_embeddings[1]).float().to(DEVICE)
    diversify_shell(prompt_embeddings)

exit()