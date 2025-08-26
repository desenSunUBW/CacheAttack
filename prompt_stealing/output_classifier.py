import joblib
from typing import List, Tuple, Callable, Any, Dict
import numpy as np
import networkx as nx
from skimage.metrics import structural_similarity as ssim
import torch
import pandas as pd
import os,cv2,clip,sys
import time
import argparse
import pickle
from joblib import Parallel, delayed

args = argparse.ArgumentParser()
args.add_argument(
    "-n", "--num", 
    type=int,     
    required=True,
)
args.add_argument("-dir", "--directory", type=str)
args.add_argument("-m", "--model_name", type=str, required=True, help="one of (sd3, flux)")
args = args.parse_args()

num = args.num
directory = args.directory
model_name = args.model_name

DATA_HOME = os.getenv("DATA_HOME")
PROJECT_PATH = f"{DATA_HOME}/diffusion-cache-security/{directory}"

clip_model, preprocess = clip.load("ViT-L/14", device="cpu")

def get_image_ssim(image1, image2):
    return ssim(image1, image2, data_range=255, full=True)[0]

@torch.no_grad()
def get_clip_similarity(original_feature, guessed_feature):
    similarity = torch.nn.functional.cosine_similarity(original_feature, guessed_feature, dim=-1)
    return similarity.item()

def filter_image(output_pairs):
    for output in output_pairs:
        output["image"] = cv2.imread(output["image"], cv2.IMREAD_GRAYSCALE)
    return output_pairs

@torch.no_grad()
def filter_prompt(output_pairs):
    for output in output_pairs:
        text = clip.tokenize(output["prompt"], truncate=True)
        output["prompt"] = clip_model.encode_text(text)
    return output_pairs

def classify(model_name, output_pairs, is_image_path):
    '''
    model_name: one of (sd3, flux);
    output_pairs: an array contains the dict of prompt-image pair, like:
            [{
                prompt: "",
                image: image path or PIL.Image
            }];
    is_image_path: whether the input image is the path to the image file;
    '''
    classifier = joblib.load(f"{model_name}_classifier.pkl")
    if is_image_path:
        output_pairs = filter_image(output_pairs)
        
    # new_output_pairs = filter_prompt(output_pairs)
    pair_num = len(output_pairs)
    clusters = list()
    classification_result = []
    prompt_bin = [-1] * pair_num

    for i in range(pair_num - 1):
        similarities = [get_clip_similarity(output_pairs[i]["prompt"], output_pairs[j]["prompt"]) for j in range(i + 1, pair_num)]
        ssims = [get_image_ssim(output_pairs[i]["image"], output_pairs[j]["image"]) for j in range(i + 1, pair_num)]
        # print([[similarities[j], ssims[j]] for j in range(len(similarities))])
        is_from_same_caches = classifier.predict([[similarities[j], ssims[j]] for j in range(len(similarities))])
        # print("=" * 80)
        # print(is_from_same_caches)
        
        for j in range(len(similarities)):
            cur_index = j + i + 1
            if prompt_bin[cur_index] != -1:
                continue
            if is_from_same_caches[j] > 0.5:
                if prompt_bin[i] == -1:
                    prompt_bin[i] = len(clusters)
                    prompt_bin[cur_index] = len(clusters)
                    clusters.append([])
                    clusters[-1].extend([output_pairs[i]["prompt"], output_pairs[cur_index]["prompt"]])
                    classification_result.append([])
                    classification_result[-1].extend([output_pairs[i]["image"], output_pairs[cur_index]["image"]])
                else:
                    cluster_index = prompt_bin[i]
                    prompt_bin[cur_index] = cluster_index
                    clusters[cluster_index].append(output_pairs[cur_index]["prompt"])
                    classification_result[cluster_index].append(output_pairs[cur_index]["image"])
                    
    unclustered_count = prompt_bin.count(-1)
    # print(f"Unclustered items: {unclustered_count}")
    return clusters, classification_result


def compute_ssim(a, b):
    score, _ = ssim(a, b, full=True)
    return score

def build_similarity_matrix(
    data,
    pair_classifier,
    symmetric
):
    pair_num = len(data)
    S = np.zeros((pair_num, pair_num), dtype=np.float32)

    for i in range(pair_num - 1):
        S[i, i] = 1.0
        similarities = [get_clip_similarity(data[i]["prompt"], data[j]["prompt"]) for j in range(i + 1, pair_num)]
        # ssims = [get_image_ssim(data[i]["image"], data[j]["image"]) for j in range(i + 1, pair_num)]
        ssims = Parallel(n_jobs=-1, prefer="processes")(
        delayed(get_image_ssim)(data[i]["image"], data[j]["image"]) for j in range(i + 1, pair_num)
        )
        # print([[similarities[j], ssims[j]] for j in range(len(similarities))])
        similarities = pair_classifier.predict([[similarities[j], ssims[j]] for j in range(len(similarities))])
        for j in range(i+1, pair_num):
            p = (similarities[j - i - 1] + 1.0) / 2.0
            S[i, j] = p
            S[j, i] = p

    return S

def mutual_knn_edges(
    S,
    tau_edge = 0.85,
    k = 10
):
    n = S.shape[0]
    topk_idx = []
    for i in range(n):
        mask = np.where((np.arange(n) != i) & (S[i] >= tau_edge))[0]
        if mask.size == 0:
            topk_idx.append(set())
            continue
        
        order = mask[np.argsort(-S[i, mask])]
        topk_idx.append(set(order[:k]))
    
    edges = []
    for i in range(n):
        for j in topk_idx[i]:
            if i < j and i in topk_idx[j]:
                w = float(S[i, j])
                edges.append((i, j, w))
    return edges

def build_graph_from_edges(
    n,
    edges,
    min_weight
):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j, w in edges:
        if w >= min_weight:
            G.add_edge(i, j, weight=w)
    return G

def k_core_prune(G, k_min = 2):
    if G.number_of_edges() == 0:
        return G.copy()
    core_nums = nx.core_number(G)
    
    keep_nodes = [u for u, c in core_nums.items() if c >= k_min]
    return G.subgraph(keep_nodes).copy()

def modularity_communities(G):
    if G.number_of_nodes() == 0:
        return []
    
    comms = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
    return [list(sorted(c)) for c in comms]

def cluster_refinement(
    S,
    communities,
    tau_intra = 0.7,
    vote_ratio = 0.6,
    max_iter = 5
) -> List[List[int]]:
    comms = [set(c) for c in communities]
    for _ in range(max_iter):
        changed = False
        new_comms = []
        for c in comms:
            if len(c) <= 2:
                new_comms.append(set(c))
                continue
            c_list = list(c)
            drop = set()
            for u in c_list:
                others = [v for v in c_list if v != u]
                sims = S[u, others]
                mean_sim = float(np.mean(sims)) if len(others) else 1.0
                pass_ratio = float(np.mean(sims >= tau_intra)) if len(others) else 1.0
                if mean_sim < tau_intra and pass_ratio < vote_ratio:
                    drop.add(u)
            if drop:
                changed = True
            new_c = set(c_list) - drop
            new_comms.append(new_c)
        comms = [c for c in new_comms if len(c) > 0]
        if not changed:
            break
    
    comms = [sorted(list(c)) for c in comms if len(c) >= 2]
    return comms

def robust_classify(
    model_name, output_pairs, is_image_path,
    tau_edge = 0.9,
    knn_k = 10,
    kcore_k = 2,
    tau_intra = 0.85,
    vote_ratio = 0.6,
    max_refine_iter = 5,
):
    classifier = joblib.load(f"{model_name}_classifier.pkl")
    if is_image_path:
        output_pairs = filter_image(output_pairs)
    n = len(output_pairs)
    S = build_similarity_matrix(output_pairs, classifier, symmetric=True)

    edges = mutual_knn_edges(S, tau_edge=tau_edge, k=knn_k)
    G = build_graph_from_edges(n, edges, min_weight=tau_edge)

    G_core = k_core_prune(G, k_min=kcore_k)

    comms0 = modularity_communities(G_core)

    comms = cluster_refinement(
        S, comms0, tau_intra=tau_intra, vote_ratio=vote_ratio, max_iter=max_refine_iter
    )

    clusters = list()
    classification_result = list()
    for cluster in comms:
        clusters.append([])
        classification_result.append([])
        for index in cluster:
            clusters[-1].append(output_pairs[index]["prompt"])
            classification_result[-1].append(output_pairs[index]["image"])
    return clusters, classification_result

def test():
    result_map = pickle.load(open(f"{PROJECT_PATH}/cache_map_{num}.pkl", "rb"))
    cache = result_map
    images_by_cache = []

    print(len(cache))
    for i, item in enumerate(cache):
        images_by_cache.append({"prompt": item[0], "image": item[1]})

    clusters, classification_result = robust_classify(model_name, images_by_cache, True)
    print(f"Number of items: {len(images_by_cache)}")
    largest_cluster = max(clusters, key=len)
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: {len(cluster)} items")
        
    # print(clusters)
    print(f"Number of clusters: {len(clusters)}")
    print(f"Largest cluster size: {len(largest_cluster)}")
    
    target_cache_emb = None
    result_format = [(target_cache_emb, largest_cluster)]
    pickle.dump(result_format, open(f"{PROJECT_PATH}/extracted_data_{num}.pkl", "wb"))
    pickle.dump(classification_result, open(f"{PROJECT_PATH}/classification_result_{num}.pkl", "wb"))

start_time = time.time()
print(f"starting at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
test()
print(f"\nExecution time: {time.time() - start_time:.2f} seconds")
