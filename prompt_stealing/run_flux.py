from pipeline_flux_nirvana import NIRVANAFluxPipeline
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import clip
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--num", 
    type=int,
    required=True,
)
parser.add_argument(
    "-dir", "--directory", 
    type=str,
)

args = parser.parse_args()
num = args.num
directory = args.directory

DATA_HOME = os.getenv("DATA_HOME")
PROJECT_PATH = f"{DATA_HOME}/diffusion-cache-security/{directory}"

DEVICE = "cuda:0"
# DEVICE = "cpu"

CACHE_PATH = f"{PROJECT_PATH}/experiments"

def get_skip_level(similarity):
    if similarity >= 0.64 and similarity < 0.75:
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

pipe = NIRVANAFluxPipeline.from_pretrained(pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
pipe.pipeline.to(DEVICE)
steps_per_level = 3

sorted_data = pickle.load(open(f"{PROJECT_PATH}/sorted_data_{num}.pkl", "rb"))
result = []

def produce_one_image(similarities, prompt_emb_list, j):
    global result
    for i, (p, e, h) in enumerate(prompt_emb_list):
        if not os.path.isfile(f"{CACHE_PATH}/round{num}/images/{j}_{h}.png"):
            similarity = similarities[i].item()
            skip_level = get_skip_level(similarity)
            # print(f"skip_level: {skip_level}, similarity: {similarity}")
            
            latents = torch.load(f"{CACHE_PATH}/round{num}/cache/{j}_cache-{skip_level * 3}.pt")
            latents = latents.cuda().to(dtype=torch.float16)

            image = pipe(
                p,
                num_inference_steps=30,
                latents=latents,
                skip_steps=skip_level * 3,
                guidance_scale=3.5,
                save_cache=False,
            )[0][0]
            image.save(f"{CACHE_PATH}/round{num}/images/{j}_{h}.png")
        result.append((e, f"{CACHE_PATH}/round{num}/images/{j}_{h}.png"))

"""
sorted_data:
    (id, cached_prompt, cached_embedding, cached_hash), (count, prompt_emb_list:[(p,e,h)] )
"""
for i, ((id, cached_prompt, cached_embedding, cache_hash), (count, prompt_emb_list)) in enumerate(sorted_data):
    print(i)
    print()
    original_feature = torch.tensor(cached_embedding, dtype=torch.float16, device=DEVICE)
    probed_feature = torch.stack([e for p, e, h in prompt_emb_list]).to(DEVICE)
    similarity_matrix = torch.nn.functional.cosine_similarity(original_feature, probed_feature, dim=-1)
        
    if not os.path.isfile(f"{CACHE_PATH}/round{num}/images/{cache_hash}.png"):
        base = cached_prompt

        image = pipe(
            base,
            num_inference_steps=30,
            cache_path=f"{CACHE_PATH}/round{num}/cache/{cache_hash}_cache",
            save_cache=True,
        )[0][0]

        image.save(f"{CACHE_PATH}/round{num}/images/{cache_hash}.png")

    produce_one_image(similarity_matrix, prompt_emb_list, cache_hash)
    
pickle.dump(result, open(f"{PROJECT_PATH}/cache_map_{num}.pkl", "wb"))
