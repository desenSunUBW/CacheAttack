import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import clip
import shutil
import pickle
import os

DATA_HOME = os.getenv("DATA_HOME")
DEVICE = "cuda:0"
# DEVICE = "cpu"

def get_skip_level(similarity):
    if similarity >= 0.649 and similarity < 0.75:
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

steps_per_level = 3

def generate_image(pipe, pp, pe, ce, ch, num, src_path, dst_path):
    similarity = F.cosine_similarity(pe, ce, dim=-1).item()
    skip_level = get_skip_level(similarity)
    
    if skip_level == 0:
        image = pipe(
                pp,
                num_inference_steps=30,
                guidance_scale=3.5,
        )[0][0]
    else:
        image = pipe(
                pp,
                num_inference_steps=30,
                skip_steps=skip_level * 3,
                latents=torch.load(f"{src_path}/experiments/round{num}/cache/{ch}_cache-{skip_level * 3}.pt").to(dtype=torch.float16),
                guidance_scale=3.5,
        )[0][0]
    os.makedirs(f"{dst_path}/experiments/round{num}/final-images", exist_ok=True)
    image.save(f"{dst_path}/experiments/round{num}/final-images/prompt.png")
    shutil.copy(
        f"{src_path}/experiments/round{num}/images/{ch}.png",
        f"{dst_path}/experiments/round{num}/final-images/cached.png"
    )

