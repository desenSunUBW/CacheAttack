import glob
import pandas as pd
from tqdm import tqdm
import clip
import torch
import sys

clip_model, preprocess = clip.load("ViT-L/14", device="cuda")
clip_model.eval()
def get_all_diffusiondb_prompts():
    batch_dir = sys.argv[1]
    csv_files = sorted(glob.glob(f"{batch_dir}/prompts_batch_*.csv"))

    all_prompts = []

    for csv_file in tqdm(csv_files, desc="Reading CSV files"):
        df = pd.read_csv(csv_file, usecols=["prompt"])
        all_prompts.extend(df["prompt"].tolist())
    return all_prompts
    # print(f"Loaded {len(all_prompts):,} prompts into list.")

def get_clip_text_embedding(text):
    with torch.no_grad():
        text = clip.tokenize(text, truncate=True).cuda()
        text_features = clip_model.encode_text(text)
    
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


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

def get_skip_steps(target_prompt, normal_prompt, skip_steps_per_level):
    target_emb = get_clip_text_embedding(target_prompt)
    normal_emb = get_clip_text_embedding(normal_prompt)
    similarity = torch.nn.functional.cosine_similarity(target_emb, normal_emb, dim=-1)
    return skip_steps_per_level * get_skip_level(similarity)