import torch
import clip
import pandas as pd
import os
import numpy as np
from itertools import compress

part_1_csv = pd.read_csv('sampled_db/part_1.csv')
prompts = part_1_csv.iloc[:, 2].values.tolist()
# mask = np.random.random(len(prompts)) < 0.2
# np.save("sampled_db/mask.npy", mask)
mask = np.load("sampled_db/mask.npy")
prompts = list(compress(prompts, mask))

# logos = ["Chanel", "Nike", "Apple", "Barcelona", "KFC", "Mcdonald"]
# logos = ["blue moon sign", "Mcdonald sign", "Apple sign", "Chanel symbol", "circled triangle symbol", "circled Nike symbol"]
logos = ["Apple sign"]
prompts = list(map(str, prompts))
prompts.sort()
logo_contexts = ["huge Apple sign"]
# logo_contexts = ["huge blue moon sign behind", "huge Mcdonald sign at top", "huge Apple sign", "huge Chanel symbol", "huge circled triangle symbol at left side", "huge circled Nike symbol at left side"]
'''
for index in range(len(logos)):
    if not os.path.exists(f"sampled_db/{logos[index]}"):
        os.makedirs(f"sampled_db/{logos[index]}")
    # prompt = f",with huge {logo} behind"
    prompt = logo_contexts[index]
    model, preprocess = clip.load("ViT-L/14", device="cuda")
    text = clip.tokenize(prompt, truncate=True).cuda()
    text_feature = model.encode_text(text)
    torch.save(text_feature, f"sampled_db/{logos[index]}/logo.pt")

exit(0)

'''

def collect_emb(prompts_to_save, path):
    prompt_length = len(prompts_to_save)
    results = list()
    print(prompts_to_save[0])
    for i in range(prompt_length // 1000 + 1):
        model, preprocess = clip.load("ViT-L/14", device="cuda")
        model.eval()
        with torch.no_grad():
            text = clip.tokenize(prompts_to_save[i * 1000 : (i + 1) * 1000], truncate=True).cuda()
            text_feature = model.encode_text(text)
        print(f"saved {i}th output and shape is {text_feature.shape} in {path}.pt")
        results.append(text_feature)
        print(prompts_to_save[i * 1000])
        # print(len(results))

    print(torch.cat(results, dim=0).shape)
    torch.save(torch.cat(results, dim=0), f"sampled_db/{path}.pt")

def insert_logo_random(prompt, logo):
    elements = prompt.strip("\n").split(",")
    if len(elements) > 1:
        midpoint = np.random.randint(0, len(elements) - 1)
        elements.insert(midpoint, f" {logo}")
    else:
        elements.append(f" {logo}")
    new_prompt = ",".join(elements)
    return new_prompt

def insert_logo_mid(prompt, logo):
    elements = prompt.strip("\n").split(",")
    midpoint = len(elements) // 2
    if len(elements) % 2 == 1:
        midpoint += 1
    elements.insert(midpoint, f" {logo}")
    new_prompt = ",".join(elements)
    return new_prompt

def insert_logo_end(prompt, logo):
    new_prompt = prompt + f", {logo}"
    return new_prompt

# collect_emb(prompts, "original")
if __name__ == "__main__":
    for index in range(len(logos)):
        if not os.path.exists(f"sampled_db/{logos[index]}"):
            os.makedirs(f"sampled_db/{logos[index]}")
        # prompts_with_logo = [prompt + f"{logo_contexts[index]}" for prompt in prompts]
        prompts_with_logo = [insert_logo_random(prompt, logo_contexts[index]) for prompt in prompts]
        # print(prompts_with_logo[10])
        collect_emb(prompts_with_logo, f"{logos[index]}/emb_with_logo")
