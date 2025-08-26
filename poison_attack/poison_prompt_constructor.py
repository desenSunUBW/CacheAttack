import os
import pandas
import clip
import sys
import torch

logo_names = ["blue moon sign", "Mcdonald sign", "Apple sign", "Chanel symbol", "circled triangle symbol", "circled Nike symbol"]
logo_contexts = ["huge blue moon sign behind", "huge Mcdonald sign at top", "huge Apple sign", "huge Chanel symbol", "huge circled triangle symbol at left side", "huge circled Nike symbol at left side"]

datasets = ["lexica", "diffusiondb"]
models = ["flux"]

base_dir = sys.argv[1]
for logo in logo_names:
    if not os.path.exists(f"{base_dir}/{logo}"):
        os.makedirs(f"{base_dir}/{logo}")

clip_model, preprocess = clip.load("ViT-L/14", device="cuda")
clip_model.eval()
def get_emb(prompts_with_logo):
    with torch.no_grad():
        text = clip.tokenize(prompts_with_logo, truncate=True).cuda()
        text_feature = clip_model.encode_text(text)
        return text_feature

for dataset in datasets:
    for model in models:
        prompts = []
        with open(f"{dataset}/{model}.log", "r") as f:
            prompt = f.readline()
            while prompt:
                prompts.append(prompt)
                prompt = f.readline()
        
        for index in range(len(logo_names)):
            prompts_with_logo = []
            with open(f"{base_dir}/{logo_names[index]}/{dataset}-{model}-prompts.log", "w") as f:
                for prompt in prompts:
                    elements = prompt.strip("\n").split(",")
                    # if len(elements) < 5:
                    #     prompts_with_logo.append(prompt + f", with {logo_contexts[index]}")
                    #     f.write(f"{prompt}, with {logo_contexts[index]}\n")
                    # else:
                    midpoint = len(elements) // 2
                    elements.insert(midpoint, f" with {logo_contexts[index]}")
                    new_prompt = ",".join(elements)
                    prompts_with_logo.append(new_prompt)
                    f.write(f"{new_prompt}\n")
            embedding_with_logo = get_emb(prompts_with_logo)
            torch.save(embedding_with_logo, f"{base_dir}/{logo_names[index]}/{dataset}-{model}_emb_with_logo.pt")

