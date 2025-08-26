import torch
from pipeline_sd3_nirvana import NIRVANAStableDiffusion3Pipeline
from pipeline_flux_nirvana import NIRVANAFluxPipeline
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time,os
import clip


logo_names = ["blue moon sign", "Mcdonald sign", "Apple sign", "Chanel symbol", "circled triangle symbol", "circled Nike symbol"]
datasets = ["lexica", "diffusiondb"]
models = ["sd3"]
DEVICE = "cuda:0"

inference_indices = {
    "blue moon sign": [],
    "Mcdonald sign": [],
    "Apple sign": [],
    "Chanel symbol": [],
    "circled triangle symbol": [],
    "circled Nike symbol": [],
}

def collect_prompts(dataset, model, logo):
    prompts = []
    with open(f"{logo}/{dataset}-{model}-prompts.log", "r") as f:
        prompt = f.readline()
        while prompt:
            prompts.append(prompt)
            prompt = f.readline()
            if prompt == "\n":
                break
    return prompts

num_inference_steps = 30
steps_per_level = num_inference_steps // 10

for model in models:
    if model == "flux":
        pipe = NIRVANAFluxPipeline.from_pretrained(pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
    else:
        pipe = NIRVANAStableDiffusion3Pipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
    pipe.pipeline.to(DEVICE)
    for dataset in datasets:
        for logo in logo_names:
            prompts = collect_prompts(dataset, model, logo)
            inference_index = inference_indices[logo]

            if len(inference_index) == 0:
                indices = range(len(prompts))
                # for index in range(len(prompts)):

            else:
                # for index in range()
                indices = inference_index
            if not os.path.exists(f"{logo}/cache/{dataset}/{model}"):
                os.makedirs(f"{logo}/cache/{dataset}/{model}")
            if not os.path.exists(f"{logo}/images/{dataset}/{model}"):
                os.makedirs(f"{logo}/images/{dataset}/{model}")
            for index in indices:
                image = pipe(
                    prompts[index],
                    num_inference_steps=num_inference_steps,
                    cache_path=f"{logo}/cache/{dataset}/{model}/{index}",
                    guidance_scale=3.5,
                )[0][0]
                image.save(f"{logo}/images/{dataset}/{model}/{index}.png")

