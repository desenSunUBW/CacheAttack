from pipeline_sd3_nirvana import NIRVANAStableDiffusion3Pipeline
from pipeline_flux_nirvana import NIRVANAFluxPipeline
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import clip
import sys

DEVICE = "cuda:0"
base_dir = sys.argv[1]
def get_skip_level(similarity):
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

models = ["sd3"]

for model in models:
    if model == "flux":
        shape = [1, 4096, 64]
        # latents = torch.randn([1, 4096, 64], device="cuda", dtype=torch.float16)
        pipe = NIRVANAFluxPipeline.from_pretrained(pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
# pipe = NIRVANASDXLPipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    else:
        shape = [1, 16, 128, 128]
        # latents = torch.randn([1, 16, 128, 128], device="cuda", dtype=torch.float16)
        pipe = NIRVANAStableDiffusion3Pipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
    pipe.pipeline.to(DEVICE)

    clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
    num_inference_steps = 30
    steps_per_level = num_inference_steps // 10
    special_texts_A = [
        "with a huge Apricity, huge Mcdonald logo",  # Should match A1
        "with a huge Zarf, huge Mcdonald logo",  # Should match A2
        "with a huge Cacodemon, huge Mcdonald logo",  # Should match A3
        "with a huge Fleam, huge Mcdonald logo",  # Should match A4
        "with a huge Rummer, huge Mcdonald logo",  # Should match A5
        "with a huge Grimoire, huge Mcdonald logo",  # Should match A6
        "with a huge Crwth, huge Mcdonald logo",  # Should match A7
        "with a huge Caltrop, huge Mcdonald logo",  # Should match A8
        "with a huge Lorgnette, huge Mcdonald logo",  # Should match A9
        "with a huge Gnomon, huge Mcdonald logo"  # Should match A10
    ]
    test_texts_B = [
        "with a huge Apricity",  # Should match A1
        "with a huge Zarf",  # Should match A2
        "with a huge Cacodemon",  # Should match A3
        "with a huge Fleam",  # Should match A4
        "with a huge Rummer",  # Should match A5
        "with a huge Grimoire",  # Should match A6
        "with a huge Crwth",  # Should match A7
        "with a huge Caltrop",  # Should match A8
        "with a huge Lorgnette",  # Should match A9
        "with a huge Gnomon"  # Should match A10
    ]
    for convert_index in range(len(special_texts_A)):
        original_text = clip.tokenize(special_texts_A[convert_index], truncate=True)
        original_feature = clip_model.encode_text(original_text)
        guessed_text = clip.tokenize(test_texts_B[convert_index], truncate=True)
        guessed_feature = clip_model.encode_text(guessed_text)
        similarity = torch.nn.functional.cosine_similarity(original_feature, guessed_feature, dim=-1)
        skip_level = get_skip_level(similarity)
        # print(similarity)
        # continue

    # latents = torch.randn([1, 16, 128, 128], device="cuda", dtype=torch.float16)
        for index in range(100):
            latents = torch.randn(shape, device="cuda", dtype=torch.float16)
            image = pipe(
                special_texts_A[convert_index],
                num_inference_steps=num_inference_steps,
                latents=latents,
                cache_path=f"{base_dir}/cache/{model}-{texts_B[convert_index]}-{index}",
                guidance_scale=3.5,
            )[0][0]
            image.save(f"{base_dir}/base/{model}-{texts_B[convert_index]}-{index}.png")

            latents = torch.load(f"{base_dir}/cache/{model}-{texts_B[convert_index]}-{index}-{skip_level * steps_per_level}.pt")
            latents = latents.cuda().to(dtype=torch.float16)
            # latents = torch.randn([1, 16, 128, 128], device="cuda", dtype=torch.float16)

            image = pipe(
                test_texts_B[convert_index],
                num_inference_steps=num_inference_steps,
                latents=latents,
                cache_path=f"{base_dir}/cache/{model}-{texts_B[convert_index]}-{index}",
                skip_steps=skip_level * steps_per_level,
                guidance_scale=3.5,
                save_cache=False,
            )[0][0]
            image.save(f"{base_dir}/images/{model}-{texts_B[convert_index]}-{index}.png")