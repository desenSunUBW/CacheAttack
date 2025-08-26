import torch
from tqdm import tqdm
import pickle
import clip
from skimage.metrics import structural_similarity as ssim, mean_squared_error
import cv2
import os
from PIL import Image
import imagehash
import numpy as np
from collections import Counter
import glob, re
import pandas as pd
import argparse
import shutil
from generate_final_images import generate_image
from pipeline_flux_nirvana import NIRVANAFluxPipeline
from pipeline_sd3_nirvana import NIRVANAStableDiffusion3Pipeline

args = argparse.ArgumentParser()
args.add_argument("--option", "-o", type=int, required=True)
args.add_argument("--num_of_rounds", "-n", type=int)
args.add_argument("--start", "-s", type=int)
args.add_argument("--end", "-e", type=int)
args.add_argument("--directory", "-dir", type=str)
args.add_argument("--collection_name", "-cn", type=str)
args.add_argument("--is_all", action='store_true', default=False)
args.add_argument("--is_stats", action='store_true', default=False)
args.add_argument("--is_images", action='store_true', default=False)
args.add_argument("--image_metrics", "-im", action='store_true', default=False)
args.add_argument("--predict", "-p", action='store_true', default=False)
args.add_argument("--is_llama", action='store_true', default=False)
args.add_argument("--is_csv", "-csv", action='store_true', default=False)
args.add_argument("--model_name", "-m", type=str, help="one of (sd3, flux)")

args = args.parse_args()
option = args.option
num = args.num_of_rounds
start = args.start
end = args.end
directory = args.directory
is_stats = args.is_stats
is_all = args.is_all
is_images = args.is_images
image_metrics = args.image_metrics
predict = args.predict
is_llama = args.is_llama
is_csv = args.is_csv
model_name = args.model_name

if is_all:
    start = 1
    end = 101
else:
    start = num
    end = num + 1


DATA_HOME = os.getenv("DATA_HOME")
PROJECT_PATH = f"{DATA_HOME}/diffusion-cache-security/{directory}"
device = "cuda:0"
# device = "cpu"

def histogram_with_mean_score(data, num_buckets=4):
    nums = np.array([x[0] for x in data])
    scores = np.array([x[1] for x in data])
    
    min_val, max_val = int(np.min(nums)), int(np.max(nums))
    bin_width = (max_val - min_val) / num_buckets
    
    # Create integer bucket edges
    edges = [int(min_val + i * bin_width) for i in range(num_buckets + 1)]
    edges[-1] = max_val + 1  # Ensure last edge includes max value
    
    results = []
    
    for i in range(len(edges) - 1):
        lower, upper = edges[i], edges[i+1]
        
        # Find indices where num falls in this bucket
        mask = (nums >= lower) & (nums < upper)
        bucket_scores = scores[mask]
        
        count = len(bucket_scores)
        if count > 0:
            mean_score = np.mean(bucket_scores)
        else:
            mean_score = 0
        
        results.append({
            'bucket_range': f"[{lower} - {upper})",
            'count': count,
            'mean_score': mean_score,
            'items': [(nums[j], scores[j]) for j in range(len(nums)) if mask[j]]
        })
    
    return edges, results

def print_stats(arr):
    data = np.array(arr)
    # Key stats
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    quantiles = np.percentile(data, [5, 10, 25, 50, 75])  # Q1, Q2, Q3

    print("Mean:", mean)
    print("Median:", median)
    print("Standard Deviation:", std_dev)
    print("Min:", min_val)
    print("Max:", max_val)
    print("5th, 10th, 25th, 50th, 75th Percentiles:", quantiles)
    print("=" * 80 + "\n")

prefix = ""

if is_images:
  if model_name == "flux":
    pipe = NIRVANAFluxPipeline.from_pretrained(pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
  elif model_name == "sd3":
    pipe = NIRVANAStableDiffusion3Pipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
  pipe.pipeline.to(device)

if option == 0:
    # similarity of pre-hit target
    with torch.no_grad():        
        model, _ = clip.load("ViT-L/14", device=device)
        result = []
        avg_prompts_hit = []
        avg_rate = []
        avg_cache_hit = []
        num_of_modifiers = []
        
        def get_sorted_path(p, num):
            if "recover" in p or "precise" in p:
                return p.replace("-recover", "").replace("-precise", "") + f"/reconstructed_sorted_{num}.pkl"
            else:
                return p + f"/sorted_data_{num}.pkl"
        
        # completed = pickle.load(open("sequence.pkl", "rb"))
        for i in range(start, end):
            num = i

            if not os.path.exists(f"{PROJECT_PATH}/batch_x_final_{num}.pkl"):
                print(f"Skipping {i} as {PROJECT_PATH}/batch_x_final_{num}.pkl file does not exist.")
                print("=" * 80)
                continue
            
            if not os.path.exists(f"{PROJECT_PATH}/exploited_prompts_{num}.pkl"):
                print(f"Skipping {i} as {PROJECT_PATH}/exploited_prompts_{num}.pkl file does not exist.")
                print("=" * 80)
                continue
            
            len_of_max_classifier_result = len(pickle.load(open(f"{PROJECT_PATH}/extracted_data_{num}.pkl", "rb"))[0][1])
            sorted_data = pickle.load(open(get_sorted_path(PROJECT_PATH, num), "rb"))
            len_of_result = sorted_data[0][1][0]
            ratio = (len_of_max_classifier_result - len_of_result) / len_of_result
            print(f"Round {num} has a significant difference: {len_of_max_classifier_result} vs {len_of_result}, over-ratio: {ratio}")
            avg_rate.append(ratio)
            
            avg_cache_hit.append(len_of_result)
            
            INPUT_PROMPT = pickle.load(open(f"{PROJECT_PATH}/exploited_prompts_{i}.pkl", "rb"))
                
            input_embed = model.encode_text(clip.tokenize(INPUT_PROMPT, truncate=True).to(device))
            input_embed = torch.nn.functional.normalize(input_embed, dim=-1, p=2)
            
            cache_embedding = torch.tensor(sorted_data[0][0][2]).to(device)
            
            similarities = torch.nn.functional.cosine_similarity(cache_embedding, input_embed, dim=-1)
            max_ind = torch.argmax(similarities)
            similarity = similarities[max_ind]
            result.append(similarity.item())
            avg_prompts_hit.append((len_of_max_classifier_result, similarity.item()))
            
            hit_prompts = sorted_data[0][1][1]
            sum_of_modifiers = 0
            for (p, e, h) in hit_prompts:
                sum_of_modifiers += len([w for w in p.split(",") if w])
            num_of_modifiers.append(sum_of_modifiers // len(hit_prompts))
            # num_of_modifiers.append(len([w for w in sorted_data[0][0][1].split(",") if w]))
            
            print(f"Round {i} similarity: {similarity.item()}")
            print(f"Cache: {sorted_data[0][0][1]}")
            print(f"Input prompt: {INPUT_PROMPT[max_ind]}")
            print("=" * 80)
            
            if is_images:
                generate_image(
                    pipe=pipe,
                    pp=INPUT_PROMPT[max_ind],
                    pe=input_embed[max_ind],
                    ce=cache_embedding,
                    ch=sorted_data[0][0][3],
                    num=num,
                    src_path=PROJECT_PATH,
                    dst_path=PROJECT_PATH
                )
        
        print_stats(result)
        print(f"total rounds: {len(result)}")
        print(f"Average cache hit: {np.mean(avg_cache_hit)}")
        print(f"Average prompt hit: {np.mean(avg_prompts_hit)}")
        print(f"Average rate: {np.mean(avg_rate)}")
        
        if is_stats:
            edges, results = histogram_with_mean_score(avg_prompts_hit, 6)
            print("Histogram with Mean Scores:")
            for i, result in enumerate(results):
                print(f"Bucket {i+1}: {result['bucket_range']}")
                print(f"  Count: {result['count']}")
                print(f"  Mean Score: {result['mean_score']:.2f}")
                print(f"  Items: {result['items']}")
                print()
        
        if is_csv:
            df = pd.DataFrame(result, columns=["semantic"])
            df['num_modifiers'] = num_of_modifiers
            df.to_csv(f"result-{directory}.csv", index=False)
elif option == 1:
    # remedial   
    with torch.no_grad():      
        BASE_PATH = PROJECT_PATH.replace("-new-classifier", "")
        
        model, _ = clip.load("ViT-L/14", device=device)
        result = []
        avg_prompts_hit = []
        avg_rate = []
        avg_cache_hit = []
        num_of_modifiers = []
        predict_result = []
        fps = 0
        overatio = 0
        
        if is_llama:
            llama_index = 0
            import json
            with open(f"{DATA_HOME}/results_json/diffdb_flux_results.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            llama_result = []
            llama_prompts = list(data.values())
            llama_embed = model.encode_text(clip.tokenize(llama_prompts, truncate=True).to(device))
            
        # completed = pickle.load(open("sequence.pkl", "rb"))
        for i in range(start, end):
            num = i
            if not os.path.exists(f"{BASE_PATH}/reconstructed_sorted_{i}.pkl"):
                print(f"Skipping {i} as {BASE_PATH}/reconstructed_sorted_{i}.pkl does not exist.")
                print("=" * 80)
                continue
            if not os.path.exists(f"{PROJECT_PATH}/re_exploited_prompts_{num}.pkl"):
                print(f"Skipping {i} as {PROJECT_PATH}/re_exploited_prompts_{num}.pkl file does not exist.")
                print("=" * 80)
                continue
            
            sorted_data = pickle.load(open(f"{BASE_PATH}/reconstructed_sorted_{num}.pkl", "rb"))
            
            if predict:
                before = pickle.load(open(f"{PROJECT_PATH}/re_batch_x_final_{num}.pkl", "rb"))
                cache_embedding = sorted_data[0][0][2]
                cos_sim_mat = torch.nn.functional.cosine_similarity(before, torch.tensor(cache_embedding).to(device), dim=-1)
                sim = torch.max(cos_sim_mat).item()
                predict_result.append(sim)
                print(f"Round {i} before similarity: {sim}")
                print("=" * 80)
                # continue
            
            len_of_max_classifier_result = len(pickle.load(open(f"{PROJECT_PATH}/re_extracted_data_{num}.pkl", "rb"))[0][1])
            len_of_result = sorted_data[0][1][0]
            ratio = (len_of_max_classifier_result - len_of_result) / len_of_result
            print(f"Round {num} has a significant difference: {len_of_max_classifier_result} vs {len_of_result}, over-ratio: {ratio}")
            avg_rate.append(ratio)
            if len_of_max_classifier_result > len_of_result:
                fps += len_of_max_classifier_result - len_of_result
                overatio += ratio
            
            hit_prompts = sorted_data[0][1][1]
            sum_of_modifiers = 0
            for (p, e, h) in hit_prompts:
                sum_of_modifiers += len([w for w in p.split(",") if w])
            num_of_modifiers.append(sum_of_modifiers // len(hit_prompts))
            # num_of_modifiers.append(len([w for w in sorted_data[0][0][1].split(",") if w]))
            
            avg_cache_hit.append(len_of_result)
            
            INPUT_PROMPT = pickle.load(open(f"{PROJECT_PATH}/re_exploited_prompts_{i}.pkl", "rb"))
                
            input_embed = model.encode_text(clip.tokenize(INPUT_PROMPT, truncate=True).to(device))
            input_embed = torch.nn.functional.normalize(input_embed, dim=-1, p=2)            
            cache_embedding = torch.tensor(sorted_data[0][0][2]).to(device)
            
            similarities = torch.nn.functional.cosine_similarity(cache_embedding, input_embed, dim=-1)
            max_ind = torch.argmax(similarities)
            similarity = similarities[max_ind]
            result.append(similarity.item())
            avg_prompts_hit.append((len_of_max_classifier_result, similarity.item()))
            
            if is_llama:
                llama_sim = torch.nn.functional.cosine_similarity(cache_embedding, llama_embed[llama_index], dim=-1)
                llama_result.append(llama_sim.item())
            
            print(f"Round {i} similarity: {similarity.item()}")
            print(f"Cache: {sorted_data[0][0][1]}")
            print(f"Input prompt: {INPUT_PROMPT[max_ind]}")
            
            if is_llama:
                print(f"Llama prompt: {llama_prompts[llama_index]}")
                print(f"Llama similarity: {llama_sim.item()}")
                llama_index += 1
            
            print("=" * 80)
            
            if is_images:
                generate_image(
                    pipe=pipe,
                    pp=INPUT_PROMPT[max_ind],
                    pe=input_embed[max_ind],
                    ce=cache_embedding,
                    ch=sorted_data[0][0][3],
                    num=num,
                    src_path=BASE_PATH,
                    dst_path=PROJECT_PATH
                )
                
        if is_llama:
            print(f"~" * 80)
            print(f"Llama results:")
            print_stats(llama_result)
            print("=" * 80)
            
            if is_csv:
                df = pd.DataFrame(llama_result, columns=["semantic"])
                df.to_csv(f"result-llama-{directory}.csv", index=False)
                exit()
                
        if predict:
            print_stats(predict_result)
            print(f"total rounds: {len(predict_result)}")
            print("=" * 80)
            exit()
            if is_csv:
                df = pd.DataFrame(predict_result, columns=["similarity"])
                df.to_csv(f"result-predict-{directory}.csv", index=False)
            exit()
        
        print_stats(result)
        print(f"total rounds: {len(result)}")
        print(f"Average cache hit: {np.mean(avg_cache_hit)}")
        print(f"Average prompt hit: {np.mean(avg_prompts_hit)}")
        print(f"Average rate: {np.mean(avg_rate)}")
        print(f"avg fps: {fps / len(result)}")
        print(f"avg over ratio: {overatio / len(result)}")
        
        if is_stats:
            edges, results = histogram_with_mean_score(avg_prompts_hit, 6)
            print("Histogram with Mean Scores:")
            for i, result in enumerate(results):
                print(f"Bucket {i+1}: {result['bucket_range']}")
                print(f"  Count: {result['count']}")
                print(f"  Mean Score: {result['mean_score']:.2f}")
                print(f"  Items: {result['items']}")
                print()

        if is_csv:
            df = pd.DataFrame(result, columns=["semantic"])
            df['num_modifiers'] = num_of_modifiers
            df.to_csv(f"result-{directory}.csv", index=False)
    
elif option == 2:
    # recover all / precise / fp
    
    with torch.no_grad():        
        model, _ = clip.load("ViT-L/14", device=device)
        result = []
        hit_numbers = []
        
        import re
        
        BASE_PATH = PROJECT_PATH.replace("-recover-all", "-7_17").replace("-recover-precise", "-7_17")
        BASE_PATH = re.sub(r"-fp-\d+", "-7_17", BASE_PATH)
               
        # completed = pickle.load(open("sequence.pkl", "rb"))
        for i in range(start, end):
            num = i
            
            if not os.path.exists(f"{BASE_PATH}/reconstructed_sorted_{num}.pkl"):
                print(f"Skipping {i} as {BASE_PATH}/reconstructed_sorted_{num}.pkl file does not exist.")
                print("=" * 80)
                continue
            
            if not os.path.exists(f"{PROJECT_PATH}/exploited_prompts_{num}.pkl"):
                print(f"Skipping {i} as {PROJECT_PATH}/exploited_prompts_{num}.pkl file does not exist.")
                print("=" * 80)
                continue

            if not os.path.exists(f"{PROJECT_PATH}/batch_x_final_{num}.pkl"):
                print(f"Skipping {i} as {PROJECT_PATH}/batch_x_final_{num}.pkl file does not exist.")
                print("=" * 80)
                continue
            
            sorted_data = pickle.load(open(f"{BASE_PATH}/reconstructed_sorted_{num}.pkl", "rb"))
            
            INPUT_PROMPT = pickle.load(open(f"{PROJECT_PATH}/exploited_prompts_{i}.pkl", "rb"))
                
            input_embed = model.encode_text(clip.tokenize(INPUT_PROMPT, truncate=True).to(device))
            input_embed = torch.nn.functional.normalize(input_embed, dim=-1, p=2)
            
            hit_numbers.append(sorted_data[0][1][0])
            
            cache_embedding = torch.tensor(sorted_data[0][0][2]).to(device)
            
            similarities = torch.nn.functional.cosine_similarity(cache_embedding, input_embed, dim=-1)
            max_ind = torch.argmax(similarities)
            similarity = similarities[max_ind]
            result.append(similarity.item())
            print(f"Round {i} similarity: {similarity.item()}")
            print(f"Cache: {sorted_data[0][0][1]}")
            print(f"Input prompt: {INPUT_PROMPT[max_ind]}")
            print("=" * 80)
            
            if is_images:
                generate_image(
                    pipe=pipe,
                    pp=INPUT_PROMPT[max_ind],
                    pe=input_embed[max_ind],
                    ce=cache_embedding,
                    ch=sorted_data[0][0][3],
                    num=num,
                    src_path=BASE_PATH,
                    dst_path=PROJECT_PATH
                )
        
        print_stats(result)
        print(f"total rounds: {len(result)}")
        print("=" * 80)
        
        if is_all:
            df = pd.DataFrame(result, columns=["semantic"])
            df['hit_numbers'] = hit_numbers
            df.to_csv(f"result-{directory}.csv", index=False)

if image_metrics:
    import torch
    import clip
    from PIL import Image
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    # CLIP similarity
    model, preprocess = clip.load("ViT-L/14", device=device)
    
    name = directory
    BASE_PATH = f'{DATA_HOME}/diffusion-cache-security/{directory}/experiments/'
    
    # BASE_PATTERN = f'{path}/*'
    scores = []
    not_hit = []
    
    clips_scores = []
    psnr_scores = []
    ssim_scores = []
    
    max_score = (-1.0, "")

    # Loop through all round* folders
    for round_dir in sorted(glob.glob(os.path.join(BASE_PATH, "round*"))):
        if "naive-probing" in directory:
            final_images_path = os.path.join(round_dir, "images")
        else:
            final_images_path = os.path.join(round_dir, "final-images")

        # Skip if final-images doesn't exist
        if not os.path.isdir(final_images_path):
            print(f"Skipping {final_images_path} as final-images folder does not exist.")
            continue

        print(f"Reading images from: {final_images_path}")        
        files = glob.glob(f"{final_images_path}/*.png")
        
        if len(files) != 2:
            print(f"Skipping {final_images_path} as it does not contain exactly two PNG images.")
            continue
        
        img1 = preprocess(Image.open(files[0])).unsqueeze(0).to(device)
        img2 = preprocess(Image.open(files[1])).unsqueeze(0).to(device)

        with torch.no_grad():
            emb1 = model.encode_image(img1)
            emb2 = model.encode_image(img2)

        emb1 /= emb1.norm(dim=-1, keepdim=True)
        emb2 /= emb2.norm(dim=-1, keepdim=True)
        clip_similarity = (emb1 * emb2).sum(dim=-1).item()
        # if clip_similarity > max_score[0]:
        #     max_score = (clip_similarity, folder)

        # PSNR and SSIM
        im1_np = np.array(Image.open(files[0]).convert("RGB"))
        im2_np = np.array(Image.open(files[1]).convert("RGB"))
        
        if im1_np.shape != im2_np.shape:
            raise ValueError(f"Images must match. Got {im1_np.shape} vs {im2_np.shape}")

        psnr_value = psnr(im1_np, im2_np, data_range=255)
        ssim_value, _ = ssim(im1_np, im2_np, multichannel=True, channel_axis=-1, full=True)
        
        clips_scores.append(clip_similarity)
        psnr_scores.append(psnr_value)
        ssim_scores.append(ssim_value)
    
    print("CLIP Similarity Stats:")
    print_stats(clips_scores)
    print("\nPSNR Stats:")
    print_stats(psnr_scores)
    print("\nSSIM Stats:")
    print_stats(ssim_scores)
    # print(f"Max CLIP similarity: {max_score[0]} in folder {max_score[1]}")
    
    df = pd.DataFrame(clips_scores, columns=['clip_similarity'])
    df['psnr'] = psnr_scores
    df['ssim'] = ssim_scores
    df.to_csv(f'result-images-{directory}.csv', index=False)
    # shutil.copy(f'{DATA_HOME}/{name}/result-{directory}.csv',
    #             f'{DATA_HOME}/result-{directory}.csv')