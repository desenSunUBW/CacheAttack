from pipeline_sd3_nirvana import NIRVANAStableDiffusion3Pipeline
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
import utils

# DATA_HOME = os.getenv("DATA_HOME")
# PROJECT_PATH = f"{DATA_HOME}/diffusion-cache-security"

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "-n", "--num", 
#     type=int,           # or float if you want decimal numbers
#     required=True,
# )

# args = parser.parse_args()
# num = args.num

DEVICE = "cuda:0"
# DEVICE = "cpu"

# pipe = NIRVANAStableDiffusion3Pipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
# pipe.pipeline.to(DEVICE)
# prompts_index = [59,116,167,433,1115,1278,1373,1430,1792,2534,2872,3233,3278,3537,6066,8180,11089,11166,13579,15863,19576,19975,24240,24607,27084,32931]
# diffusiondb = pd.read_csv('../get_db/unique_tmp.csv')
# diffusiondb = diffusiondb.iloc[:, 0].values.tolist()
# diffusiondb = ["A hyper-detailed portrait of a warrior with glowing blue eyes, intricate armor, digital painting, artstation, concept art, sharp focus, art by Artgerm and Greg Rutkowski.",  "A warrior in golden armor, holding a glowing sword, standing in a battlefield, highly detailed, digital painting, artstation, concept art, sharp focus, art by Artgerm and Greg Rutkowski.", "A hyper-realistic portrait of a futuristic cyborg, ray tracing, highly detailed, 8k, digital painting, artstation, concept art, sharp focus, art by Greg Rutkowski and Artgerm."]
# torch.cuda.manual_seed(42)
clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
steps_per_level = 3
import os
import random
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import argparse
from pathlib import Path

# Optional imports for diffusion models - uncomment based on your setup
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# import torch
# from PIL import Image

class PromptProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_prompts = []
        self.unique_prompts_per_index = defaultdict(set)
        self.stats = {}
        
    def load_file(self) -> None:
        """Load and parse the prompt file."""
        print(f"Loading file: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(',', 1)  # Split only on first comma
                if len(parts) < 2:
                    print(f"Warning: Skipping malformed line {line_num}: {line}")
                    continue
                    
                index = parts[0].strip()
                prompt = parts[1].strip()
                if "insert_cache" not in index:
                    continue
                self.raw_prompts.append((int(index.split("_")[-1]), prompt))
                self.unique_prompts_per_index[int(index.split("_")[-1])].add(prompt)
    
    def load_poisoned_cache(self, cache_path):
        self.poisoned_cache = []
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line:
                    continue
                self.poisoned_cache.append(line)

    def calculate_stats(self) -> Dict:
        """Calculate statistics about the prompts."""
        total_prompts = len(self.raw_prompts)
        total_unique_prompts = sum(len(prompts) for prompts in self.unique_prompts_per_index.values())
        total_indices = len(self.unique_prompts_per_index)
        hit_distribution = sorted([len(prompts) for prompts in self.unique_prompts_per_index.values()])
        # Calculate unique prompts per index
        unique_counts = {index: len(prompts) for index, prompts in self.unique_prompts_per_index.items()}
        
        self.stats = {
            'total_prompts': total_prompts,
            'total_unique_prompts': total_unique_prompts,
            'total_indices': total_indices,
            'unique_counts_per_index': unique_counts,
            'hit_distribution': hit_distribution,
            'duplicate_ratio': 1 - (total_unique_prompts / total_prompts) if total_prompts > 0 else 0
        }
        
        return self.stats
    
    def print_stats(self) -> None:
        """Print detailed statistics."""
        stats = self.calculate_stats()
        
        print("\n" + "="*60)
        print("PROMPT ANALYSIS RESULTS")
        print("="*60)
        print(f"Total prompts loaded: {stats['total_prompts']}")
        print(f"Total unique prompts: {stats['total_unique_prompts']}")
        print(f"Total indices: {stats['total_indices']}")
        print(f"Hit Distribution: {stats['hit_distribution']}")
        print(f"Duplicate ratio: {stats['duplicate_ratio']:.2%}")
        
        print("\n" + "-"*40)
        print("UNIQUE PROMPTS PER INDEX:")
        print("-"*40)
        
        # Sort by unique count (descending)
        sorted_indices = sorted(stats['unique_counts_per_index'].items(), 
                              key=lambda x: x[1], reverse=True)
        
        for index, count in sorted_indices:
            print(f"Index {index}: {count} unique prompts")
            
        print("-"*40)
        
    def generate_balanced_selection(self, target_count: int = 100) -> List[Tuple[str, str]]:
        if not self.unique_prompts_per_index:
            raise ValueError("No prompts loaded. Call load_file() first.")
            
        indices = list(self.unique_prompts_per_index.keys())
        prompts_per_index = target_count // len(indices)
        remainder = target_count % len(indices)
        
        selected_prompts = []
        distribution = {}
        
        for i, index in enumerate(indices):
            available_prompts = list(self.unique_prompts_per_index[index])
            target_for_this_index = prompts_per_index + (1 if i < remainder else 0)
            actual_count = min(target_for_this_index, len(available_prompts))
            
            # Randomly sample without replacement
            selected = random.sample(available_prompts, actual_count)
            distribution[index] = actual_count
            
            for prompt in selected:
                selected_prompts.append((index, prompt))
        
        # Shuffle the final selection to mix indices
        random.shuffle(selected_prompts)
        
        print("\nDistribution achieved:")
        for index, count in distribution.items():
            print(f"  Index {index}: {count} prompts")
            
        return selected_prompts
    
    def save_selected_prompts(self, selected_prompts: List[Tuple[str, str]], 
                            output_file: str = "selected_prompts.txt") -> None:
        """Save selected prompts to a file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Selected Prompts for Image Generation\n")
            f.write("="*50 + "\n\n")
            
            for i, (index, prompt) in enumerate(selected_prompts, 1):
                f.write(f"Image {i:03d} (Index {index}): {prompt}\n")
        
        print(f"\nSelected prompts saved to: {output_file}")

class DiffusionImageGenerator:
    def __init__(self, model_name: str = "sd3", 
                 device: str = "cuda", output_dir: str = "generated_images"):
        
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipe = None
        
    def load_model(self):
        try:
            if self.model_name == "sd3":
                self.shape = [1, 16, 128, 128]
                self.pipe = NIRVANAStableDiffusion3Pipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
            else:
                self.shape = [1, 4096, 64]
                self.pipe = NIRVANAFluxPipeline.from_pretrained(pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
            self.pipe.pipeline.to(DEVICE)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have the required libraries installed:")
            print("pip install diffusers transformers accelerate")
            raise
    
    def generate_images(self, prompt: str, 
                       width: int = 512, height: int = 512, 
                       num_inference_steps: int = 30,
                       latents: torch.Tensor = None,
                       guidance_scale: float = 7.5,
                       cache_dir: str = "",
                       index: int = 0,
                       cache_index: int = 0,
                       skip_steps: int = 0,
                       save_cache: bool = False) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        image = self.pipe(prompt,num_inference_steps=num_inference_steps,
                                latents=latents,
                                save_cache=save_cache,
                                cache_path=cache_dir/f"{index}",
                                skip_steps=skip_steps,
                                guidance_scale=3.5).images[0]
        if save_cache:
            image.save(self.output_dir/f"{index}.png")
            path = self.output_dir/f"{index}.png"
        else:
            if skip_steps == 0:
                path = self.output_dir/f"image-{index}.png"
            else:
                path = self.output_dir/f"cache-{cache_index}-image-{index}.png"
            image.save(path)
        # print(path)        

def main():
    parser = argparse.ArgumentParser(description="Process prompts and generate images")
    parser.add_argument("file_path", help="Path to the prompt file")
    parser.add_argument("--count", type=int, default=50, 
                       help="Number of images to generate (default: 100)")
    parser.add_argument("--model", default="sd3",
                       help="Diffusion model ID")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device for inference")
    parser.add_argument("--output-dir", default="generated_images",
                       help="Output directory for images")
    parser.add_argument("--cache-dir", default="generated_images",
                       help="Output directory for images")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--steps", type=int, default=30, 
                       help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--logo-name", type=str, help="logo_name")
    parser.add_argument("--cache-path", type=str, help="cache file path")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze prompts, don't generate images")
    parser.add_argument("--generate-cache", action="store_true",
                       help="Only analyze prompts, don't generate images")
    
    args = parser.parse_args()
    
    # Process prompts
    processor = PromptProcessor(args.file_path)
    processor.load_file()
    processor.load_poisoned_cache(args.cache_path)
    processor.print_stats()
    
    # Generate balanced selection
    selected_prompts = processor.generate_balanced_selection(args.count)
    # processor.save_selected_prompts(selected_prompts, 
    #                               f"selected_prompts_{args.count}.txt")
    generator = DiffusionImageGenerator(
        model_name="sd3",
        device=args.device,
        output_dir=args.output_dir
    )
    generator.load_model()
    target_index = 43
    
    if args.generate_cache:
        for index, _ in processor.unique_prompts_per_index.items():
            torch.cuda.manual_seed(time.time() % 1000000)
            latents = torch.randn(generator.shape, device="cuda", dtype=torch.float16)
            # print(processor.poisoned_cache[index])
            generator.generate_images(
                processor.poisoned_cache[index],
                num_inference_steps=args.steps,
                save_cache=True,
                cache_dir=args.cache_dir,
                latents=latents,
                index=index,                
            )
            # break

    '''
    if not args.analyze_only:
        # Generate images
        image_number = 0
        for (index, prompt) in selected_prompts:
            skip_steps = utils.get_skip_steps(processor.poisoned_cache[index], prompt, args.steps // 10)
            if skip_steps == 0:
                print(f"index {index} has a wrong hit with {prompt}")
                skip_steps = 3
            try:
                latents = torch.load((f"{args.cache_dir}/{index}-{skip_steps}.pt")).cuda()
            except:
                continue
            generator.generate_images(
                prompt,
                num_inference_steps=args.steps,
                save_cache=False,
                cache_dir=args.cache_dir,
                latents=latents,
                index=image_number,
                cache_index=index,
                skip_steps=skip_steps,
            )
            # torch.cuda.manual_seed(time.time() % 1000000)
            # latents = torch.randn(generator.shape, device="cuda", dtype=torch.float16)
            # generator.generate_images(
            #     prompt,
            #     num_inference_steps=args.steps,
            #     save_cache=False,
            #     cache_dir=args.cache_dir,
            #     latents=latents,
            #     index=image_number,
            #     cache_index=index,
            #     skip_steps=0,
            # )
            image_number += 1
            break
    '''

if __name__ == "__main__":
    # Uncomment the line below to run the example
    # example_usage()
    
    # Run main CLI
    main()