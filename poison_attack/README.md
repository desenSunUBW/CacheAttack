# Poison Attack

This directory contains the implementation of the Poison Attack, which pollutes the cache of text-to-image diffusion models by inserting poisoned embeddings that cause normal prompts to generate images with logos/markers.

## Overview

The Poison Attack workflow consists of several steps:
1. **Prepare Logo Embeddings**: Generate CLIP embeddings for logos
2. **Generate Poisoned Embeddings**: Create embeddings that will be inserted into the cache
3. **Recover Prompts**: Use prompt recovery model to generate prompts from embeddings
4. **Simulate Cache Hits**: Determine which prompts will hit the poisoned cache
5. **Generate Images**: Generate images using the poisoned cache
6. **Detect Logos**: Evaluate success rate by detecting logos in generated images

## Prerequisites

- PyTorch with CUDA support
- CLIP model (ViT-L/14)
- OWLv2 and DINOv2 models for object detection
- Prompt recovery model (see Step 1)
- Stable Diffusion 3.5 Medium or FLUX.1-schnell models
- Logo insertion model (`clip_phrase_model.pt`) - should be trained or downloaded

## Setup

### Step 1: Download Prompt Recovery Model

Download the prompt recovery model checkpoint. The model path should be specified when running `poison_emb/emb2prompt.py`. The model is used to recover prompts from CLIP embeddings.

**Model Location**: The model path is specified via the `--model_path` (`-mp`) argument in `emb2prompt.py`. You can download a pre-trained model or train your own using the training scripts in this repository.

### Step 2: Prepare Logo Images and Embeddings

1. **Prepare Logo Image**: Place your logo image (e.g., `logo.png`) in the appropriate directory. The detection script expects logos to be in `logo/{logo_name}.png` format.

2. **Generate Logo CLIP Embedding**: Create the CLIP embedding for your logo and save it as `logo.pt`. The embedding should be placed in:
   ```
   {base_dir}/{logo_name}/logo.pt
   ```
   
   You can generate the logo embedding using CLIP:
   ```python
   import torch
   import clip
   
   model, preprocess = clip.load("ViT-L/14", device="cuda")
   logo_text = "huge Apple sign"  # Your logo description
   text = clip.tokenize(logo_text, truncate=True).cuda()
   text_feature = model.encode_text(text)
   torch.save(text_feature, f"{base_dir}/{logo_name}/logo.pt")
   ```

3. **Prepare Logo Insertion Model**: Ensure `clip_phrase_model.pt` is available at:
   ```
   {base_dir}/{logo_name}/clip_phrase_model.pt
   ```
**Note**: The prepared `clip_phrase_model.pt` is trained for Apple Sign.

### Step 3: Prepare Stolen Prompts

Prepare a file containing the stolen prompts (one prompt per line). This file will be used as input for the embedding generation process.

Example format:
```
prompt1
prompt2
prompt3
...
```

## Usage

### Step 1: Generate CLIP Embeddings with Logos

Run `poison_emb/emb_generator.py` to generate CLIP embeddings for prompts with logos inserted:

```bash
python poison_emb/emb_generator.py
```

This script:
- Loads prompts from `sampled_db/part_1.csv`
- Inserts logo contexts into prompts
- Generates CLIP embeddings for prompts with logos
- Saves embeddings to `sampled_db/{logo_name}/emb_with_logo.pt`

**Note**: Modify the `logos` list and `logo_contexts` in the script to match your logo names and contexts. Also modify the diffusiondb file path and replace it with `sampled_db/part_1.csv` in the script.

### Step 2: Calculate Similarity and Generate Cache Embeddings

Run the `get_similarity()` function in `similarity_eval.py`:

```bash
python similarity_eval.py <base_dir> similarity
```

**Arguments**:
- `base_dir`: Base directory path where logo files and embeddings are stored

This script:
- Loads stolen prompts from `{dataset}/{model}.log`
- Generates CLIP embeddings for the prompts
- Uses the logo insertion model to generate poisoned embeddings
- Calculates similarity metrics between original, direct insert, and generated embeddings
- Saves poisoned embeddings to `{base_dir}/{logo}/{dataset}-{model}_insert_emb_space.pt`

**Note**: Ensure the following files exist:
- `{dataset}/{model}.log` - stolen prompts file
- `{base_dir}/{logo}/{dataset}-{model}_emb_with_logo.pt` - direct insert embeddings
- `{base_dir}/{logo}/clip_phrase_model.pt` - logo insertion model
- `{base_dir}/{logo}/logo.pt` - logo CLIP embedding

### Step 3: Recover Prompts from Embeddings

Run `poison_emb/emb2prompt.py` to recover prompts from the poisoned embeddings:

```bash
python poison_emb/emb2prompt.py --option 2 --num_of_rounds <num> --model_path <model_path> --emb_dir <base_dir>
```

**Arguments**:
- `--option` (`-o`): Option flag (use `2` for logo embedding recovery)
- `--num_of_rounds` (`-n`): Number of rounds (optional)
- `--model_path` (`-mp`): Path to the prompt recovery model checkpoint
- `--emb_dir` (`-ed`): Base directory containing embeddings

This script:
- Loads poisoned embeddings from `{emb_dir}/{logo}/{dataset}-{model}_insert_emb_space.pt`
- Uses the prompt recovery model to generate text prompts from embeddings
- Saves recovered prompts to `{emb_dir}/{logo}/{dataset}-{model}_insert_emb_space.log`

### Step 4: Get Recovered Prompt Embeddings

Run the `get_recover_prompt_emb()` function in `similarity_eval.py`:

```bash
python similarity_eval.py <base_dir> recover
```

**Arguments**:
- `base_dir`: Base directory path where files are stored

This script:
- Loads recovered prompts from `{base_dir}/{logo}/{dataset}-{model}_insert_emb_space.log`
- Generates CLIP embeddings for recovered prompts
- Compares with original and direct insert embeddings
- Selects the best prompts (based on similarity) and saves to `{base_dir}/{logo}/{dataset}-{model}_cachetox.log`

**Note**: Ensure the following files exist:
- `{base_dir}/{logo}/{dataset}-{model}_insert_emb_space.log` - recovered prompts
- `{dataset}/{model}_original.log` - original stolen prompts
- `{base_dir}/{logo}/{dataset}-{model}-prompts.log` - direct insert prompts

### Step 5: Simulate Cache Hits

Run `cache_simulation.py` to simulate which prompts will hit the poisoned cache:

```bash
python cache_simulation.py <base_dir> <logo_index>
```

**Arguments**:
- `base_dir`: Base directory path
- `logo_index`: Index of the logo in the logos list (0-5)

This script:
- Loads target prompts from `{dataset}/{model}_original.log`
- Loads poisoned prompts from `{base_dir}/{logo}/{dataset}-{model}_cachetox.log`
- Simulates cache behavior using LCBFU replacement policy
- Records cache hits and saves results to `{base_dir}/{logo}/{dataset}-{model}_LCBFU_{cache_size}_cachetox_hit_rate.csv`

**Note**: Modify the `logos` list in the script to match your logo names. The script supports both "diffusiondb" and "lexica" datasets.

### Step 6: Generate Images Using Poisoned Cache

Run `generate.sh` to generate images after hitting the poisoned cache:

```bash
bash generate.sh
```

**Note**: Before running, edit `generate.sh` to set:
- `BASIC_PATH`: Base directory path
- `MODEL`: Model name ("flux" or "sd3")
- `DATASET`: Dataset name ("lexica" or "diffusiondb")
- `LOGOs`: Array of logo names

This script:
- Reads cache hit information from the CSV file generated in Step 5
- Generates images using the poisoned cache
- Saves images to `{BASIC_PATH}/{logo}/images/{dataset}/{model}/`
- Saves cache files to `{BASIC_PATH}/{logo}/cache/{dataset}/{model}/`

### Step 7: Detect Logos in Generated Images

Run `detection.py` to detect whether logos exist in the generated images:

```bash
python detection.py <base_dir>
```

**Arguments**:
- `base_dir`: Base directory path containing generated images (e.g., `{logo}/images/{dataset}/{model}`)

This script:
- Loads reference logo images from `logo/{logo_name}.png`
- Uses OWLv2 for object detection to find logo regions
- Uses DINOv2 to compute embeddings and cosine similarity
- Calculates success rate based on logo detection

**Output**: Success rate showing how often logos are detected in the generated images.

**Expected Result**: The success rate should be **non-zero**, indicating that the poison attack successfully polluted the cache and caused normal prompts to generate images with logos.

## Complete Workflow Summary

```bash
# Step 1: Generate embeddings with logos
python poison_emb/emb_generator.py

# Step 2: Calculate similarity and generate cache embeddings
python similarity_eval.py /path/to/base_dir similarity

# Step 3: Recover prompts from embeddings
python poison_emb/emb2prompt.py --option 2 --model_path /path/to/model.pt --emb_dir /path/to/base_dir

# Step 4: Get recovered prompt embeddings
python similarity_eval.py /path/to/base_dir recover

# Step 5: Simulate cache hits
python cache_simulation.py /path/to/base_dir 0

# Step 6: Generate images (edit generate.sh first)
bash generate.sh

# Step 7: Detect logos
python detection.py /path/to/base_dir/{logo}/images/{dataset}/{model}
```

## File Structure

The expected file structure after running all steps:

```
{base_dir}/
├── {logo_name}/
│   ├── logo.pt                          # Logo CLIP embedding
│   ├── clip_phrase_model.pt             # Logo insertion model
│   ├── {dataset}-{model}_emb_with_logo.pt
│   ├── {dataset}-{model}_insert_emb_space.pt
│   ├── {dataset}-{model}_insert_emb_space.log
│   ├── {dataset}-{model}_cachetox.log
│   ├── {dataset}-{model}_LCBFU_{cache_size}_cachetox_hit_rate.csv
│   ├── images/
│   │   └── {dataset}/
│   │       └── {model}/
│   └── cache/
│       └── {dataset}/
│           └── {model}/
└── logo/
    └── {logo_name}.png                   # Logo image for detection
```

## Notes

- The attack supports multiple logos: "blue moon sign", "Mcdonald sign", "Apple sign", "Chanel symbol", "circled triangle symbol", "circled Nike symbol"
- Cache simulation uses LCBFU (Least Cost-Benefit Frequently Used) replacement policy
- The success of the attack is measured by the logo detection rate in generated images
- Ensure all intermediate files are generated in the correct order before proceeding to the next step
- Modify dataset and model configurations in each script to match your setup
