# Covert Channel Attack

This directory contains the implementation of the Covert Channel attack, which exploits approximate caches in text-to-image diffusion models to covertly transmit information through the generation process.

## Overview

The Covert Channel attack consists of three main steps:
1. **Latency Classifier Evaluation**: Measure the success rate of identifying target prompts based on cache latency patterns
2. **Image Generation**: Generate images using cached latents from the sender's cache
3. **Content Classifier Evaluation**: Measure the success rate of detecting markers/logos in the generated images

## Prerequisites

- PyTorch with CUDA support
- CLIP model (ViT-L/14)
- OWLv2 and DINOv2 models for object detection
- DiffusionDB dataset (CSV file at `../get_db/diffusiondb.csv`)
- Stable Diffusion 3.5 Medium or FLUX.1-schnell models

## Usage

### Step 1: Evaluate Latency Classifier

Run `success_rate.py` to calculate the success rate of the latency classifier:

```bash
python success_rate.py
```

This script:
- Generates CLIP embeddings for special texts A (with markers), test texts B (without markers), and dataset D (DiffusionDB prompts)
- Calculates cosine similarities between test texts and reference texts
- Determines success rates based on similarity thresholds and skip levels
- Outputs the probability of correctly identifying target prompts

**Output**: Success rate metrics for the latency classifier based on embedding similarity.

### Step 2: Generate Images from Sender's Cache

Run `convert_channel_generation.py` to generate images using cached latents:

```bash
python convert_channel_generation.py <base_dir>
```

**Arguments**:
- `base_dir`: Base directory path where generated images and cache files will be stored

This script:
- Generates base images using special texts A (with markers) and saves them to `{base_dir}/base/`
- Saves cache files (latents) to `{base_dir}/cache/`
- Loads cached latents at appropriate skip levels based on similarity scores
- Generates Coverted images using test texts B (without markers) and saves them to `{base_dir}/images/`

**Output Structure**:
```
{base_dir}/
├── base/          # Base images with markers
├── cache/         # Cached latents at different skip levels
└── images/        # Coverted images without markers
```

### Step 3: Evaluate Content Classifier

Run `detection.py` to calculate the success rate of the content classifier (Notice that you should prepare for the dog and Mcdonald logo image for comparison):

```bash
python detection.py <base_dir>
```

**Arguments**:
- `base_dir`: Base directory path containing the generated images (same as used in Step 2)

This script:
- Loads reference marker images (dog.png for FLUX, Mcdonald.png for SD3)
- Uses OWLv2 for object detection to find logo/marker regions in generated images
- Uses DINOv2 to compute embeddings and cosine similarity with reference markers
- Calculates success rates based on whether markers are detected in the images

**Output**: Success rate metrics for the content classifier, showing how often markers are successfully detected in the generated images. The success rate should be over **90%** on average.

## Workflow Summary

```bash
# Step 1: Evaluate latency classifier
python success_rate.py

# Step 2: Generate images from cache
python Covert_channel_generation.py /path/to/output/directory

# Step 3: Evaluate content classifier
python detection.py /path/to/output/directory
```

## Notes

- The scripts support both SD3 (`sd3`) and FLUX (`flux`) models
- Skip levels are determined based on CLIP similarity scores between original and guessed prompts
- The detection script requires marker images (`dog.png` for FLUX, `Mcdonald.png` for SD3) to be present in the working directory
- Ensure sufficient GPU memory for model loading and image generation
