# Attacks on Approximate Caches in Text-to-Image Diffusion Models

This repository contains implementations of three attacks on approximate caches in text-to-image diffusion models:
* **Covert Channel Attack** - Exploits cache latency patterns to covertly transmit information
* **Prompt Stealing Attack** - Recovers prompts from cached embeddings
* **Poison Attack** - Pollutes the cache to inject logos/markers into generated images

## System Requirements

### GPU Requirements
- **Minimum**: GPU with at least **16GB VRAM** (capable of running FLUX.1-schnell model)
- **Recommended**: GPU with **24GB+ VRAM** (e.g., NVIDIA A100, RTX Ada6000, RTX 4090, or similar)
- **CUDA**: CUDA 12.6+ compatible GPU
- The project supports both FLUX.1-schnell and Stable Diffusion 3.5 Medium models

### Memory Requirements
- **RAM**: **40GB** system memory (for handling large datasets and model operations)
- **Storage**: At least **100GB** free space for:
  - Model checkpoints (~20-30GB)
  - Generated images and cache files (~50-70GB)
  - Dataset files (DiffusionDB, Lexica, etc.)

### Estimated Runtime
- **Full experiment suite**: **10-20 hours** (depending on GPU, dataset size, and number of images generated)
  - Covert Channel Attack: ~2-4 hours
  - Prompt Stealing Attack: ~3-5 hours
  - Poison Attack: ~5-11 hours (includes cache simulation, image generation, and detection)

## Environment Setup

### Step 1: Create Conda Environment

Create a new conda environment from the provided YAML file:

```bash
conda env create -f diffusion_sec.yaml
```

This will create an environment named `diffusion_sec` with all required dependencies including:
- Python 3.9
- PyTorch 2.7.1 with CUDA 12.6 support
- Transformers, Diffusers, and other ML libraries
- CLIP, OWLv2, DINOv2 models support
- Jupyter and development tools

### Step 2: Activate Environment

```bash
conda activate diffusion_sec
```

### Step 3: Verify Installation

Verify that PyTorch can access your GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")
```

### Step 4: Download Pre-trained Models

#### Prompt Recovery Model
Download the embedding-to-prompt recovery model:
```bash
# The model is available at:
# https://huggingface.co/snownhonoka/attacks-on-approximate-caches-in-text_to_image-diffusion-models
```

#### Diffusion Models
The diffusion models (FLUX.1-schnell and Stable Diffusion 3.5 Medium) will be automatically downloaded from Hugging Face when first used.

### Step 5: Prepare Additional Files

1. **Logo Insertion Model**: The pre-trained model is located at `poison_attack/poison_emb/sampled_db/clip_phrase_model.pt`
2. **Logo Images**: Prepare logo images for detection (see individual attack READMEs for details)
3. **Datasets**: Prepare or download DiffusionDB and Lexica datasets as needed

## Model Information

### Pre-trained Models

The embedding-2-prompt reversion model used in the experiment is uploaded to `https://huggingface.co/snownhonoka/attacks-on-approximate-caches-in-text_to_image-diffusion-models`, trained using the DiffusionDB dataset. You can also train your own model using the training scripts provided in this repo.

The logo insertion at embedding space model used in this experiment is `poison_attack/poison_emb/sampled_db/clip_phrase_model.pt`, trained on a self-constructed dataset. The dataset construction code is in `poison_attack/poison_emb/convert_data_format.py`, and the training script is `poison_attack/poison_emb/logo_insertion_model.py`. You can train your new model using these training scripts.

The embedding to prompt with logo model used in this experiment is `coco-prefix_latest.pt`. The training script is `poison_attack/poison_emb/recover_prompt_with_logo_model.py` - you can train your own model with this script.

## Quick Start

1. **Set up the environment** (see Environment Setup above)
2. **Read the individual attack READMEs**:
   - `convert_channel/README.md` - For Covert Channel Attack
   - `prompt_stealing/README.md` - For Prompt Stealing Attack
   - `poison_attack/README.md` - For Poison Attack
3. **Follow the workflow** in each README to run the experiments

## Troubleshooting

### GPU Memory Issues
- Reduce batch sizes in the scripts
- Use gradient checkpointing if available
- Generate fewer images per experiment

### CUDA Compatibility
- Ensure your CUDA version matches PyTorch's CUDA version (12.6)
- Update GPU drivers if needed

### Environment Issues
- If conda environment creation fails, try updating conda: `conda update conda`
- For package conflicts, consider creating a fresh environment and installing packages incrementally
