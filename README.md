# Attacks on Approximate Caches in Text-to-Image Diffusion Models

This repository contains implementations of three attacks on approximate caches in text-to-image diffusion models:
* **Covert Channel Attack** - Exploits cache latency patterns to covertly transmit information
* **Prompt Stealing Attack** - Recovers prompts from cached embeddings
* **Poison Attack** - Pollutes the cache to inject logos/markers into generated images

## Pre-trained Models

The pre-trained models used in this experiment are available at:
- **Zenodo**: `https://doi.org/10.5281/zenodo.17957900`
- **Hugging Face**: `https://huggingface.co/snownhonoka/attacks-on-approximate-caches-in-text_to_image-diffusion-models`

### Model Details

1. **Embedding-to-Prompt Recovery Model** (`coco_prefix-049.pt`): Trained using the DiffusionDB dataset. You can also train your own model using the training scripts provided in this repo.

2. **Logo Insertion Model** (`clip_phrase_model.pt`): Located at `poison_attack/poison_emb/sampled_db/clip_phrase_model.pt`, trained on a self-constructed dataset. The dataset construction code is in `poison_attack/poison_emb/convert_data_format.py`, and the training script is `poison_attack/poison_emb/logo_insertion_model.py`.

3. **Embedding to Prompt with Logo Model** (`coco-prefix_latest.pt`): Also available at the Zenodo link above. The training script is `poison_attack/poison_emb/recover_prompt_with_logo_model.py`.

## System Requirements

### GPU Requirements
- **Minimum**: GPU with at least **32GB VRAM** (capable of running FLUX.1-schnell model)
- **Recommended**: GPU with **40GB+ VRAM** (e.g., NVIDIA A100, RTX H100, RTX Ada 6000, or similar)
- **CUDA**: CUDA 12.6+ compatible GPU
- The project supports both FLUX.1-schnell and Stable Diffusion 3.5 Medium models

### Memory Requirements
- **RAM**: **32GB** system memory (for handling large datasets and model operations)
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

Download the pre-trained models from the Zenodo or Hugging Face links above and place them in the appropriate directories as specified in each attack's README.

#### Diffusion Models
The diffusion models (FLUX.1-schnell and Stable Diffusion 3.5 Medium) will be automatically downloaded from Hugging Face when first used.

### Step 5: Prepare Additional Files

1. **Logo Images**: Prepare logo images for detection (see individual attack READMEs for details)
2. **Datasets**: Prepare or download DiffusionDB and Lexica datasets as needed

This repository uses two datasets: **Lexica** and **DiffusionDB (text-only)**.

---

#### 1. Lexica Dataset

The **Lexica dataset** provided in this repository was prepared by following the methodology and data processing pipeline described in:

> Xinyue Shen, Yiting Qu, Michael Backes, and Yang Zhang.  
> *Prompt Stealing Attacks Against Text-to-Image Generation Models*.  
> USENIX Security Symposium (USENIX Security), 2024.

The original implementation is publicly available at:  
https://github.com/verazuo/prompt-stealing-attack

We gratefully acknowledge the authors for releasing their code and enabling reproducible research.

If you use the dataset prepared in this repository, please cite the original paper.

---

#### 2. DiffusionDB Dataset (Text Only)

For the second dataset, we use the **text-only metadata** from DiffusionDB.

We download the CSV file by following **Method 3 – "Use metadata.parquet (Text Only)"** from the official DiffusionDB GitHub repository:

https://github.com/poloclub/diffusiondb

Specifically:

1. Navigate to the DiffusionDB repository.
2. Locate the **Method 3: Use metadata.parquet (Text Only)** section.
3. Download the `metadata.parquet` file.
4. Convert the parquet file to CSV if needed.
5. Use the resulting CSV file for text-only experiments.

Please refer to the original repository for detailed instructions and licensing information.

**Important**: Make sure to update the `MODEL_PATH` variable in `emb_to_text.py` to point to your pre-trained model checkpoint. The model should be trained to recover prompts from CLIP embeddings (see training scripts in `model-training/` directory).


## Attack Implementations

### Covert Channel Attack

The Covert Channel attack exploits approximate caches to covertly transmit information through the generation process.

#### Overview

The attack consists of three main steps:
1. **Latency Classifier Evaluation**: Measure the success rate of identifying target prompts based on cache latency patterns
2. **Image Generation**: Generate images using cached latents from the sender's cache
3. **Content Classifier Evaluation**: Measure the success rate of detecting markers/logos in the generated images

#### Prerequisites

- PyTorch with CUDA support
- CLIP model (ViT-L/14)
- OWLv2 and DINOv2 models for object detection
- DiffusionDB dataset (CSV file at `../get_db/diffusiondb.csv`)
- Stable Diffusion 3.5 Medium or FLUX.1-schnell models

#### Usage

**Step 1: Evaluate Latency Classifier**

Run `success_rate.py` to calculate the success rate of the latency classifier:

```bash
python convert_channel/success_rate.py
```

This script:
- Generates CLIP embeddings for special texts A (with markers), test texts B (without markers), and dataset D (DiffusionDB prompts)
- Calculates cosine similarities between test texts and reference texts
- Determines success rates based on similarity thresholds and skip levels
- Outputs the probability of correctly identifying target prompts

**Output**: Success rate metrics for the latency classifier based on embedding similarity.

**Step 2: Generate Images from Sender's Cache**

Run `convert_channel_generation.py` to generate images using cached latents:

```bash
python convert_channel/convert_channel_generation.py <base_dir>
```

**Arguments**:
- `base_dir`: Base directory path where generated images and cache files will be stored

This script:
- Generates base images using special texts A (with markers) and saves them to `{base_dir}/base/`
- Saves cache files (latents) to `{base_dir}/cache/`
- Loads cached latents at appropriate skip levels based on similarity scores
- Generates converted images using test texts B (without markers) and saves them to `{base_dir}/images/`

**Output Structure**:
```
{base_dir}/
├── base/          # Base images with markers
├── cache/         # Cached latents at different skip levels
└── images/        # Converted images without markers
```

**Step 3: Evaluate Content Classifier**

Run `detection.py` to calculate the success rate of the content classifier (Note: prepare the dog and Mcdonald logo images for comparison):

```bash
python convert_channel/detection.py <base_dir>
```

**Arguments**:
- `base_dir`: Base directory path containing the generated images (same as used in Step 2)

This script:
- Loads reference marker images (dog.png for FLUX, Mcdonald.png for SD3)
- Uses OWLv2 for object detection to find logo/marker regions in generated images
- Uses DINOv2 to compute embeddings and cosine similarity with reference markers
- Calculates success rates based on whether markers are detected in the images

**Output**: Success rate metrics for the content classifier, showing how often markers are successfully detected in the generated images. The success rate should be over **90%** on average.

#### Workflow Summary

```bash
# Step 1: Evaluate latency classifier
python convert_channel/success_rate.py

# Step 2: Generate images from cache
python convert_channel/convert_channel_generation.py /path/to/output/directory

# Step 3: Evaluate content classifier
python convert_channel/detection.py /path/to/output/directory
```

#### Notes

- The scripts support both SD3 (`sd3`) and FLUX (`flux`) models
- Skip levels are determined based on CLIP similarity scores between original and guessed prompts
- The detection script requires marker images (`dog.png` for FLUX, `Mcdonald.png` for SD3) to be present in the working directory
- Ensure sufficient GPU memory for model loading and image generation

### Prompt Stealing Attack

See `prompt_stealing/README.md` for detailed instructions.

### Poison Attack

See `poison_attack/README.md` for detailed instructions.

## Quick Start

1. **Set up the environment** (see Environment Setup above)
2. **Download pre-trained models** from Zenodo or Hugging Face
3. **Read the individual attack READMEs**:
   - `convert_channel/README.md` - For Covert Channel Attack (or see above)
   - `prompt_stealing/README.md` - For Prompt Stealing Attack
   - `poison_attack/README.md` - For Poison Attack
4. **Follow the workflow** in each README to run the experiments

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

### Model Loading Issues
- Verify that model paths are correct in the scripts
- Ensure model checkpoints are downloaded from Zenodo or Hugging Face
- Check that model files are not corrupted
