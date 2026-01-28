# Prompt Stealing Attack

This directory contains the implementation of the Prompt Stealing attack, which recovers prompts from cached embeddings in text-to-image diffusion models by exploiting the approximate cache mechanism.

## Overview

The Prompt Stealing attack demonstrates that cached embeddings in diffusion models can be reverse-engineered to recover the original prompts. The attack uses a pre-trained embedding-to-text recovery model to reconstruct prompts from cached CLIP embeddings.

## Prerequisites

- PyTorch with CUDA support
- CLIP model (ViT-L/14)
- Pre-trained prompt recovery model (see Setup section)
- Stable Diffusion 3.5 Medium or FLUX.1-schnell models
- Access to DiffusionDB or Lexica datasets

## Environment Setup

### Step 1: Install Conda Environments

This project requires two conda environments. Install them from the `conda-envs` folder:

```bash
# Install diff-sec environment (for diffusion model operations)
conda env create -f conda-envs/diff-sec.yml

# Install ml environment (for machine learning operations)
conda env create -f conda-envs/ml.yml
```

**Note**: The environments will be automatically activated by the `script-run.sh` script as needed.

### Step 2: Set Environment Variables

Set the following environment variables:

```bash
export DATA_HOME=/path/to/your/data/directory
export PROGRAM_HOME=/path/to/your/program/directory
```

These variables are used by the scripts to locate data files and project directories.

### Step 3: Prepare Pre-trained Model

Download or prepare the pre-trained embedding-to-text recovery model. The model path is specified in `emb_to_text.py`:

```python
MODEL_PATH = f"./coco_prefix-049.pt"
```

**Important**: Make sure to update the `MODEL_PATH` variable in `emb_to_text.py` to point to your pre-trained model checkpoint. The model should be trained to recover prompts from CLIP embeddings (see training scripts in `model-training/` directory).

## Usage

### Main Experiment Pipeline

The main experiment pipeline is controlled by `script-run.sh`. This script orchestrates the entire attack workflow.

#### Script Arguments

The `script-run.sh` script takes the following arguments (documented at the bottom of the file):

```bash
bash script-run.sh <start> <end> <exp_dir> <min_words> <max_words> <model_name>
```

**Arguments**:
- `start`: Starting run ID (integer)
- `end`: Ending run ID (integer)
- `exp_dir`: Experiment directory name (string)
- `min_words`: Minimum number of words in prompts (integer)
- `max_words`: Maximum number of words in prompts (integer)
- `model_name`: Model name, either `"flux"` or `"sd3"` (string)

**Example**:
```bash
bash script-run.sh 1 10 my_experiment 5 15 flux
```

This will run experiments 1 through 10 in the `my_experiment` directory, using prompts with 5-15 words, and the FLUX model.

#### Pipeline Function Arguments

The `run_pipeline()` function in `script-run.sh` takes the following parameters (documented at the beginning of the function):

- `run_id`: The run ID number
- `is_test`: Boolean flag controlling whether to generate new images from the diffusion model
  - `true`: Generate images (slower, requires GPU)
  - `false`: Skip image generation (faster, for testing pipeline)
- `min_words`: Minimum number of words in prompts
- `max_words`: Maximum number of words in prompts
- `model_name`: Model name (`"flux"` or `"sd3"`)

**Note**: The `is_test` argument is hardcoded to `true` in the script. To skip image generation, modify line 133 in `script-run.sh` to set `is_test` to `false`.

### Pipeline Workflow

The pipeline executes the following steps:

1. **Input Construction** (`input_constructor.py`):
   - Option 0: Initial setup and candidate embedding generation
   - Option 1: Generate prompts from recovered embeddings
   - Option 2: Final processing and validation

2. **Embedding Recovery** (`recover_emb.py`):
   - Option 0: Recover embeddings using optimization (multiple rounds with different thresholds)
   - Option 1: Final embedding recovery

3. **Embedding to Text** (`emb_to_text.py`):
   - Option 0: Convert candidate embeddings to text prompts
   - Option 1: Convert final recovered embeddings to text prompts
   - **Important**: Ensure `MODEL_PATH` in this file points to your pre-trained model

4. **Image Generation** (if `is_test=true`):
   - `run_flux.py` or `run_sd3.py`: Generate images using the diffusion model
   - Uses cached latents to speed up generation

5. **Output Classification** (`output_classifier.py`):
   - Classify generated images (if images were generated)

6. **Similarity Evaluation** (`check_cosine_sim.py`):
   - Option 0: Calculate cosine similarity between original and recovered prompts
   - This is the final evaluation step

### Evaluation Scripts

#### Cosine Similarity Check

The main evaluation script is `check_cosine_sim.py`:

```bash
conda activate diff-sec
python check_cosine_sim.py -o 0 -n <run_id> -cn <collection_name> -dir <exp_dir>
```

**Arguments**:
- `-o 0`: Option 0 (similarity check)
- `-n`: Run ID number
- `-cn`: Collection name (e.g., `"diffdb"` or `"lexica"`)
- `-dir`: Experiment directory name

**Output**: Prints the cosine similarity score between the original cached prompt and the recovered prompt. The expected result should be **over 0.75 on average**.

#### Additional Evaluation Scripts

The following scripts provide additional similarity metrics:

- **`check_bleu_score.py`**: Calculates BLEU scores, word precision/recall/F1, Jaccard similarity, and BERT scores between original and recovered prompts
- **`check_cosine_sim.py`** (with `--image_metrics`): Calculates CLIP image similarity, PSNR, and SSIM between generated images

**Note**: These scripts are provided as reference implementations. You are strongly encouraged to modify them based on your specific evaluation needs.

## Expected Results

After running the complete pipeline, `check_cosine_sim.py` should print similarity results. The expected performance is:

- **Cosine Similarity**: **Over 0.75 on average** between the original cached prompt and the recovered prompt

This indicates that the attack successfully recovers semantic information from the cached embeddings.

## File Structure

After running experiments, the directory structure will be:

```
{DATA_HOME}/diffusion-cache-security/{exp_dir}/
├── run_01/
│   ├── result.txt              # Main results
│   ├── result-recov.txt        # Embedding recovery results
│   ├── result-generated.txt    # Generated prompt results
│   ├── result-mlp.txt          # MLP (embedding-to-text) results
│   ├── result-check.txt        # Final similarity check results
│   ├── classifier-result.txt   # Classification results
│   └── image_result.txt        # Image generation results
├── experiments/
│   └── round{run_id}/
│       ├── cache/              # Cached latents
│       └── images/             # Generated images
├── batch_x_final_{run_id}.pkl  # Final recovered embeddings
├── exploited_prompts_{run_id}.pkl  # Recovered prompts
├── extracted_data_{run_id}.pkl     # Extracted cache data
└── sorted_data_{run_id}.pkl       # Sorted cache data
```

## Troubleshooting

### Model Loading Issues

If you encounter errors loading the pre-trained model in `emb_to_text.py`:

1. Verify that `MODEL_PATH` points to the correct model file
2. Ensure the model checkpoint is compatible with the model architecture defined in the script
3. Check that the model file exists and is not corrupted

### Environment Activation Issues

The script uses `conda activate` commands. If you encounter activation errors:

1. Ensure conda is properly initialized in your shell
2. Verify that both `diff-sec` and `ml` environments are installed
3. Check that the conda initialization is in your `~/.bashrc` or `~/.zshrc`

### GPU Memory Issues

If you run out of GPU memory:

1. Set `is_test=false` in `script-run.sh` to skip image generation
2. Reduce batch sizes in the scripts
3. Use a GPU with more VRAM

### Collection Name

The default collection name in `script-run.sh` is `"diffdb"`. To use a different dataset (e.g., `"lexica"`), modify line 7 in `script-run.sh`:

```bash
collection_name="lexica"  # or "diffdb"
```

## Notes

- The pipeline runs multiple rounds with different similarity thresholds (0.60, 0.55, 0.50) and optimization steps (3000, 5000, 7000)
- Image generation is optional and can be skipped by setting `is_test=false`
- The attack works on both FLUX and SD3 models
- Results are saved in both text files and pickle files for further analysis
- The similarity evaluation is the key metric for attack success
