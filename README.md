# Attacks on Approximate Caches in Text-to-Image Diffusion Models
Three Attacks:
* Convert Channel
* Prompt Stealing
* Poison Attack

The embedding-2-prompt reversion model used in the experiment is uploaded to `https://huggingface.co/snownhonoka/attacks-on-approximate-caches-in-text_to_image-diffusion-models`, trained by using diffutiondb dataset. You can also train your new one by using the training scripts provided in this repo.

The logo insertion at embedding space model used in this experiment is "poison_attack/poison_emb/sampled_db/clip_phrase_model.pt", trained by self-constructed dataset. The dataset construction code is in "poison_attack/poison_emb/convert_data_format.py", and the training script is "poison_attack/poison_emb/logo_insertion_model.py". You can train your new model by using the training scripts.

The embedding to prompt with logo model used in this experiment is "coco-prefix_latest.pt". The training script is "poison_attack/poison_emb/recover_prompt_with_logo_model.py", you can train your own model with this script.
