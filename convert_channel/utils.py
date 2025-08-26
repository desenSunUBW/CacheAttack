import glob
import pandas as pd
from tqdm import tqdm

batch_dir = ""
def get_all_diffusiondb_prompts():
    
    csv_files = sorted(glob.glob(f"{batch_dir}/prompts_batch_*.csv"))

    all_prompts = []

    for csv_file in tqdm(csv_files, desc="Reading CSV files"):
        df = pd.read_csv(csv_file, usecols=["prompt"]) 
        all_prompts.extend(df["prompt"].tolist())
    return all_prompts
    # print(f"Loaded {len(all_prompts):,} prompts into list.")
