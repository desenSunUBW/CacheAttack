import random

from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import pickle
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

import clip
from tqdm import tqdm
import os

import argparse

args = argparse.ArgumentParser()
args.add_argument("--option", "-o", type=int, required=True)
args.add_argument("--num_of_rounds", "-n", type=int)
args = args.parse_args()
option = args.option
num = args.num_of_rounds

DATA_HOME = os.getenv("DATA_HOME")
PROJECT_PATH = f"{DATA_HOME}/diffusion-cache-security"

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

# WEIGHTS_PATHS = [
#     "./data/checkpoints-max_tokens_76-prefix_length_35-bs_24-compiled-MLP/coco_prefix-049.pt",
#     "./data/checkpoints-max_tokens_76-prefix_length_35-bs_24-compiled-MLP/coco_prefix-044.pt",
#     "./data/checkpoints-max_tokens_76-prefix_length_35-bs_24-compiled-MLP/coco_prefix-039.pt",
#     "./data/checkpoints-max_tokens_76-prefix_length_35-bs_24-compiled-MLP/coco_prefix-034.pt",
#     "./data/checkpoints-max_tokens_76-prefix_length_35-bs_24-compiled-MLP/coco_prefix-029.pt",
# ]

WEIGHTS_PATHS = [
#    "./data/checkpoints-max_tokens_101-prefix_length_20-bs_32-compiled-MLP-prefix_only/coco_prefix-006.pt"
{"dir": "./data/checkpoints-max_tokens_101-prefix_length_20-bs_32-compiled-MLP", "model": "coco_prefix-049.pt"}
]

MODEL_PATH=f"./data/checkpoints-max_tokens_101-prefix_length_20-bs_32-compiled-MLP/coco_prefix-049.pt"

D = torch.device
CPU = torch.device("cpu")
DEVICE = "cuda:1"
# DEVICE = "cpu"

gpt2_model = "gpt2"

torch.set_float32_matmul_precision('high')
        
def main(path, data, option, model_name="coco", use_beam_search=False, is_only_prefix=False, batch_size=128):
    # Device setup
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = DEVICE
    
    # model_path = path["dir"] + "/" + path["model"]
    model_path = path
    
    if option == 1:
        surfix = ""
    elif option == 2:
        surfix = "_half"
    
    # Load CLIP model and tokenizer
    # clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
    
    # Model loading
    prefix_length = 20
    model = ClipCaptionModel(prefix_length, 768)
    checkpoint = torch.load(model_path, map_location=torch.device(device))

    model = torch.compile(model)
    model.load_state_dict(checkpoint["model_state_dict"])
        
    model = model.eval()
    model = model.to(device)

    # clip_model, _ = clip.load("ViT-L/14", device=device)

    cosine_similarities_result = []

    with torch.no_grad():
        prompts = []
        for i in tqdm(range(0, len(data['clip_embedding']), batch_size), desc="Processing batches", position=0, leave=False, dynamic_ncols=True):
            # print(f"Processing batch {i} to {i + batch_size}")

            prompt_emb = data['clip_embedding'][i : i + batch_size].to(device, dtype=torch.float32)
            num_prompts = len(prompt_emb)
            prefix_embed = model.clip_project(prompt_emb).reshape(num_prompts, prefix_length, -1)
        
            if use_beam_search:
                result = generate_beam(model, tokenizer, embed=prefix_embed)
            else:
                result = generate2(model, tokenizer, embed=prefix_embed)
            
            prompts.extend([p[0] for p in result])
            
            # print for verification
            # for i, item in enumerate(result):
            #     print("=" * 80)
            #     # print(f"original: {data['captions']['caption']}\n")

            #     for j, res in enumerate(item):
            #         print(f"generated_{j}: {item[j]}")
            #     exit()
            
            """
            flattened_result = [item for sublist in result for item in sublist]
            generated_prompts_embedding = clip_model.encode_text(clip.tokenize(flattened_result, truncate=True).to(device))
            
            normed_generated_prompts_embedding = generated_prompts_embedding / generated_prompts_embedding.norm(dim=-1, keepdim=True)     
            normed_generated_prompts_embedding = normed_generated_prompts_embedding.view(num_prompts, num_options, -1)

            # Compute cosine similarity between each prompt and its corresponding generated candidates.
            # Since embeddings are normalized, cosine similarity is the dot product.
            # We expand prompt_embeddings to (i, 1, embedding_dim) so that broadcasting works.

            cosine_similarities = torch.nn.functional.cosine_similarity(normed_generated_prompts_embedding, prompt_emb.unsqueeze(1), dim=-1)
            print(f"Similarities: {cosine_similarities}")
            # print(f"generated_prompts_embedding: {generated_prompts_embedding}")

            max_similarity, max_index = torch.max(cosine_similarities, dim=-1)
            # print(f"max_similarity: {max_similarity}")
            cosine_similarities_result.extend(max_similarity.tolist())
            print(f"result: {cosine_similarities_result}")

            # print(f"avg: {sum(cosine_similarities_result) / len(cosine_similarities_result)}")
            
            # print("=" * 80)
                # progress.update()
            
            # if (i // batch_size) % 10 == 0:
            #     with open(f"./data/evaluations/evaluation_result_{path['dir'][7:]}-{path['model'][:-3]}{surfix}.pkl", "wb") as f:
            #         pickle.dump(cosine_similarities_result, f)


            # tensor = torch.tensor(cosine_similarities_result, dtype=torch.float32, device="cuda:1")

            # # Compute statistics
            # mean = torch.mean(tensor)            # Mean
            # median = torch.median(tensor)        # Median
            # std_dev = torch.std(tensor)          # Standard deviation
            # variance = torch.var(tensor)         # Variance
            # min_val = torch.min(tensor)          # Minimum
            # max_val = torch.max(tensor)          # Maximum

            # print(f"Mean: {mean}, Median: {median}, Std Dev: {std_dev}, Variance: {variance}")
            # print(f"Min: {min_val}, Max: {max_val}")
            # print()
            
        # with open(f"./data/evaluations/evaluation_result_{path['dir'][7:]}-{path['model'][:-3]}{surfix}.pkl", "wb") as f:
        #     pickle.dump(cosine_similarities_result, f)
            """
        return prompts
        print(f"Generated prompts: {len(prompts)}")
        
        pickle.dump(prompts, open(f"{PROJECT_PATH}/exploited_prompts_{num}.pkl", "wb"))
        if option != 0:
            for i, prompt in enumerate(prompts):
                print("=" * 80)
                print(f"{i}: {prompt}")

class MLP(nn.Module):
    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        # if prefix_length > 10:  # not enough memory
        #     self.clip_project = nn.Linear(
        #         prefix_size, self.gpt_embedding_size * prefix_length
        #     )
        # else:
        #     self.clip_project = MLP(
        #         (
        #             prefix_size,
        #             (self.gpt_embedding_size * prefix_length) // 2,
        #             self.gpt_embedding_size * prefix_length,
        #         )
        #     )
            
        self.clip_project = MLP(
            (
                prefix_size,
                (self.gpt_embedding_size * prefix_length) // 2,
                self.gpt_embedding_size * prefix_length,
            )
        )


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 3,
    prompt=None,
    embed=None,
    entry_length=75,
    temperature=0.7,
    stop_token: str = "#",
):

    model.eval()
    # stop_token_index = tokenizer.encode(stop_token)[0]
    stop_token_index = tokenizer.eos_token_id
    tokens = None
    scores = None
    device = next(model.parameters()).device
    generated = embed
    batch_size = generated.shape[0]
        
    seq_lengths = torch.ones(batch_size, beam_size, device=device)
    is_stopped = torch.zeros(batch_size, beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():

        for i in range(entry_length):
            #logits = (batch_size * beam_size, seq_len, vocab_size)
            logits = model.gpt(inputs_embeds=generated).logits

            # logits = (batch_size * beam_size, vocab_size)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            
            if scores is None:
                # scores, next_tokens = (batch_size, beam_size)
                scores, next_tokens = logits.topk(beam_size, -1)
            
                # generated = (batch_size,  beam_size, prefix_len, dim)
                generated = generated.unsqueeze(1).expand(batch_size, beam_size, *generated.shape[1:])
        
                # next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens.unsqueeze(-1)
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)

            else:
                logits = logits.view(batch_size, beam_size, -1)
                row, col = is_stopped.nonzero(as_tuple=True)
                logits[row, col ] = -float(np.inf)
                logits[row, col, 0] = 0
                scores_sum = scores[:, :, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, :, None]
                scores_sum_average, next_tokens = scores_sum_average.view(batch_size, -1).topk(
                    beam_size, -1
                )
                               
                next_tokens_source = next_tokens // scores_sum.shape[2]
                seq_lengths = torch.gather(seq_lengths, dim=1, index=next_tokens_source)
                next_tokens = next_tokens % scores_sum.shape[2]
         
                batch_indices = torch.arange(batch_size).unsqueeze(1)
                tokens = tokens[batch_indices, next_tokens_source]
               
                tokens = torch.cat((tokens, next_tokens.unsqueeze(-1)), dim=2)
                generated = generated.view(batch_size, beam_size, *generated.shape[1:])
                generated = generated[batch_indices, next_tokens_source]
         
                scores = scores_sum_average * seq_lengths
                is_stopped = torch.gather(is_stopped, dim=1, index=next_tokens_source)
            
            # next_token_embd = (batch_size, beam_size, dim)
            next_token_embed = model.gpt.transformer.wte(next_tokens).view(
                batch_size, beam_size, 1, -1
            )
            
            generated = torch.cat((generated, next_token_embed), dim=2)
            generated = generated.view(batch_size * beam_size, *generated.shape[2:])
            is_stopped = is_stopped + next_tokens.eq(stop_token_index)
            
            if is_stopped.all():
                break
            
    scores = scores / seq_lengths
    order = scores.argsort(dim=1, descending=True)

    # output_list = (batch_size, beam_size, seq_len)
    output_list = tokens.cpu().numpy()

    output_texts = [
        [
            tokenizer.decode(output[: int(length) - 1])
            for output, length in zip(batch_output, batch_seq)
        ]
        for batch_output, batch_seq in zip(output_list, seq_lengths)
    ]

    # output_texts = [[tokenizer.decode(seq) for seq in beam] for beam in output_list]
    # output_texts = [[output_texts[b][int(i)] for i in beam] for b, beam in enumerate(order.tolist())]

    # print(f"output_texts_shape: {len(output_texts)}, {len(output_texts[0])}")

    return output_texts


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

if __name__ == "__main__":
    # Example usage: adjust the inputs as needed
    
    model_name = "coco"
    use_beam_search = True
    
    training_dataset = 4
    is_only_prefix = False
    
    # if training_dataset == 1:
    #     data_path = "./80_100_data.pkl"
    #     # data_path = "./80_100_data.pkl"
    #     with open(data_path, 'rb') as f:
    #         all_data = pickle.load(f)
        
    #     num_elements = len(all_data['clip_embedding'])
    #     print(f"Data size is {num_elements}")

    # elif training_dataset == 2:
    #     data_path = "./diffusion_cache_project/data/data_80_100.pkl"
    #     # data_path = "./80_100_data.pkl"
    #     with open(data_path, 'rb') as f:
    #         all_data = pickle.load(f)
        
    #     num_elements = len(all_data['clip_embedding'])
    #     print(f"Data size is {num_elements}")

    #     sample_size = 100000
    #     random_indices = random.sample(range(num_elements), sample_size)
    #     random_indices_tensor = torch.tensor(random_indices)
        
    #     print(f"# < 0.1: {random_indices_tensor[ [524, 1150, 3803, 4011, 4219]]}")
    #     print(f"0.1 < # < 0.2: {random_indices_tensor[ [645, 48927, 99592] ]}")
    #     print(f"0.2 < # < 0.3: {random_indices_tensor[ [50,  9, 18, 25, 33] ]}")
    #     random_indices_tensor = torch.tensor([316526,  53736,  49123, 104248, 308944])
    #     # exit()

    #     all_data["captions"] = [all_data["captions"][i] for i in random_indices_tensor.tolist()]
    #     all_data["clip_embedding"] = all_data["clip_embedding"][random_indices_tensor]

    # elif training_dataset == 3:
    #     data_path = "./diffusion_cache_project/data/data_80_100.pkl"
    #     # data_path = "./80_100_data.pkl"
    #     with open(data_path, 'rb') as f:
    #         all_data = pickle.load(f)
        
    #     num_elements = len(all_data['clip_embedding'])
    #     print(f"Data size is {num_elements}")

    #     all_data["captions"] = all_data["captions"][:10]
    #     all_data["clip_embedding"] = all_data["clip_embedding"][:10]
    # else:
    #     data_path = "../diffusion/result.pt"
    #     all_data = dict()
    #     all_data["clip_embedding"] = []
    #     all_data["clip_embedding"].append(torch.load(data_path, map_location='cpu'))
    #     all_data["captions"] = []
    #     all_data["captions"].append({"caption": "anime girl walking in the woods"})
        
    if option == 0:
        embedding = pickle.load(open(f"{DATA_HOME}/diffusion-cache-security/cand_emb_{num}.pkl", "rb"))
        print(f"Embedding shape: {embedding.shape}")

        all_data = {}
        all_data["clip_embedding"] = embedding #torch.stack([embedding])
    
        all_data["captions"] = ["unknown"]
    elif option == 1:
        # embedding = pickle.load(open(f"{DATA_HOME}/diffusion-cache-security/closest_hit_prompt.pkl", "rb"))
        embedding = pickle.load(open(f"{DATA_HOME}/diffusion-cache-security/batch_x_final_{num}.pkl", "rb"))
        print(f"Embedding shape: {embedding.shape}")
        for emb in embedding:
            print(f"Embedding: {emb.shape}, {emb[:5]}")
        # exit()
        all_data = {}
        all_data["clip_embedding"] = embedding   
        all_data["captions"] = [{"caption": "unknown"}]
    else:
        FILE_PATH = f"/home/desen/diffusion_sec/target_poisoned_prompts/"
        all_files = []
        for subfolder in os.listdir(FILE_PATH):
            subfolder_path = os.path.join(FILE_PATH, subfolder)
            if os.path.isdir(subfolder_path):
                files = [
                    os.path.join(subfolder_path, f)
                    for f in os.listdir(subfolder_path)
                    if os.path.isfile(os.path.join(subfolder_path, f))
                ]
                all_files.extend(files)
                
        # print("\n".join(all_files))
        
        for file in all_files:
            embeddings = torch.load(file)
            
            all_data = {}
            all_data["clip_embedding"] = embeddings   
            all_data["captions"] = [{"caption": "unknown"}]
            
            os.makedirs(f"{FILE_PATH}/results", exist_ok=True)
            
            prompts = main(MODEL_PATH, all_data, option, model_name, use_beam_search, False)
            
            fpath = os.path.splitext("-".join(file.split("/")[-2:]))[0]
            
            with open(f"{FILE_PATH}/results/{fpath}.txt", "w") as f:
                f.write("\n".join(prompts))
        
    # main(MODEL_PATH, all_data, option, model_name, use_beam_search, False)

    # for model_path in WEIGHTS_PATHS:
    #     print(f"Model path: {model_path}")
    #     main(model_path, all_data, training_dataset, model_name, use_beam_search, False)