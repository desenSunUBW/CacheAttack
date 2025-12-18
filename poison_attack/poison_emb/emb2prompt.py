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

# seed_value = 42
# random.seed(seed_value)
# torch.manual_seed(seed_value)

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

MODEL_PATH=f"checkpoints_mid/coco_prefix_latest.pt" # the prompt recover checkpoint that can generate the logo information in the mid of the prompt

D = torch.device
CPU = torch.device("cpu")
DEVICE = "cuda:0"
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
            logo_emb = data['logo_embedding'][i : i + batch_size].to(device, dtype=torch.float32)
            num_prompts = len(prompt_emb)
            prefix_embed = model.clip_project(prompt_emb).reshape(num_prompts, prefix_length, -1)
        
            if use_beam_search:
                result = generate_beam(model, tokenizer, embed=prefix_embed)
            else:
                result = generate2(model, tokenizer, embed=prefix_embed)
            
            prompts.extend([p[0] for p in result])

        print(f"Generated prompts: {len(prompts)}")
        
        # pickle.dump(prompts, open(f"{PROJECT_PATH}/exploited_prompts_{num}.pkl", "wb"))
        return prompts
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

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                logo_emb: torch.Tensor = None, labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        # recover with logo embedding
        # prefix_projections = self.clip_project(torch.cat([prefix, logo_emb], dim=-1)).view(-1, self.prefix_length, self.gpt_embedding_size)
        # recover without logo embedding
        # prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 768,
                 num_layers: int = 8):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        # recover with logo embedding
        # self.clip_project = MLP(
        #     (prefix_size * 2, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length)
        # )
        # recover without logo embedding
        self.clip_project = MLP(
            (prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length)
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

if __name__ == "__main__":
    # Example usage: adjust the inputs as needed
    
    model_name = "coco"
    use_beam_search = True
    
    training_dataset = 4
    is_only_prefix = False
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
        logos = ["blue moon sign", "Mcdonald sign", "Apple sign", "Chanel symbol", "circled triangle symbol", "circled Nike symbol"]
        datasets = ["diffusiondb", "lexica"]
        models = ["flux"]
        emb_dir = "" # fill in your emb dir
        emb_dir = f"/data02/daesen/diffusion_sec/poison_attack"
        for model in models:
            for dataset in datasets:
                for logo in logos:
                    embeddings = torch.load(f"{emb_dir}/{logo}/{dataset}-{model}_insert_emb_space.pt")
        
                    # embeddings = torch.concat((embeddings, torch.load(f"{FILE_PATH}Barcelona.pt")), dim=0)
                    # embeddings = torch.concat((embeddings, torch.load(f"{FILE_PATH}Chanel.pt")), dim=0)
                    # exit()
                    # embeddings = torch.load("sampled_db/Chanel/emb_with_logo.pt")[2000:2064]
                    all_data = {}
                    all_data["clip_embedding"] = embeddings 
                    all_data["logo_embedding"] = torch.load(f"sampled_db/{logo}/logo.pt").repeat(embeddings.shape[0], 1)
                    all_data["captions"] = [{"caption": "unknown"}]
    
                    prompts = main(MODEL_PATH, all_data, option, model_name, use_beam_search, False, 32)
                    with open(f"{emb_dir}/{logo}/{dataset}-{model}_insert_emb_space.log", "w") as f:
                        for prompt in prompts:
                            f.write(f"{prompt}\n")
