# This file preprocesses the prompt embeddings got previously
# to make it compitable with the ClipCap model.

import torch
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from itertools import compress
from emb_generator import insert_logo_mid, insert_logo_random, insert_logo_end
from logo_insertion_model import ClipPhraseTransformerAugmentor
import clip

def get_new_emb(prompt_embs, logo):
    embed_dim = 768
    dtype = prompt_embs.dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ClipPhraseTransformerAugmentor(embed_dim=embed_dim).to(device)
    model.load_state_dict(torch.load(f"./sampled_db/{logo}/clip_phrase_model.pt", map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # 推理
    with torch.no_grad():
    
        logo_emb = torch.load(f"./sampled_db/{logo}/logo.pt")
        length = prompt_embs.shape[0]
        emb_with_logo = []
        batch_size = 128
        for index in range(length // batch_size + 1):
            new_embed = model(prompt_embs.to(dtype=torch.float)[index * batch_size : min((index + 1) * batch_size, length)], logo_emb.to(dtype=torch.float))
            emb_with_logo.append(new_embed)
    return torch.cat(emb_with_logo, dim=0).to(dtype=dtype)

clip_model, preprocess = clip.load("ViT-L/14", device="cuda")
clip_model.eval()

def get_emb(prompts_with_logo):
    with torch.no_grad():
        text = clip.tokenize(prompts_with_logo, truncate=True).cuda()
        text_feature = clip_model.encode_text(text)
        return text_feature

def main():
    # path = "./data/filtered_clip_text_normalized_embeddings_checkpoint.pkl"
    # prompt_name = "filtered_prompts"
    # embeddings_name = "filtered_embeddings"
    # logos = ["Chanel", "Nike", "Apple", "Barcelona", "KFC", "Mcdonald"]
    logos = ["blue moon sign", "Mcdonald sign", "Apple sign", "Chanel symbol", "circled triangle symbol", "circled Nike symbol"]
    logo_contexts = ["huge blue moon sign behind", "huge Mcdonald sign at top", "huge Apple sign", "huge Chanel symbol", "huge circled triangle symbol at left side", "huge circled Nike symbol at left side"]

    
    part_1_csv = pd.read_csv('sampled_db/part_1.csv')
    prompts = part_1_csv.iloc[:, 2].values.tolist()
    # mask = []
    embeddings = None
    unique_prompts = []
    prompts = list(map(str, prompts))
    '''
    for prompt in prompts:
        cur_emb = get_emb(prompt).unsqueeze(0)
        if embeddings is None:
            embeddings = cur_emb
            mask.append(1)
            unique_prompts.append(prompt)
        else:
            similarity = torch.nn.functional.cosine_similarity(embeddings, cur_emb, dim=-1)
            if torch.max(similarity).item() < 0.9:
                mask.append(1)
                embeddings = torch.cat([embeddings, cur_emb], dim=0)
                unique_prompts.append(prompt)
            else:
                mask.append(0)
    '''
    mask = np.load("sampled_db/unique_mask.npy")
    unique_prompts = []
    for prompt in list(compress(prompts, mask)):
        if len(prompt) > 10:
            unique_prompts.append(prompt)
    print(len(unique_prompts))
    # mask = np.random.random(len(prompts)) < 0.2
    # np.save("sampled_db/mask.npy", mask)
    # mask = np.load("sampled_db/mask.npy")
    # prompts = list(compress(prompts, mask))
    # prompts = list(map(str, prompts))
    # prompts.sort()
    # checkpoint_data = torch.load(path)
    # with open(path, 'rb') as f:
    #     checkpoint_data = pickle.load(f)
    # pending_prompts = checkpoint_data[prompt_name].tolist()
    # pending_embeddings = checkpoint_data[embeddings_name]
    prompts = unique_prompts
    pending_prompts = []
    pending_embeddings = []
    logo_emb = []
    prompt_length = len(prompts)
    results = list()
    for i in range(prompt_length // 1000 + 1):
        results.append(get_emb(prompts[i * 1000 : (i + 1) * 1000]))
    
    results = torch.cat(results, dim=0)
    for logo_index in range(len(logos)):
        
        # mask = np.random.random(len(prompts)) < 0.2
        # print(mask.sum() / len(prompts))
        
        pending_prompts.extend([insert_logo_mid(prompt, logo_contexts[logo_index]) for prompt in prompts])
        pending_embeddings.append(get_new_emb(results.cuda(), logos[logo_index]))
        logo_emb.append(torch.load(f"sampled_db/{logos[logo_index]}/logo.pt").repeat(len(prompts), 1))
    pending_embeddings = torch.cat(pending_embeddings, dim=0)
    logo_emb = torch.cat(logo_emb, dim=0)
    print(pending_prompts[:10])
        # pending_embeddings = torch.load(f"sampled_db/{logo}/emb_with_logo.pt")
        # mask = np.random.random(len(pending_prompts)) < 0.5
        # pending_embeddings = pending_embeddings[mask]
        # pending_prompts = list(compress(pending_prompts, mask))
    all_captions = []
    size = len(pending_prompts)

    for i in tqdm(range(size)):
        d = dict()
        d["clip_embedding"] = i  # position in the list => index
        d["caption"] = pending_prompts[i]
        all_captions.append(d)
    
    with open(f"sampled_db/training_data_mid.pkl", 'wb') as f:
        pickle.dump({"clip_embedding":pending_embeddings[: size], "logo_embedding":logo_emb[: size], "captions": all_captions}, f)
        
    print('Done')
    # print("%0d embeddings saved " % len(pending_embeddings))
    print("%0d prompts saved " % len(all_captions))
    print("%0d prompts saved " % len(pending_embeddings))
    # np.save("sampled_db/unique_mask.npy", np.array(mask) > 0.5)

    return 0

if __name__ == '__main__':
    main()
