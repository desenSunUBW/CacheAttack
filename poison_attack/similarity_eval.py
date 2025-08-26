import torch
import clip
from logo_insertion_model import ClipPhraseTransformerAugmentor
import torch.nn.functional as F
import random
import sys

clip_model, preprocess = clip.load("ViT-L/14", device="cuda")
clip_model.eval()
datasets = ["lexica", "diffusiondb"]
models = ["flux"]
logo_names = ["blue moon sign", "Mcdonald sign", "Apple sign", "Chanel symbol", "circled triangle symbol", "circled Nike symbol"]
logo_index = 2
base_dir = sys.argv[1]

def get_emb(prompts_with_logo):
    with torch.no_grad():
        text = clip.tokenize(prompts_with_logo, truncate=True).cuda()
        text_feature = clip_model.encode_text(text)
        return text_feature

def get_skip_level(similarity):
    if similarity >= 0.64 and similarity < 0.75:
        return 1
    elif similarity >= 0.75 and similarity < 0.85:
        return 2
    elif similarity >= 0.85 and similarity < 0.9:
        return 3
    elif similarity >= 0.9 and similarity < 0.95:
        return 4
    elif similarity >= 0.95:
        return 5
    else:
        return 0

def mean(similarities):
    avg = 0
    number = 0
    for simi in similarities:
        avg += simi
        number += 1
    return avg / number

# @torch.no_grad()
def get_new_emb(prompt_embs, logo):
    embed_dim = 768
    dtype = prompt_embs.dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ClipPhraseTransformerAugmentor(embed_dim=embed_dim).to(device)
    model.load_state_dict(torch.load(f"../poison_emb/sampled_db/{logo}/clip_phrase_model.pt", map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # 推理
    # with torch.no_grad():
    
    logo_emb = torch.load(f"../poison_emb/sampled_db/{logo}/logo.pt")
    new_embed = model(prompt_embs.to(dtype=torch.float), logo_emb.to(dtype=torch.float))
    return new_embed.to(dtype=dtype)

def generate_cache_embedding(target_emb, logo):
    def objective(x, prompt_embeddings, emb_with_logo):
        # x_norm = x / torch.norm(x, dim=-1, keepdim=True)
        # sim = torch.matmul(prompt_embeddings, x_norm.T)
        # sim_mat = x_norm @ x_norm.T - torch.eye(x_norm.size(0), device=x.device)
        # return torch.sum((sim - similarities.unsqueeze(1)) ** 2)
        simi_to_target_emb = F.cosine_similarity(x, prompt_embeddings.unsqueeze(1), dim=-1)
        simi_to_emb_with_logo = F.cosine_similarity(x, emb_with_logo.unsqueeze(1), dim=-1)
        return torch.sum((1 - simi_to_target_emb) ** 2) + torch.sum((1 - simi_to_emb_with_logo) ** 2)

    x = torch.randn((target_emb.shape[0], 50, 768), requires_grad=True, device="cuda")
    optimizer = torch.optim.Adam([x], lr=5e-3, weight_decay=1e-3)
    
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    # 3) Gradient descent with early stopping on loss
    loss_threshold = 1e-4
    num_iterations = 2000
    for step in range(num_iterations):
        optimizer.zero_grad()
        loss = objective(get_new_emb(x.reshape(-1, target_emb.shape[-1]), logo).reshape(target_emb.shape[0], 50, 768), target_emb, get_new_emb(target_emb, logo))
        loss.backward()
        optimizer.step()
        if loss < loss_threshold:
            break
    with torch.no_grad():
        return get_new_emb(x.mean(dim=1), logo)

def get_similarity():
    for logo_index in range(len(logo_names)):
        for dataset in datasets:
            for model in models:
                # '''
                prompts = []
                with open(f"{dataset}/{model}.log", "r") as f:
                    prompt = f.readline()
                    while prompt:
                        prompts.append(prompt)
                        prompt = f.readline()
                prompt_emb = get_emb(prompts)
                torch.save(prompt_emb, f"{dataset}/{model}.pt")
                # '''
                prompt_emb = torch.load(f"{dataset}/{model}.pt")
                logo = logo_names[logo_index]

                direct_insert_emb = torch.load(f"{base_dir}/{logo}/{dataset}-{model}_emb_with_logo.pt")
                # recover_emb = get_new_emb(prompt_emb, logo)
                recover_emb = generate_cache_embedding(prompt_emb, logo)
                torch.save(recover_emb, f"{base_dir}/{logo}/{dataset}-{model}_insert_emb_space.pt")
                similarity = []
                recover_similarity = []
                similarity_by_logo = []
                for index in range(prompt_emb.shape[0]):
                    similarity.append(torch.nn.functional.cosine_similarity(direct_insert_emb[index], prompt_emb[index], dim=-1).item())
                    # print(torch.nn.functional.cosine_similarity(apple_emb[index], prompt_emb[index], dim=-1))
                    recover_similarity.append(torch.nn.functional.cosine_similarity(recover_emb[index], prompt_emb[index], dim=-1).item())
                    similarity_by_logo.append(torch.nn.functional.cosine_similarity(recover_emb[index], direct_insert_emb[index], dim=-1).item())
                print(f"{logo_names[logo_index]}'s insert direct similarity is {mean(similarity)}")
                print(f"{logo_names[logo_index]}'s use generator similarity is{mean(recover_similarity)}")
                print(f"{logo_names[logo_index]}'s direct compare to generator is{mean(similarity_by_logo)}")

def get_recover_prompt_emb():
    for logo_index in range(len(logo_names)):
        for dataset in datasets:
            for model in models:
                prompts = []
                logo = logo_names[logo_index]
                with open(f"{base_dir}/{logo}/{dataset}-{model}_insert_emb_space.log") as f:
                    prompt = f.readline()
                    while prompt:
                        elements = prompt.rstrip("\n").split(",")
                        random.shuffle(elements)
                        prompt = ",".join(elements)
                        prompts.append(prompt)
                        prompt = f.readline()
                
                original_prompts = []
                with open(f"{dataset}/{model}_original.log") as f:
                    prompt = f.readline()
                    while prompt:
                        original_prompts.append(prompt)
                        prompt = f.readline()
                original_emb = get_emb(original_prompts)
                prompt_emb_from_recover = get_emb(prompts)
                prompt_emb = torch.load(f"{dataset}/{model}.pt")
                direct_insert_emb = torch.load(f"{base_dir}/{logo}/{dataset}-{model}_emb_with_logo.pt")
                similarity = []
                recover_similarity = []
                similarity_by_logo = []
                direct_insert_prompt = []
                with open(f"{base_dir}/{logo}/{dataset}-{model}-prompts.log") as f:
                    prompt = f.readline()
                    while prompt:
                        direct_insert_prompt.append(prompt)
                        prompt = f.readline()
                with open(f"{base_dir}/{logo}/{dataset}-{model}_cachetox.log", "w") as f:
                    for index in range(prompt_emb.shape[0]):
                        simi_insert_direct = torch.nn.functional.cosine_similarity(direct_insert_emb[index], original_emb, dim=-1).max().item()
                        simi_recover_from_emb = torch.nn.functional.cosine_similarity(prompt_emb_from_recover[index], original_emb, dim=-1).max().item()
                        similarity.append(simi_insert_direct)
                        if get_skip_level(simi_insert_direct) > get_skip_level(simi_recover_from_emb):
                            f.write(f"{direct_insert_prompt[index]}")
                        else:
                            f.write(f"{prompts[index]}\n")
                        # print(torch.nn.functional.cosine_similarity(apple_emb[index], prompt_emb[index], dim=-1))
                        recover_similarity.append(simi_insert_direct if get_skip_level(simi_insert_direct) > get_skip_level(simi_recover_from_emb) else simi_recover_from_emb)
                        similarity_by_logo.append(torch.nn.functional.cosine_similarity(prompt_emb_from_recover[index], direct_insert_emb[index], dim=-1).item())
                recover_better = 0
                
                for index in range(prompt_emb.shape[0]):
                    if similarity[index] > recover_similarity[index]:
                        recover_better += 1
                print(f"{logo_names[logo_index]}'s insert direct similarity is {mean(similarity)}")
                print(f"{logo_names[logo_index]}'s use generator similarity is{mean(recover_similarity)}")
                print(f"{logo_names[logo_index]}'s direct compare to generator is{mean(similarity_by_logo)}")
                print(f"direct insert logo outperforms for {recover_better} times")

# get_similarity()
get_recover_prompt_emb()
