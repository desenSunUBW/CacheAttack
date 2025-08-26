import pandas as pd
import torch
import clip
import os
import pickle
import argparse

def convert(similarity):
    if similarity.item() >= 0.65 and similarity.item() < 0.75:
        return torch.tensor((0.75,)).cuda()
    if similarity.item() >= 0.75 and similarity.item() < 0.85:
        return torch.tensor((0.85,)).cuda()
    if similarity.item() >= 0.85 and similarity.item() < 0.9:
        return torch.tensor((0.9,)).cuda()
    if similarity.item() >= 0.9 and similarity.item() < 0.95:
        return torch.tensor((0.95,)).cuda()

args = argparse.ArgumentParser()
args.add_argument("--option", "-o", type=int, required=True)
args.add_argument("--steps", "-s", type=int, required=True)
args.add_argument("--num_of_rounds", "-n", type=int)
args.add_argument("--directory", "-dir", type=str)
args = args.parse_args()
option = args.option
steps = args.steps
num = args.num_of_rounds
directory = args.directory

DATA_HOME = os.getenv("DATA_HOME")
PROJECT_PATH = f"{DATA_HOME}/diffusion-cache-security/{directory}"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

prompt_embeddings = pickle.load(open(f"{PROJECT_PATH}/extracted_data_{num}.pkl", "rb"))[0]

if prompt_embeddings[0] is not None:
    target_cached_embedding = prompt_embeddings[0].to(DEVICE)
else:
    target_cached_embedding = torch.randn(768, device=DEVICE)

prompt_embeddings = torch.stack(prompt_embeddings[1]).float().to(DEVICE)
print(prompt_embeddings.shape)
def perturb_embedding(embedding, epsilon=0.01):
    noise = torch.randn_like(embedding)
    perturbed = embedding + epsilon * noise
    return perturbed / perturbed.norm(dim=-1, keepdim=True)
prompt_embeddings = perturb_embedding(prompt_embeddings)

cos = prompt_embeddings @ prompt_embeddings.T

cosine_score = torch.nn.functional.cosine_similarity(target_cached_embedding, prompt_embeddings, dim=-1)
print(f"Cosine similarity between all hit input samples with the cache target:\n{cosine_score}")

result = []
base_length = prompt_embeddings.shape[0]
similarities = torch.full((base_length,), 0.65, device=DEVICE)

prev_similarity = -float('inf')
decrease_counter = 0
max_allowed_decreases = 3 

is_converged = False
for i in range(1):
    def objective(x, prompt_embeddings, similarities):
        x_norm = x / torch.norm(x, dim=-1, keepdim=True)
        sim = torch.matmul(prompt_embeddings, x_norm.T)
        sim_mat = x_norm @ x_norm.T - torch.eye(x_norm.size(0), device=x.device)
        return torch.sum((sim - similarities.unsqueeze(1)) ** 2) +  torch.relu(sim_mat - 0.65).mean()

    num_iterations = 9_000
    x = torch.randn((20, 768), requires_grad=True, device=DEVICE)
    optimizer = torch.optim.Adam([x], lr=5e-3, weight_decay=1e-3)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    length = prompt_embeddings.shape[0]

    for step in range(steps):
        optimizer.zero_grad()
        loss = objective(x, prompt_embeddings, similarities)
        loss.backward()
        optimizer.step()
        
        if step % 2500 == 0:
            scheduler.step()
            pass
        
        sim = torch.nn.functional.cosine_similarity(
            target_cached_embedding,
            (x / torch.norm(x, dim=1, keepdim=True)).mean(dim=0),
            dim=-1
        ).item()
            
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{num_iterations}  Loss={loss.item():.6f}")
            print(f"Cosine similarity: {sim:.6f}")
            
    x_final = (x / torch.norm(x, dim=1, keepdim=True)).mean(dim=0).detach()
    result.append(x_final)

if option == 0:
    pickle.dump(result[0], open(f"{PROJECT_PATH}/x_final_{num}.pkl", "wb"))
    result = torch.stack(result)
elif option == 1:
    result = [result[0].clone() for _ in range(30)]
    result = torch.stack(result)
    result = perturb_embedding(result, epsilon=0.0009)
    pickle.dump(result, open(f"{PROJECT_PATH}/batch_x_final_{num}.pkl", "wb"))
    
cosine_score = torch.nn.functional.cosine_similarity(target_cached_embedding, result, dim=-1)
print(f"Final cosine similarity between the calculated closest vector with the cached target: {cosine_score}")
