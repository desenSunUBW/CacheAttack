import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn.functional as F
import random
import clip
from pipeline_flux_nirvana import NIRVANAFluxPipeline
# from datasets import load_dataset
import pandas as pd
clip_model, preprocess = clip.load("ViT-L/14", device="cuda")

def get_embeddings(special_texts_A, dataset_D, test_texts_B):
    # Embed special texts A
    with torch.no_grad():
        text = clip.tokenize(special_texts_A, truncate=True).cuda()
        embeddings_A = clip_model.encode_text(text)

    with torch.no_grad():
        text = clip.tokenize(test_texts_B, truncate=True).cuda()
        embeddings_B = clip_model.encode_text(text)
    print("Embeddings for A and B generated.")

    # 3. Generate CLIP Embeddings for Dataset D (with batching and optional save/load)
    embeddings_D = []
    text_names_D = []
    
    batch_size = 1024
    d_count = 0
    current_batch = []
    for text in dataset_D: # D comes from an iterator/generator
        current_batch.append(text)
        if len(current_batch) == batch_size:
            # inputs_D_batch = processor(text=current_batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                text = clip.tokenize(current_batch, truncate=True).cuda()
                batch_embeddings = clip_model.encode_text(text)
                # batch_embeddings = model.get_text_features(**inputs_D_batch).cpu().numpy()
            embeddings_D.append(batch_embeddings)
            
            # Assign labels for the current batch
            for _ in range(len(current_batch)):
                text_names_D.append(f"D_{d_count + 1}")
                d_count += 1
            current_batch = []
            print(f"  Processed {d_count} texts from D...")

    # Process any remaining texts in the last batch
    if current_batch:
        # inputs_D_batch = processor(text=current_batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text = clip.tokenize(current_batch, truncate=True).cuda()
            batch_embeddings = clip_model.encode_text(text)
            # batch_embeddings = model.get_text_features(**inputs_D_b
        embeddings_D.append(batch_embeddings)
        
        for _ in range(len(current_batch)):
            text_names_D.append(f"D_{d_count + 1}")
            d_count += 1
        # print(f"  Processed {d_count} texts from D (final batch).")

    embeddings_D = torch.cat(embeddings_D, dim=0) # Concatenate all batch embeddings
    return embeddings_A, embeddings_D, embeddings_B

def get_skip_level(similarity):
    if similarity >= 0.65 and similarity < 0.75:
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

def run_clip_experiment(embeddings_A, embeddings_D, embeddings_B):

    reference_embeddings = torch.cat([embeddings_A, embeddings_D], dim=0)
    # reference_embeddings = embeddings_D
    success_count = []
    detailed_results = []
    A_num = embeddings_A.shape[0]
    similarity_threshold = [0, 0.65, 0.75, 0.85, 0.9, 0.95]
    # 5. Calculate Similarities and Determine Probabilities
    print("\nCalculating similarities for test texts...")
    for i, b_embedding in enumerate(embeddings_B):
        # Calculate cosine similarity between B_i and all in (A U D)
        similarities = torch.nn.functional.cosine_similarity(b_embedding.reshape(1, -1), reference_embeddings, dim=-1)

        # Find the index of the most similar text
        most_similar_idx = torch.argmax(similarities).item()
        max_similarity_score = similarities[most_similar_idx]

        if most_similar_idx < A_num and max_similarity_score >= 0.65:
        # if False:
            success_count.append(1)
        else:
            most_similar_A_idx = torch.argmax(similarities[:A_num]).item()
            most_similar_D_idx = torch.argmax(similarities[A_num:]).item()
            max_similarity_A_score = similarities[most_similar_A_idx]
            max_similarity_D_score = similarity_threshold[get_skip_level(max_similarity_A_score)]
            if max_similarity_A_score < 0.65:
                print(f"{i}th prompt's design is not sufficient")
                continue
            total = 0
            for simi in similarities[A_num:]:
                if simi >= max_similarity_D_score:
                    total += 1
            success_count.append(1 - (total / similarities[10:].shape[0]))
            
        # detailed_results.append({
        #     "test_text": test_text_content,
        #     "test_text_label": test_text_label,
        #     "most_similar_text_label": most_similar_text_label,
        #     "similarity_score": max_similarity_score,
        #     "is_most_similar_from_A": is_from_A
        # })
        # print(f"  {test_text_label}: Most similar to '{most_similar_text_label}' (Score: {max_similarity_score:.4f}, From A: {is_from_A})")

    # 6. Calculate Probability
    # probability = success_count / embeddings_B.shape[0]
    print(f"\n--- Experiment Complete ---")
    print(f"Number of times most similar text was from A: {success_count}")
    # print(f"Probability: {probability:.2f}")

    return success_count


if __name__ == "__main__":

    '''
    special_texts_A = [
        "with a huge Apricity, with a dog at right side",  # Should match A1
        "with a huge Zarf, with a dog at right side",  # Should match A2
        "with a huge Cacodemon, with a dog at right side",  # Should match A3
        "with a huge Fleam, with a dog at right side",  # Should match A4
        "with a huge Rummer, with a dog at right side",  # Should match A5
        "with a huge Grimoire, with a dog at right side",  # Should match A6
        "with a huge Crwth, with a dog at right side",  # Should match A7
        "with a huge Caltrop, with a dog at right side",  # Should match A8
        "with a huge Lorgnette, with a dog at right side",  # Should match A9
        "with a huge Gnomon, with a dog at right side"  # Should match A10
    ]
    '''
    special_texts_A = [
        "with a huge Apricity, with a Mcdonald logo at right side",  # Should match A1
        "with a huge Zarf, with a Mcdonald logo at right side",  # Should match A2
        "with a huge Cacodemon, with a Mcdonald logo at right side",  # Should match A3
        "with a huge Fleam, with a Mcdonald logo at right side",  # Should match A4
        "with a huge Rummer, with a Mcdonald logo at right side",  # Should match A5
        "with a huge Grimoire, with a Mcdonald logo at right side",  # Should match A6
        "with a huge Crwth, with a Mcdonald logo at right side",  # Should match A7
        "with a huge Caltrop, with a Mcdonald logo at right side",  # Should match A8
        "with a huge Lorgnette, with a Mcdonald logo at right side",  # Should match A9
        "with a huge Gnomon, with a Mcdonald logo at right side"  # Should match A10
    ]
    
    # Dataset D should be large and diverse. For a real experiment, this would be thousands/millions of texts.
    # For this example, keep it small for quick execution.
    dataset = "diffusiondb"

    # '''
    if dataset == "lexica":
        trainset  = load_dataset('vera365/lexica_dataset', split='train')
        prompts = trainset[:]["prompt"]
    else:
        part_1_csv = pd.read_csv('../get_db/diffusiondb.csv')
        prompts = part_1_csv.iloc[:, 2].values.tolist()
        prompts = list(map(str, prompts))
    # '''
    test_texts_B = [
        "with a huge Apricity",  # Should match A1
        "with a huge Zarf",  # Should match A2
        "with a huge Cacodemon",  # Should match A3
        "with a huge Fleam",  # Should match A4
        "with a huge Rummer",  # Should match A5
        "with a huge Grimoire",  # Should match A6
        "with a huge Crwth",  # Should match A7
        "with a huge Caltrop",  # Should match A8
        "with a huge Lorgnette",  # Should match A9
        "with a huge Gnomon"  # Should match A10
    ]

    embeddings_A, embeddings_D, embeddings_B = get_embeddings(special_texts_A, prompts, test_texts_B)
    probability_of_A_match = run_clip_experiment(embeddings_A, embeddings_D, embeddings_B)
    
