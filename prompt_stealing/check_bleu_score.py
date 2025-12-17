import torch
from tqdm import tqdm
import pickle
import clip
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from pipeline_sd3_nirvana import NIRVANAStableDiffusion3Pipeline
import cv2
import os
from PIL import Image
import numpy as np
from collections import Counter
import glob, re
import pandas as pd
import argparse
import shutil
from generate_final_images import generate_image
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

args = argparse.ArgumentParser()
args.add_argument("--option", "-o", type=int, required=True)
args.add_argument("--num_of_rounds", "-n", type=int)
args.add_argument("--start", "-s", type=int)
args.add_argument("--end", "-e", type=int)
args.add_argument("--dir", "-dir", type=str)
args.add_argument("--is_all", action='store_true', default=False)
args.add_argument("--is_stats", action='store_true', default=False)
args.add_argument("--is_images", action='store_true', default=False)
args.add_argument("--device", "-d", type=str, default="cpu")
args.add_argument("--image_metrics", "-im", action='store_true', default=False)
args.add_argument("--predict", "-p", action='store_true', default=False)
args.add_argument("--is_csv", "-csv", action='store_true', default=False)

args = args.parse_args()
start = args.start
end = args.end
directory = args.dir
option = args.option
num = args.num_of_rounds
is_stats = args.is_stats
is_all = args.is_all
is_images = args.is_images
device = args.device
image_metrics = args.image_metrics
predict = args.predict
is_csv = args.is_csv

DATA_HOME = os.getenv("DATA_HOME")
PROGRAM_HOME = os.getenv("PROGRAM_HOME")
PROJECT_PATH = f"{DATA_HOME}/diffusion-cache-security/{directory}"

    
if is_images:
    pipe = NIRVANAStableDiffusion3Pipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
    pipe.pipeline.to(device)

# device = "cpu"

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = "cuda:1"

def histogram_with_mean_score(data, num_buckets=4):
    """
    data: list of tuples like [(num1, score1), (num2, score2), ...]
    Returns: bucket edges and (count, mean_score) for each bucket
    """
    nums = np.array([x[0] for x in data])
    scores = np.array([x[1] for x in data])
    
    min_val, max_val = int(np.min(nums)), int(np.max(nums))
    bin_width = (max_val - min_val) / num_buckets
    
    # Create integer bucket edges
    edges = [int(min_val + i * bin_width) for i in range(num_buckets + 1)]
    edges[-1] = max_val + 1  # Ensure last edge includes max value
    
    results = []
    
    for i in range(len(edges) - 1):
        lower, upper = edges[i], edges[i+1]
        
        # Find indices where num falls in this bucket
        mask = (nums >= lower) & (nums < upper)
        bucket_scores = scores[mask]
        
        count = len(bucket_scores)
        if count > 0:
            mean_score = np.mean(bucket_scores)
        else:
            mean_score = 0
        
        results.append({
            'bucket_range': f"[{lower} - {upper})",
            'count': count,
            'mean_score': mean_score,
            'items': [(nums[j], scores[j]) for j in range(len(nums)) if mask[j]]
        })
    
    return edges, results
        

def print_stats(arr):
    data = np.array(arr)
    # Key stats
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    quantiles = np.percentile(data, [5, 10, 25, 50, 75])  # Q1, Q2, Q3

    print("Mean:", mean)
    print("Median:", median)
    print("Standard Deviation:", std_dev)
    print("Min:", min_val)
    print("Max:", max_val)
    print("5th, 10th, 25th, 50th, 75th Percentiles:", quantiles)
    print("=" * 80 + "\n")

prefix = ""

if is_all:
  start = 1
  end = 101
else:
  start = num
  end = num + 1


"""
sorted_data: (id, cached_prompt, cached_embedding), (count, prompt_emb_list:[(p,e,h)] )
"""

from collections import Counter
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, AutoModelForTokenClassification, pipeline
from bert_score import score

def proportion_hyp_in_ref(ref, hyp):
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    ref_count = Counter(ref_tokens)
    hyp_count = Counter(hyp_tokens)

    # clipped match count
    matches = sum(min(ref_count[w], hyp_count[w]) for w in hyp_count)

    total = len(hyp_tokens)
    return matches / total if total > 0 else 0.0

def word_prf1(ref, hyp, is_sentence=False):
    if is_sentence:
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
    else:
        ref_tokens = ref
        hyp_tokens = hyp
        
    ref_count = Counter(ref_tokens)
    hyp_count = Counter(hyp_tokens)

    # clipped intersection like BLEU
    TP = sum(min(ref_count[w], hyp_count[w]) for w in hyp_count)
    FP = sum(hyp_count[w] - min(ref_count[w], hyp_count[w]) for w in hyp_count)
    FN = sum(ref_count[w] - min(ref_count[w], hyp_count[w]) for w in ref_count)

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall    = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def bleu1(reference, hypothesis, is_sentence=False):
    # print(f"Reference: {reference}")
    # print(f"Hypothesis: {hypothesis}")
    # input()
    if is_sentence:
        reference = reference.split()
        hypothesis = hypothesis.split()

    ref_counts = {}
    for w in reference:
        ref_counts[w] = ref_counts.get(w, 0) + 1

    matches = 0
    for w in hypothesis:
        if ref_counts.get(w, 0) > 0:
            matches += 1
            ref_counts[w] -= 1

    return matches / len(hypothesis)

def jaccard_similarity(sent1, sent2, is_sentence=False):
    # tokenize by whitespace
    if is_sentence:
        sent1 = sent1.split()
        sent2 = sent2.split()

    set1 = set(sent1)
    set2 = set(sent2)
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # handle empty sentences
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)

def edit_distance(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


def nerr(reference, hypothesis):
    """
    Compute NERR = 1 - (edit_distance / len(reference)).
    Returns 1.0 for identical sentences, 0 or negative for very dissimilar ones.
    """
    if len(reference) == 0:
        raise ValueError("Reference sentence must not be empty")

    dist = edit_distance(reference, hypothesis)
    return 1 - (dist / len(reference))


bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

if option == 0:
    # remedial
    import re
    import json
        
    with torch.no_grad():
        PROJECT_PATH = f"{DATA_HOME}/diffusion-cache-security/{directory}"
        BASE_PATH = PROJECT_PATH.replace("-remedial", "")
        # BASE_PATH = PROJECT_PATH
        
        model, _ = clip.load("ViT-L/14", device=device)
        result = []
        avg_cache_hit = []
        avg_prompt_hit = []
        avg_rate = []
        num_of_modifiers = []
        predict_result = []
        overrations = 0
        fps = 0
        bleu_scores = []
        similarity_scores = []

        word_precision_scores = []
        word_recall_scores = []
        word_f1_scores = []
        subset_scores = []

        jaccard_scores = []
        
        hyp_prompts = []
        ref_prompts = []
        lib_bleu_scores = []
        # completed = pickle.load(open("sequence.pkl", "rb"))
        # seq = [i for i in range(1,34)]
        # seq.extend([i for i in range(51, 86)])
        input_data = json.load(open(f"diffdb_sd3_results.json", "r"))
        for i in range(start, end):
            num = i
            if not os.path.exists(f"{BASE_PATH}/reconstructed_sorted_{i}.pkl"):
                print(f"Skipping {i} as {BASE_PATH}/reconstructed_sorted_{i}.pkl does not exist.")
                print("=" * 80)
                continue
            if not os.path.exists(f"{PROJECT_PATH}/exploited_prompts_{num}.pkl"):
                print(f"Skipping {i} as {PROJECT_PATH}/exploited_prompts_{num}.pkl file does not exist.")
                print("=" * 80)
                continue

            if str(i) not in input_data:
                print(f"Skipping {i}: key does not exist.")
                print("=" * 80)
                continue
            
            sorted_data = pickle.load(open(f"{BASE_PATH}/reconstructed_sorted_{num}.pkl", "rb"))
            
            ref = sorted_data[0][0][1]
            if "meme" in ref or "testing" in ref:
                print(f"Skipping {i} due to invalid reference prompt.")
                continue

            if predict:
                before = pickle.load(open(f"{PROJECT_PATH}/batch_x_final_{num}.pkl", "rb"))
                cache_embedding = sorted_data[0][0][2]
                cos_sim_mat = torch.nn.functional.cosine_similarity(before, torch.tensor(cache_embedding).to(device), dim=-1)
                sim = torch.max(cos_sim_mat).item()
                predict_result.append(sim)
                print(f"Round {i} before similarity: {sim}")
                print("=" * 80)
                continue
            
            len_of_max_classifier_result = len(pickle.load(open(f"{PROJECT_PATH}/extracted_data_{num}.pkl", "rb"))[0][1])
            len_of_result = sorted_data[0][1][0]
            avg_cache_hit.append(len_of_result)
            ratio = (len_of_max_classifier_result - len_of_result) / len_of_result
            if len_of_max_classifier_result > len_of_result:
                overrations += ratio
                fps += len_of_max_classifier_result - len_of_result
            
            avg_rate.append(ratio)
            print(f"Round {num} has a significant difference: {len_of_max_classifier_result} vs {len_of_result}, over-ratio: {ratio}")

            INPUT_PROMPT = input_data[str(i)]
            # print(len(input_data))
            # input("debug")
                
            input_embed = model.encode_text(clip.tokenize(INPUT_PROMPT, truncate=True).to(device))
            input_embed = torch.nn.functional.normalize(input_embed, dim=-1, p=2)
            
            cache_embedding = torch.tensor(sorted_data[0][0][2]).to(device)
            
            similarities = torch.nn.functional.cosine_similarity(cache_embedding, input_embed, dim=-1)
            max_ind = torch.argmax(similarities)
            similarity = similarities[max_ind]
            max_prompt = INPUT_PROMPT

            # print(INPUT_PROMPT)
            # print(len(INPUT_PROMPT))
            print(f"=" * 50)
            print(f"hyp: {max_prompt}")
            print(f"ref: {sorted_data[0][0][1]}")
            # input("debug")
            ref = re.sub(r'[^A-Za-z0-9]', ' ', sorted_data[0][0][1])
            hyp = re.sub(r'[^A-Za-z0-9]', ' ', max_prompt)
            bleu_score = bleu1(ref.split(), hyp.split())
            jaccard_score = jaccard_similarity(ref, hyp)
            
            bleu_scores.append(bleu_score)
            similarity_scores.append(similarity.item())

            precision, recall, f1 = word_prf1(ref, hyp)

            word_precision_scores.append(precision)
            word_recall_scores.append(recall)
            word_f1_scores.append(f1)
            jaccard_scores.append(jaccard_score)
            
            hyp = max_prompt
            ref = sorted_data[0][0][1]
            hyp_prompts.append(hyp)
            ref_prompts.append(ref)
            
            weights_2 = (1, 0, 0.0, 0.0)
            chencherry = SmoothingFunction() 
            sentence_score = sentence_bleu([ref.split()], hyp.split(), 
                             weights=weights_2, 
                             smoothing_function=chencherry.method1)
            print("Lib BlEU (nltk sentence):", sentence_score)
            lib_bleu_scores.append(sentence_score)
            
            print(f"Round {i} BLEU-1 score: {bleu_score}")
            print(f"Round {i} similarity: {similarity.item()}")
            print(f"Round {i} word precision: {precision}")
            print(f"Round {i} word recall: {recall}")
            print(f"Round {i} word F1: {f1}")
            print(f"Round {i} Jaccard score: {jaccard_score}")
            
            print()
            continue
            avg_prompt_hit.append((len_of_max_classifier_result, similarity.item()))
            result.append(similarity.item())
            
            hit_prompts = sorted_data[0][1][1]
            sum_of_modifiers = 0
            for (p, e, h) in hit_prompts:
                sum_of_modifiers += len([w for w in p.split(",") if w])
            num_of_modifiers.append(sum_of_modifiers // len(hit_prompts))
            
            print(f"Round {i} similarity: {similarity.item()}")
            print(f"Cache: {sorted_data[0][0][1]}")
            print(f"Input prompt: {INPUT_PROMPT[max_ind]}")
            print("=" * 80)
            if is_images:
                generate_image(
                    pipe=pipe,
                    pp=INPUT_PROMPT[max_ind],
                    pe=input_embed[max_ind],
                    ce=cache_embedding,
                    ch=sorted_data[0][0][3],
                    num=num,
                    src_path=BASE_PATH,
                    dst_path=PROJECT_PATH
                )
        print(f"Average BLEU-1 score: {np.mean(bleu_scores)}")
        print(f"Average similarity score: {np.mean(similarity_scores)}")
        print(f"Average word precision: {np.mean(word_precision_scores)}")
        print(f"Average word recall: {np.mean(word_recall_scores)}")
        print(f"Average word F1 score: {np.mean(word_f1_scores)}")
        print(f"Average Jaccard score: {np.mean(jaccard_scores)}")
        
        P, R, F1 = score(hyp_prompts, ref_prompts, lang="en", model_type="roberta-large", verbose=True, device=device)
        print("Bert Score Precision:", P.mean().item())
        print("Bert Score Recall:   ", R.mean().item())
        print("Bert Score F1:       ", F1.mean().item())
        
        print("Average Lib BLEU score:", np.mean(lib_bleu_scores))


        MODEL_NAME = "Jean-Baptiste/roberta-large-ner-english"  # BERT model fine-tuned for NER

        nerr_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=nerr_tokenizer,
            aggregation_strategy="simple"  # groups sub-tokens into full entities
        )

        # -------------------------------
        # 2. Helper: extract entities
        # -------------------------------

        def extract_entities(text):
            """
            Run NER on the input text and return a set of normalized entities.
            Each entity is represented as (entity_type, entity_text_lowercased).
            """
            raw_ents = ner_pipeline(text)

            entities = set()
            for ent in raw_ents:
                ent_type = ent["entity_group"]  # e.g., PER, ORG, LOC, MISC
                ent_text = ent["word"].strip().lower()
                entities.add((ent_type, ent_text))
            return entities

        # -------------------------------
        # 3. Compute NERR (NER Recall)
        # -------------------------------

        def compute_nerr(hyp, ref):
            """
            Compute NERR (Named Entity Recognition Recall) of hyp w.r.t. ref.

            NERR = (# of named entities in ref that are also in hyp) / (# entities in ref)

            Additionally, compute:
            - precision = overlap / (# entities in hyp, if > 0)
            - f1 = harmonic mean of precision and recall
            """
            ref_ents = extract_entities(ref)
            hyp_ents = extract_entities(hyp)

            if not ref_ents and not hyp_ents:
                # No entities in either string -> consider it a perfect match
                return 1.0, 1.0, 1.0, ref_ents, hyp_ents

            overlap = ref_ents & hyp_ents
            overlap_count = len(overlap)

            ref_count = len(ref_ents)
            hyp_count = len(hyp_ents)

            # Recall: how many ref entities appear in hyp
            recall = overlap_count / ref_count if ref_count > 0 else 1.0

            # Precision: how many hyp entities are in ref
            precision = overlap_count / hyp_count if hyp_count > 0 else 1.0

            # F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            return recall, precision, f1, ref_ents, hyp_ents
        
        all_recalls = []
        all_precisions = []
        all_f1 = []

        for hyp, ref in zip(hyp_prompts, ref_prompts):
            recall, precision, f1, ref_ents, hyp_ents = compute_nerr(hyp, ref)
            all_recalls.append(recall)
            all_precisions.append(precision)
            all_f1.append(f1)

        print("Mean NERR (recall):", sum(all_recalls) / len(all_recalls))
        print("Mean NERR precision:     ", sum(all_precisions) / len(all_precisions))
        print("Mean NERR F1:            ", sum(all_f1) / len(all_f1))
        
        df = pd.DataFrame(bleu_scores, columns=['bleu1'])
        # df['similarity'] = similarity_scores
        df['word_precision'] = word_precision_scores
        df['word_recall'] = word_recall_scores
        df['word_f1'] = word_f1_scores
        df['jaccard'] = jaccard_scores
        
        df.to_csv(f'bleu1-scores-baseline.csv', index=False)
        exit()
                
        if predict:
            print("~" * 80)
            print(f"preict results:")
            print_stats(predict_result)
            print(f"total rounds: {len(predict_result)}")
            print("=" * 80)
            if is_all:
                df = pd.DataFrame(predict_result, columns=["similarity"])
                df.to_csv(f"result-predict-{directory}.csv", index=False)
            exit()
        
        print_stats(result)
        print(f"total rounds: {len(result)}")
        print(f"Average cache hit: {np.mean(avg_cache_hit)}")
        print(f"Average prompt hit: {np.mean(avg_prompt_hit)}")
        print(f"Average rate: {np.mean(avg_rate)}")
        print(f"Average over-ratio: {overrations / len(result)}")
        print(f"Average false positive: {fps / len(result)}")
        
        if is_stats:
            edges, results = histogram_with_mean_score(avg_prompt_hit, 6)

            print("Histogram with Mean Scores:")
            for i, result in enumerate(results):
                print(f"Bucket {i+1}: {result['bucket_range']}")
                print(f"  Count: {result['count']}")
                print(f"  Mean Score: {result['mean_score']:.2f}")
                print(f"  Items: {result['items']}")
                print()
        
        if is_csv:
            df = pd.DataFrame(result, columns=["semantic"])
            df['num_of_modifiers'] = num_of_modifiers
            df.to_csv(f"result-{directory}.csv", index=False) 
elif option == 1:
    # remedial
    import re
        
    with torch.no_grad():
        PROJECT_PATH = f"{DATA_HOME}/diffusion-cache-security/{directory}"
        BASE_PATH = PROJECT_PATH.replace("-remedial", "")
        # BASE_PATH = PROJECT_PATH
        
        model, _ = clip.load("ViT-L/14", device=device)
        result = []
        avg_cache_hit = []
        avg_prompt_hit = []
        avg_rate = []
        num_of_modifiers = []
        predict_result = []
        overrations = 0
        fps = 0
        bleu_scores = []
        similarity_scores = []

        word_precision_scores = []
        word_recall_scores = []
        word_f1_scores = []
        subset_scores = []
        jaccard_scores = []
        hyp_prompts = []
        ref_prompts = []
        lib_bleu_scores = []
        NERR_scores = []
        # completed = pickle.load(open("sequence.pkl", "rb"))
        # seq = [i for i in range(1,34)]
        # seq.extend([i for i in range(51, 86)])
        for i in range(start, end):
            num = i
            if not os.path.exists(f"{BASE_PATH}/reconstructed_sorted_{i}.pkl"):
                print(f"Skipping {i} as {BASE_PATH}/reconstructed_sorted_{i}.pkl does not exist.")
                print("=" * 80)
                continue
            if not os.path.exists(f"{PROJECT_PATH}/exploited_prompts_{num}.pkl"):
                print(f"Skipping {i} as {PROJECT_PATH}/exploited_prompts_{num}.pkl file does not exist.")
                print("=" * 80)
                continue
            
            sorted_data = pickle.load(open(f"{BASE_PATH}/reconstructed_sorted_{num}.pkl", "rb"))
            
            ref = sorted_data[0][0][1]
            # if "testing" in ref:
            if "meme" in ref or "testing" in ref:
                print(f"Skipping {i} due to invalid reference prompt.")
                continue

            if predict:
                before = pickle.load(open(f"{PROJECT_PATH}/batch_x_final_{num}.pkl", "rb"))
                cache_embedding = sorted_data[0][0][2]
                cos_sim_mat = torch.nn.functional.cosine_similarity(before, torch.tensor(cache_embedding).to(device), dim=-1)
                sim = torch.max(cos_sim_mat).item()
                predict_result.append(sim)
                print(f"Round {i} before similarity: {sim}")
                print("=" * 80)
                continue
            
            len_of_max_classifier_result = len(pickle.load(open(f"{PROJECT_PATH}/extracted_data_{num}.pkl", "rb"))[0][1])
            len_of_result = sorted_data[0][1][0]
            avg_cache_hit.append(len_of_result)
            ratio = (len_of_max_classifier_result - len_of_result) / len_of_result
            if len_of_max_classifier_result > len_of_result:
                overrations += ratio
                fps += len_of_max_classifier_result - len_of_result
            
            avg_rate.append(ratio)
            print(f"Round {num} has a significant difference: {len_of_max_classifier_result} vs {len_of_result}, over-ratio: {ratio}")
            
            INPUT_PROMPT = pickle.load(open(f"{PROJECT_PATH}/exploited_prompts_{i}.pkl", "rb"))
                
            input_embed = model.encode_text(clip.tokenize(INPUT_PROMPT, truncate=True).to(device))
            input_embed = torch.nn.functional.normalize(input_embed, dim=-1, p=2)
            
            cache_embedding = torch.tensor(sorted_data[0][0][2]).to(device)
            
            similarities = torch.nn.functional.cosine_similarity(cache_embedding, input_embed, dim=-1)
            max_ind = torch.argmax(similarities)
            similarity = similarities[max_ind]
            max_prompt = INPUT_PROMPT[max_ind]
            ref_prompts.append(sorted_data[0][0][1])
            hyp_prompts.append(max_prompt)

            # print(INPUT_PROMPT)
            # print(len(INPUT_PROMPT))
            print(f"=" * 50)
            # print(f"hyp: {max_prompt}")
            # print(f"ref: {sorted_data[0][0][1]}")
            # input("debug")
            ref = re.sub(r'[^A-Za-z0-9]', ' ', sorted_data[0][0][1].lower())
            hyp = re.sub(r'[^A-Za-z0-9]', ' ', max_prompt.lower())
            # ref = tokenizer.tokenize(sorted_data[0][0][1].lower())
            # ref = [x for x in ref if x != ","]
            # hyp = tokenizer.tokenize(max_prompt.lower())
            # hyp = [x for x in hyp if x != ","]
            print(f"hyp tokens: {hyp}")
            print(f"ref tokens: {ref}")
            # is_sentence = False
            is_sentence = True

            bleu_score = bleu1(ref, hyp, is_sentence)
            jaccard_score = jaccard_similarity(ref, hyp, is_sentence)
            
            bleu_scores.append(bleu_score)
            similarity_scores.append(similarity.item())

            precision, recall, f1 = word_prf1(ref, hyp, is_sentence)

            word_precision_scores.append(precision)
            word_recall_scores.append(recall)
            word_f1_scores.append(f1)
            jaccard_scores.append(jaccard_score)
            
            # bleu = sacrebleu.corpus_bleu([hyp], [[ref]], lowercase=True)
            # print("Lib BLEU:", bleu.score)
            # lib_bleu_scores.append(bleu.score)
            
            weights_2 = (1, 0, 0.0, 0.0)
            chencherry = SmoothingFunction() 
            sentence_score = sentence_bleu([ref.split()], hyp.split(), 
                             weights=weights_2, 
                             smoothing_function=chencherry.method1)
            print("Lib BlEU (nltk sentence):", sentence_score)
            lib_bleu_scores.append(sentence_score)
            
            # NERR_score = nerr(ref, hyp)
            # NERR_score = max(0.0, NERR_score)  # Clamp to [0, 1]
            # NERR_scores.append(NERR_score)
            print(f"Round {i} BLEU-1 score: {bleu_score}")
            print(f"Round {i} similarity: {similarity.item()}")
            print(f"Round {i} word precision: {precision}")
            print(f"Round {i} word recall: {recall}")
            print(f"Round {i} word F1: {f1}")
            print(f"Round {i} Jaccard score: {jaccard_score}")
            # print(f"Round {i} NERR score: {NERR_score}")
            
            print()
            continue
            avg_prompt_hit.append((len_of_max_classifier_result, similarity.item()))
            result.append(similarity.item())
            
            hit_prompts = sorted_data[0][1][1]
            sum_of_modifiers = 0
            for (p, e, h) in hit_prompts:
                sum_of_modifiers += len([w for w in p.split(",") if w])
            num_of_modifiers.append(sum_of_modifiers // len(hit_prompts))
            
            print(f"Round {i} similarity: {similarity.item()}")
            print(f"Cache: {sorted_data[0][0][1]}")
            print(f"Input prompt: {INPUT_PROMPT[max_ind]}")
            print("=" * 80)
            if is_images:
                generate_image(
                    pipe=pipe,
                    pp=INPUT_PROMPT[max_ind],
                    pe=input_embed[max_ind],
                    ce=cache_embedding,
                    ch=sorted_data[0][0][3],
                    num=num,
                    src_path=BASE_PATH,
                    dst_path=PROJECT_PATH
                )
        print(f"Average BLEU-1 score: {np.mean(bleu_scores)}")
        print(f"Average similarity score: {np.mean(similarity_scores)}")
        print(f"Average word precision: {np.mean(word_precision_scores)}")
        print(f"Average word recall: {np.mean(word_recall_scores)}")
        print(f"Average word F1 score: {np.mean(word_f1_scores)}")
        print(f"Average Jaccard score: {np.mean(jaccard_scores)}")

        P, R, F1 = score(hyp_prompts, ref_prompts, lang="en", model_type="roberta-large", verbose=True, device=device)
        print("Bert Score Precision:", P.mean().item())
        print("Bert Score Recall:   ", R.mean().item())
        print("Bert Score F1:       ", F1.mean().item())
        
        print("Average Lib BLEU score:", np.mean(lib_bleu_scores))
        # print("Average NERR score:", np.mean(NERR_scores))

        # MODEL_NAME = "dslim/bert-base-NER"  # BERT model fine-tuned for NER

        # nerr_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

        # ner_pipeline = pipeline(
        #     "ner",
        #     model=model,
        #     tokenizer=nerr_tokenizer,
        #     aggregation_strategy="simple"  # groups sub-tokens into full entities
        # )

        # # -------------------------------
        # # 2. Helper: extract entities
        # # -------------------------------

        # def extract_entities(text):
        #     """
        #     Run NER on the input text and return a set of normalized entities.
        #     Each entity is represented as (entity_type, entity_text_lowercased).
        #     """
        #     raw_ents = ner_pipeline(text)

        #     entities = set()
        #     for ent in raw_ents:
        #         ent_type = ent["entity_group"]  # e.g., PER, ORG, LOC, MISC
        #         ent_text = ent["word"].strip().lower()
        #         entities.add((ent_type, ent_text))
        #     return entities

        # # -------------------------------
        # # 3. Compute NERR (NER Recall)
        # # -------------------------------

        # def compute_nerr(hyp, ref):
        #     """
        #     Compute NERR (Named Entity Recognition Recall) of hyp w.r.t. ref.

        #     NERR = (# of named entities in ref that are also in hyp) / (# entities in ref)

        #     Additionally, compute:
        #     - precision = overlap / (# entities in hyp, if > 0)
        #     - f1 = harmonic mean of precision and recall
        #     """
        #     ref_ents = extract_entities(ref)
        #     hyp_ents = extract_entities(hyp)

        #     if not ref_ents and not hyp_ents:
        #         # No entities in either string -> consider it a perfect match
        #         return 1.0, 1.0, 1.0, ref_ents, hyp_ents

        #     overlap = ref_ents & hyp_ents
        #     overlap_count = len(overlap)

        #     ref_count = len(ref_ents)
        #     hyp_count = len(hyp_ents)

        #     # Recall: how many ref entities appear in hyp
        #     recall = overlap_count / ref_count if ref_count > 0 else 1.0

        #     # Precision: how many hyp entities are in ref
        #     precision = overlap_count / hyp_count if hyp_count > 0 else 1.0

        #     # F1 score
        #     if precision + recall > 0:
        #         f1 = 2 * precision * recall / (precision + recall)
        #     else:
        #         f1 = 0.0

        #     return recall, precision, f1, ref_ents, hyp_ents
        
        # all_recalls = []
        # all_precisions = []
        # all_f1 = []

        # for hyp, ref in zip(hyp_prompts, ref_prompts):
        #     recall, precision, f1, ref_ents, hyp_ents = compute_nerr(hyp, ref)
        #     all_recalls.append(recall)
        #     all_precisions.append(precision)
        #     all_f1.append(f1)

        # print("Mean NERR (recall):", sum(all_recalls) / len(all_recalls))
        # print("Mean NERR precision:     ", sum(all_precisions) / len(all_precisions))
        # print("Mean NERR F1:            ", sum(all_f1) / len(all_f1))
        
        
        MODEL_NAME = "Jean-Baptiste/roberta-large-ner-english"  # BERT model fine-tuned for NER

        nerr_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=nerr_tokenizer,
            aggregation_strategy="simple"  # groups sub-tokens into full entities
        )

        # -------------------------------
        # 2. Helper: extract entities
        # -------------------------------

        def extract_entities(text):
            """
            Run NER on the input text and return a set of normalized entities.
            Each entity is represented as (entity_type, entity_text_lowercased).
            """
            raw_ents = ner_pipeline(text)

            entities = set()
            for ent in raw_ents:
                ent_type = ent["entity_group"]  # e.g., PER, ORG, LOC, MISC
                ent_text = ent["word"].strip().lower()
                entities.add((ent_type, ent_text))
            return entities

        # -------------------------------
        # 3. Compute NERR (NER Recall)
        # -------------------------------

        def compute_nerr(hyp, ref):
            """
            Compute NERR (Named Entity Recognition Recall) of hyp w.r.t. ref.

            NERR = (# of named entities in ref that are also in hyp) / (# entities in ref)

            Additionally, compute:
            - precision = overlap / (# entities in hyp, if > 0)
            - f1 = harmonic mean of precision and recall
            """
            ref_ents = extract_entities(ref)
            hyp_ents = extract_entities(hyp)

            if not ref_ents or not hyp_ents:
                # No entities in either string -> consider it a perfect match
                return -1, -1, -1, ref_ents, hyp_ents

            overlap = ref_ents & hyp_ents
            overlap_count = len(overlap)

            ref_count = len(ref_ents)
            hyp_count = len(hyp_ents)

            # Recall: how many ref entities appear in hyp
            recall = overlap_count / ref_count #if ref_count > 0 else 1.0

            # Precision: how many hyp entities are in ref
            precision = overlap_count / hyp_count #if hyp_count > 0 else 1.0

            # F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            return recall, precision, f1, ref_ents, hyp_ents
        
        all_recalls = []
        all_precisions = []
        all_f1 = []

        for hyp, ref in zip(hyp_prompts, ref_prompts):
            recall, precision, f1, ref_ents, hyp_ents = compute_nerr(hyp, ref)
            if recall == -1:
                continue
            all_recalls.append(recall)
            all_precisions.append(precision)
            all_f1.append(f1)

        # print(all_recalls)
        print("Mean NERR (recall):", sum(all_recalls) / len(all_recalls))
        print("Mean NERR precision:     ", sum(all_precisions) / len(all_precisions))
        print("Mean NERR F1:            ", sum(all_f1) / len(all_f1))
        
        df = pd.DataFrame(bleu_scores, columns=['bleu1'])
        # df['similarity'] = similarity_scores
        df['word_precision'] = word_precision_scores
        df['word_recall'] = word_recall_scores
        df['word_f1'] = word_f1_scores
        df['jaccard'] = jaccard_scores
        df['Bert_Precision'] = P.cpu().numpy()
        df['Bert_Recall'] = R.cpu().numpy()
        df['Bert_F1'] = F1.cpu().numpy()
        df["NERR_Recall"] = all_recalls
        df["NERR_Precision"] = all_precisions
        df["NERR_F1"] = all_f1
        
        df.to_csv(f'bleu1-scores-{directory}.csv', index=False)
        exit()
                
        if predict:
            print("~" * 80)
            print(f"preict results:")
            print_stats(predict_result)
            print(f"total rounds: {len(predict_result)}")
            print("=" * 80)
            if is_all:
                df = pd.DataFrame(predict_result, columns=["similarity"])
                df.to_csv(f"result-predict-{directory}.csv", index=False)
            exit()
        
        print_stats(result)
        print(f"total rounds: {len(result)}")
        print(f"Average cache hit: {np.mean(avg_cache_hit)}")
        print(f"Average prompt hit: {np.mean(avg_prompt_hit)}")
        print(f"Average rate: {np.mean(avg_rate)}")
        print(f"Average over-ratio: {overrations / len(result)}")
        print(f"Average false positive: {fps / len(result)}")
        
        if is_stats:
            edges, results = histogram_with_mean_score(avg_prompt_hit, 6)

            print("Histogram with Mean Scores:")
            for i, result in enumerate(results):
                print(f"Bucket {i+1}: {result['bucket_range']}")
                print(f"  Count: {result['count']}")
                print(f"  Mean Score: {result['mean_score']:.2f}")
                print(f"  Items: {result['items']}")
                print()
        
        if is_csv:
            df = pd.DataFrame(result, columns=["semantic"])
            df['num_of_modifiers'] = num_of_modifiers
            df.to_csv(f"result-{directory}.csv", index=False) 
elif option == 2:
    # recover all / precise / fp
    
    with torch.no_grad():        
        model, _ = clip.load("ViT-L/14", device=device)
        result = []
        predict_result = []
        
        import re
        
        BASE_PATH = PROJECT_PATH.replace("-recover-all", "-7_17").replace("-recover-precise", "-7_17")
        BASE_PATH = re.sub(r"-fp-\d+", "-7_17", BASE_PATH)
               
        # completed = pickle.load(open("sequence.pkl", "rb"))
        for i in range(start, end):
            num = i

            if not os.path.exists(f"{PROJECT_PATH}/batch_x_final_{num}.pkl"):
                print(f"Skipping {i} as {PROJECT_PATH}/batch_x_final_{num}.pkl file does not exist.")
                print("=" * 80)
                continue
            
            if not os.path.exists(f"{PROJECT_PATH}/exploited_prompts_{num}.pkl"):
                print(f"Skipping {i} as {PROJECT_PATH}/exploited_prompts_{num}.pkl file does not exist.")
                print("=" * 80)
                continue
            
            
            sorted_data = pickle.load(open(f"{BASE_PATH}/reconstructed_sorted_{num}.pkl", "rb"))
            
            if predict:
                before = pickle.load(open(f"{PROJECT_PATH}/batch_x_final_{num}.pkl", "rb"))
                cache_embedding = sorted_data[0][0][2]
                cos_sim_mat = torch.nn.functional.cosine_similarity(before, torch.tensor(cache_embedding).to(device), dim=-1)
                sim = torch.max(cos_sim_mat).item()
                predict_result.append(sim)
                print(f"Round {i} before similarity: {sim}")
                print("=" * 80)
                continue
                
            
            INPUT_PROMPT = pickle.load(open(f"{PROJECT_PATH}/exploited_prompts_{i}.pkl", "rb"))
                
            input_embed = model.encode_text(clip.tokenize(INPUT_PROMPT, truncate=True).to(device))
            input_embed = torch.nn.functional.normalize(input_embed, dim=-1, p=2)
            
            cache_embedding = torch.tensor(sorted_data[0][0][2]).to(device)
            
            similarities = torch.nn.functional.cosine_similarity(cache_embedding, input_embed, dim=-1)
            max_ind = torch.argmax(similarities)
            similarity = similarities[max_ind]
            result.append(similarity.item())
            print(f"Round {i} similarity: {similarity.item()}")
            print(f"Cache: {sorted_data[0][0][1]}")
            print(f"Input prompt: {INPUT_PROMPT[max_ind]}")
            print("=" * 80)
            
            if is_images:
                generate_image(
                    pipe=pipe,
                    pp=INPUT_PROMPT[max_ind],
                    pe=input_embed[max_ind],
                    ce=cache_embedding,
                    ch=sorted_data[0][0][3],
                    num=num,
                    src_path=BASE_PATH,
                    dst_path=PROJECT_PATH
                )
        
        if predict:
            print_stats(predict_result)
            print(f"total rounds: {len(predict_result)}")
            print("=" * 80)
            if is_all:
                df = pd.DataFrame(predict_result, columns=["similarity"])
                df.to_csv(f"result-predict-{directory}.csv", index=False)
            exit()
        
        print_stats(result)
        print(f"total rounds: {len(result)}")
        print("=" * 80)
        
        if is_all:
            df = pd.DataFrame(result, columns=["semantic"])
            df.to_csv(f"result-{directory}.csv", index=False)
elif option == 3:
    directory = "experiments-naive-probe-diffdb-7_17"
    BASE_PATTERN = f'{DATA_HOME}/diffusion_sec/diffusion/{directory}/run_*/result-check.txt'
    scores = []
    not_hit = []
    
    hyp_prompts = []
    ref_prompts = []
    lib_bleu_scores = []
    NERR_scores = []
    
    for i, path in enumerate(sorted(glob.glob(BASE_PATTERN))):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()

                # Extract ref (after "prompts:")
                if "prompts:" in line:
                    # Split on "prompts:" and take everything after
                    ref = line.split("prompts:", 1)[1].strip()

                # Extract hyp (after "From Query:")
                if line.startswith("From Query:"):
                    hyp = line.split("From Query:", 1)[1].strip()
                    
            ref = re.sub(r'[^A-Za-z0-9]', ' ', ref.lower())
            hyp = re.sub(r'[^A-Za-z0-9]', ' ', hyp.lower())
            hyp_prompts.append(hyp)
            ref_prompts.append(ref)
            weights_2 = (1, 0, 0.0, 0.0)
            chencherry = SmoothingFunction() 
            sentence_score = sentence_bleu([ref.split()], hyp.split(), 
                            weights=weights_2, 
                            smoothing_function=chencherry.method1)
            print("Lib BlEU (nltk sentence):", sentence_score)
            lib_bleu_scores.append(sentence_score)
    
            # nerr_score = nerr(ref, hyp)
            # NERR_scores.append(nerr_score)
            
    P, R, F1 = score(hyp_prompts, ref_prompts, lang="en", model_type="roberta-large", verbose=True, device=device)
    print("Bert Score Precision:", P.mean().item())
    print("Bert Score Recall:   ", R.mean().item())
    print("Bert Score F1:       ", F1.mean().item())
    
    print("Average Lib BLEU score:", np.mean(lib_bleu_scores))
    # print("Average NERR score:", np.mean(NERR_scores))


    MODEL_NAME = "Jean-Baptiste/roberta-large-ner-english"  # BERT model fine-tuned for NER

    nerr_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=nerr_tokenizer,
        aggregation_strategy="simple"  # groups sub-tokens into full entities
    )

    # -------------------------------
    # 2. Helper: extract entities
    # -------------------------------

    def extract_entities(text):
        """
        Run NER on the input text and return a set of normalized entities.
        Each entity is represented as (entity_type, entity_text_lowercased).
        """
        raw_ents = ner_pipeline(text)

        entities = set()
        for ent in raw_ents:
            ent_type = ent["entity_group"]  # e.g., PER, ORG, LOC, MISC
            ent_text = ent["word"].strip().lower()
            entities.add((ent_type, ent_text))
        return entities

    # -------------------------------
    # 3. Compute NERR (NER Recall)
    # -------------------------------

    def compute_nerr(hyp, ref):
        """
        Compute NERR (Named Entity Recognition Recall) of hyp w.r.t. ref.

        NERR = (# of named entities in ref that are also in hyp) / (# entities in ref)

        Additionally, compute:
        - precision = overlap / (# entities in hyp, if > 0)
        - f1 = harmonic mean of precision and recall
        """
        ref_ents = extract_entities(ref)
        hyp_ents = extract_entities(hyp)

        if not ref_ents or not hyp_ents:
            # No entities in either string -> consider it a perfect match
            return -1, -1, -1, ref_ents, hyp_ents

        overlap = ref_ents & hyp_ents
        overlap_count = len(overlap)

        ref_count = len(ref_ents)
        hyp_count = len(hyp_ents)

        # Recall: how many ref entities appear in hyp
        recall = overlap_count / ref_count #if ref_count > 0 else 1.0

        # Precision: how many hyp entities are in ref
        precision = overlap_count / hyp_count #if hyp_count > 0 else 1.0

        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return recall, precision, f1, ref_ents, hyp_ents
    
    all_recalls = []
    all_precisions = []
    all_f1 = []

    for hyp, ref in zip(hyp_prompts, ref_prompts):
        recall, precision, f1, ref_ents, hyp_ents = compute_nerr(hyp, ref)
        if recall == -1:
            continue
        all_recalls.append(recall)
        all_precisions.append(precision)
        all_f1.append(f1)

    # print(all_recalls)
    print("Mean NERR (recall):", sum(all_recalls) / len(all_recalls))
    print("Mean NERR precision:     ", sum(all_precisions) / len(all_precisions))
    print("Mean NERR F1:            ", sum(all_f1) / len(all_f1))            
    
                
elif option == 4:
    # embedding predictor cosine similarity check
    
    # BASE_DIR = '/u501/sjie/diff-sec/experiments-lexica-7_17'
    # ALL_RUNS = sorted(glob.glob(os.path.join(BASE_DIR, 'run_*')))
    # ALL_RUN_IDS = [int(os.path.basename(path).split('_')[1]) for path in ALL_RUNS]

    # HAVE_RESULT = sorted(glob.glob(os.path.join(BASE_DIR, 'run_*/result-check.txt')))
    # HAVE_RUN_IDS = [int(path.split('/')[-2].split('_')[1]) for path in HAVE_RESULT]

    # missing = sorted(set(ALL_RUN_IDS) - set(HAVE_RUN_IDS))
    # print(f"Missing result-check.txt in runs: {missing}")

    DEVICE = "cuda:0"
    path = f"{DATA_HOME}/diffusion-cache-security/{directory}/"

    def return_vector_path(num):
        return path + f"batch_x_final_{num}.pkl"

    def return_target_path(num):
        return path + f"reconstructed_sorted_{num}.pkl"

    result = []
    for i in range(100):
        print(f"Processing batch {i + 1}...")
        if not os.path.exists(return_vector_path(i + 1)):
            print(f"Skipping {i + 1} as vector path does not exist.")
            continue
        vector_path = return_vector_path(i + 1)
        target_path = return_target_path(i + 1)
        
        vectors = pickle.load(open(vector_path, "rb"))
        sorted_data = pickle.load(open(target_path, "rb"))
        
        cache_embedding = torch.tensor(sorted_data[0][0][2]).to(DEVICE)
        
        cos_sim_mat = torch.nn.functional.cosine_similarity(cache_embedding, vectors, dim=-1)
        
        result.append(torch.max(cos_sim_mat).item())

    print(path)
    print_stats(result)
    df = pd.read_csv(f"{DATA_HOME}/{directory}/result-{directory}.csv")
    df['cosine_similarity'] = result
    df.to_csv(f'{DATA_HOME}/{directory}/result-{directory}.csv', index=False)
    shutil.copy(f'{DATA_HOME}/{directory}/result-{directory}.csv',
                f'{DATA_HOME}/result-{directory}.csv') 
elif option == 5:
    # recover precise
    BASE_PATH = f"{DATA_HOME}/diffusion-cache-security/{directory}"
    
    model, _ = clip.load("ViT-L/14", device=device)
    result = []
    avg_hit = []
    # completed = pickle.load(open("sequence.pkl", "rb"))
    for i in range(1, 101):
        num = i
        if not os.path.exists(f"{BASE_PATH}/batch_x_final_{i}.pkl"):
            print(f"Skipping {i} as batch_x_final_{i} does not exist.")
            print("=" * 80)
            continue
        
        sorted_data = pickle.load(open(f"{BASE_PATH}/sorted_data_{num}.pkl", "rb"))
        avg_hit.append(sorted_data[0][1][0])
        INPUT_PROMPT = pickle.load(open(f"{PROJECT_PATH}/exploited_prompts_{i}.pkl", "rb"))
            
        input_embed = model.encode_text(clip.tokenize(INPUT_PROMPT, truncate=True).to(device))
        input_embed = torch.nn.functional.normalize(input_embed, dim=-1, p=2)
        
        cache_embedding = torch.tensor(sorted_data[0][0][2]).to(device)
        
        similarities = torch.nn.functional.cosine_similarity(cache_embedding, input_embed, dim=-1)
        max_ind = torch.argmax(similarities)
        similarity = similarities[max_ind]
        result.append(similarity.item())
        print(f"Round {i} similarity: {similarity.item()}")
        print(f"Cache: {sorted_data[0][0][1]}")
        print(f"Input prompt: {INPUT_PROMPT[max_ind]}")
        print("=" * 80)
    
    print_stats(result)
    print(f"total rounds: {len(result)}")
    print(f"Average hit: {np.mean(avg_hit)}")

if image_metrics:
    import torch
    import clip
    from PIL import Image
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    # CLIP similarity
    model, preprocess = clip.load("ViT-L/14", device=device)
    
    name = directory
    BASE_PATH = f'{DATA_HOME}/diffusion-cache-security/{directory}/experiments/'
    
    # BASE_PATTERN = f'{path}/*'
    scores = []
    not_hit = []
    
    clips_scores = []
    psnr_scores = []
    ssim_scores = []
    
    max_score = (-1.0, "")

    # Loop through all round* folders
    for round_dir in sorted(glob.glob(os.path.join(BASE_PATH, "round*"))):
        final_images_path = os.path.join(round_dir, "final-images")

        # Skip if final-images doesn't exist
        if not os.path.isdir(final_images_path):
            print(f"Skipping {final_images_path} as final-images folder does not exist.")
            continue

        print(f"Reading images from: {final_images_path}")        
        files = glob.glob(f"{final_images_path}/*.png")
        
        img1 = preprocess(Image.open(files[0])).unsqueeze(0).to(device)
        img2 = preprocess(Image.open(files[1])).unsqueeze(0).to(device)

        with torch.no_grad():
            emb1 = model.encode_image(img1)
            emb2 = model.encode_image(img2)

        emb1 /= emb1.norm(dim=-1, keepdim=True)
        emb2 /= emb2.norm(dim=-1, keepdim=True)
        clip_similarity = (emb1 * emb2).sum(dim=-1).item()
        # if clip_similarity > max_score[0]:
        #     max_score = (clip_similarity, folder)

        # PSNR and SSIM
        im1_np = np.array(Image.open(files[0]).convert("RGB"))
        im2_np = np.array(Image.open(files[1]).convert("RGB"))
        
        if im1_np.shape != im2_np.shape:
            raise ValueError(f"Images must match. Got {im1_np.shape} vs {im2_np.shape}")

        psnr_value = psnr(im1_np, im2_np, data_range=255)
        ssim_value, _ = ssim(im1_np, im2_np, multichannel=True, channel_axis=-1, full=True)
        
        clips_scores.append(clip_similarity)
        psnr_scores.append(psnr_value)
        ssim_scores.append(ssim_value)
    
    print("CLIP Similarity Stats:")
    print_stats(clips_scores)
    print("\nPSNR Stats:")
    print_stats(psnr_scores)
    print("\nSSIM Stats:")
    print_stats(ssim_scores)
    # print(f"Max CLIP similarity: {max_score[0]} in folder {max_score[1]}")
    
    df = pd.DataFrame(clips_scores, columns=['clip_similarity'])
    df['psnr'] = psnr_scores
    df['ssim'] = ssim_scores
    df.to_csv(f'result-images-{directory}.csv', index=False)
    # shutil.copy(f'{DATA_HOME}/{name}/result-{directory}.csv',
    #             f'{DATA_HOME}/result-{directory}.csv')