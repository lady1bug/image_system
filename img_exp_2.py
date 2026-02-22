import os
import timeit
import numpy as np
import pandas as pd
import pickle as pkl
import json
from tqdm import tqdm
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score

from core.metrics import compute_eer, compute_min_dcf, compute_far_frr
from core.cosine_similarity import cosine_similarity
from face_exp_2_avg_embeds import get_embedding, load_grouped_dataset 

# --- Налаштування ---
DATASET_FOLDER = "dataset_img"
RESULTS_CSV = "face_scores_exp_2.csv"
JSON_DIR = "avg_embedings"
MODELS = ["Facenet", "ArcFace", "VGG-Face", "SFace", "OpenFace"]

def evaluate_pipeline(model_name: str, data: List[List[str]]) -> Dict[str, Any]:
    scores, labels = [], []
    distance_self, distance_other = [], []
    elapsed_time = []

    # 1. Завантаження усереднених профілів
    try:
        with open(f'./{JSON_DIR}/{model_name}.json', 'r') as f:
            profiles_json = json.load(f)
        profiles_np = {k: np.array(v) for k, v in profiles_json.items()}
    except Exception as e:
        print(f"Error loading profiles for {model_name}: {e}")
        return {}
    
    if not profiles_np:
        return {}

    # 2. Завантаження порогу (Threshold) з першого експерименту
    try:
        df_exp1 = pd.read_csv("face_scores_exp_1.csv")
        thresh = float(df_exp1[df_exp1['pipeline'] == model_name]["threshold"].iloc[0])
    except Exception:
        print(f"Warning: Using default threshold 0.0 for {model_name}")
        thresh = 0.0
        
    # 3. Тестування (1:N порівняння)
    for group in tqdm(data, desc=f"Evaluating {model_name}"):
        person_id = group[0].split(os.sep)[-2] 

        for file in group:
            test_emb = get_embedding(file, model_name)
            if test_emb is None:
                continue
            
            for profile_id, avg_emb in profiles_np.items():
                start_time = timeit.default_timer()
                
                similarity = cosine_similarity(test_emb, avg_emb)
                distance = 1 - similarity
                
                elapsed_time.append(timeit.default_timer() - start_time)
                
                label = int(profile_id == person_id) 
                scores.append(similarity)
                labels.append(label)
                
                if label == 1:
                    distance_self.append(distance)
                else:
                    distance_other.append(distance)

    if not scores:
        return {}
        
    # 4. Обчислення біометричних показників
    fa_score, fr_score = compute_far_frr(scores, labels, thresh)
    min_dcf = compute_min_dcf(fr_score, fa_score)
    ee_rate = (fr_score + fa_score) / 2 
    
    predictions = [1 if s >= thresh else 0 for s in scores]

    return {
        "pipeline": model_name,
        "fa_score": fa_score,
        "fr_score": fr_score,
        "ee_rate": ee_rate,
        "dcf": min_dcf, 
        "threshold": thresh,
        "accuracy": accuracy_score(labels, predictions),
        "distance_self_mean": np.mean(distance_self),
        "distance_self_std": np.std(distance_self),
        "distance_other_mean": np.mean(distance_other) if distance_other else 0.0,
        "distance_other_std": np.std(distance_other) if distance_other else 0.0,
        "elapsed_time_mean": np.mean(elapsed_time),
        "elapsed_time_std": np.std(elapsed_time),
    }

def main():
    # 1. Завантаження датасету
    if os.path.exists("dataset_exp_2.pkl"):
        with open("dataset_exp_2.pkl", "rb") as f:
            dataset = pkl.load(f)
    else:
        dataset = load_grouped_dataset(DATASET_FOLDER) 
        with open("dataset_exp_2.pkl", "wb") as f:
            pkl.dump(dataset, f)
    
    print(f"Groups processed: {len(dataset)}")
    
    # 2. Експеримент для кожної моделі
    results = {}
    for model_name in MODELS:
        print(f"\nProcessing model: {model_name}")
        results[model_name] = evaluate_pipeline(model_name, dataset)
    
    # 3. Збереження фінального звіту
    pd.DataFrame(results).transpose().to_csv(RESULTS_CSV) 
    print(f"\nResults exported to {RESULTS_CSV}")

if __name__ == "__main__":
    if os.path.exists(DATASET_FOLDER):
        main()
    else:
        print(f"Dataset folder '{DATASET_FOLDER}' not found.")