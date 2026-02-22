import os
from tqdm import tqdm
import timeit
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import accuracy_score
from typing import List, Tuple, Dict, Any

from core.face_pipeline import FacePipeline 
from core.metrics import (
    compute_eer, 
    compute_min_dcf,
    compute_far_frr,
)
import utils.face_dataset as data_utils 

# --- Налаштування ---
ROOT = "dataset_img"
DATASET_PKL = "face_dataset_exp_1.pkl"
RESULTS_CSV = "face_scores_exp_1.csv"
CACHE_DIR = "embedings_exp_1/"
os.makedirs(CACHE_DIR, exist_ok=True)

MODELS = ["ArcFace", "Facenet", "VGG-Face", "SFace", "OpenFace"]

def get_pipeline(name: str) -> FacePipeline:
    return FacePipeline(name, cache_dir=CACHE_DIR)

def get_label(file1: str, file2: str) -> int:
    def _get_name(x):
        return x.split(os.sep)[-2]
    return int(_get_name(file1) == _get_name(file2))

def evaluate_pipeline(
    face_model: FacePipeline, 
    data: List[Tuple[str, str]], 
) -> Dict[str, Any]:
    scores = []
    labels = []
    distance_self = []
    distance_other = []
    elapsed_time = []

    for file1, file2 in tqdm(data, total=len(data), desc=f"Evaluating {face_model.name}"):
        _st = timeit.default_timer()
        
        # Обчислення ембедінгів та схожості
        similarity, distance = face_model(file1, file2)
        elapsed_time.append(timeit.default_timer() - _st)
        
        if similarity == -1.0:
            continue
            
        label = get_label(file1, file2)
        scores.append(similarity)
        labels.append(label)
        
        if label == 1:
            distance_self.append(distance)
        else:
            distance_other.append(distance)

    if not scores:
        print(f"Warning: No valid scores collected for {face_model.name}")
        return {}
        
    # Обчислення біометричних метрик
    ee_rate, thresh, fa_rate, fr_rate = compute_eer(scores, labels)
    min_dcf = compute_min_dcf(fr_rate, fa_rate)
    fa_score, fr_score = compute_far_frr(scores, labels, thresh)
    
    predictions = [1 if score >= thresh else 0 for score in scores]

    result = {
        "pipeline": face_model.name,
        "fa_score": fa_score,
        "fr_score": fr_score,
        "ee_rate": ee_rate,
        "dcf": min_dcf, 
        "threshold": thresh,
        "accuracy": accuracy_score(labels, predictions),
        "distance_self_mean": np.mean(distance_self),
        "distance_self_std": np.std(distance_self),
        "distance_other_mean": np.mean(distance_other),
        "distance_other_std": np.std(distance_other),
        "elapsed_time_mean": np.mean(elapsed_time),
        "elapsed_time_std": np.std(elapsed_time),
    }
    return result

def main():
    # 1. Підготовка набору даних
    if os.path.exists(DATASET_PKL):
        print(f"Loading {DATASET_PKL}")
        with open(DATASET_PKL, "rb") as f:
            dataset = pkl.load(f)
    else:
        print("Generating dataset (1:1 pairs)")
        dataset = data_utils.make_dataset(ROOT) 
        print(f"Saving {DATASET_PKL}")
        with open(DATASET_PKL, "wb") as f:
            pkl.dump(dataset, f)
    
    print(f"Number of pairs in dataset: {len(dataset)}")

    # 2. Ініціалізація та оцінювання моделей
    face_pipelines = [get_pipeline(model_name) for model_name in MODELS]
    results = {}

    for face_model in face_pipelines:
        print(f"\nEvaluating pipeline: {face_model.name}")
        results[face_model.name] = evaluate_pipeline(face_model, dataset)
    
    # 3. Збереження результатів
    pd.DataFrame(results).transpose().to_csv(RESULTS_CSV)
    print(f"\nDone! Results saved to {RESULTS_CSV}")

if __name__ == "__main__":
    if not os.path.exists(ROOT):
        print(f"Error: Dataset directory '{ROOT}' not found.")
    else:
        main()