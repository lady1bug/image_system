import os
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from sklearn.metrics import accuracy_score, roc_curve
from deepface import DeepFace
from core.metrics import compute_eer, compute_min_dcf, compute_far_frr
from core.cosine_similarity import cosine_similarity


ROOT = "dataset_img"
OUT_DIR = "plots_images"
MODELS = ["Facenet", "ArcFace", "VGG-Face", "SFace", "OpenFace"]
os.makedirs(OUT_DIR, exist_ok=True)


# --- Допоміжні функції ---
def find_images(root):
    return [
        (person, os.path.join(root, person, f))
        for person in os.listdir(root)
        if os.path.isdir(os.path.join(root, person))
        for f in os.listdir(os.path.join(root, person))
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

def preprocess_image(path):
    """Попередня обробка: RGB + легке згладжування"""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def compute_embedding(model, path, use_preprocess=True):
    """Обчислення ембедінгу для моделі через DeepFace"""
    try:
        img = preprocess_image(path) if use_preprocess else path
        rep = DeepFace.represent(
            img_path=img,
            model_name=model,
            detector_backend="mtcnn",  # 'retinaface' для GPU
            enforce_detection=True,
            align=True
        )
        return np.array(rep[0]["embedding"])
    except Exception as e:
        print(f"[{model}] Error on {path}: {e}")
        return None

# --- Основна функція ---
def evaluate_pipeline(model_name, entries):
    embeddings = {
        fp: emb for _, fp in tqdm(entries, desc=f"Compute {model_name} embeddings")
        if (emb := compute_embedding(model_name, fp)) is not None
    }

    pairs = list(combinations(embeddings.keys(), 2))
    scores, labels, dist_same, dist_diff, times = [], [], [], [], []

    for f1, f2 in tqdm(pairs, desc=f"Compare pairs ({model_name})"):
        start = time.time()
        sim = cosine_similarity(embeddings[f1], embeddings[f2])
        times.append(time.time() - start)

        label = int(f1.split(os.sep)[-2] == f2.split(os.sep)[-2])
        scores.append(sim)
        labels.append(label)

        dist = 1 - sim
        (dist_same if label else dist_diff).append(dist)

    eer, thr, fa, fr = compute_eer(scores, labels)
    dcf = compute_min_dcf(fr, fa)
    fa_s, fr_s = compute_far_frr(scores, labels, thr)
    preds = np.array(scores) >= thr

    return {
        "model": model_name,
        "EER": eer,
        "Threshold": thr,
        "FAR": fa_s,
        "FRR": fr_s,
        "DCF": dcf,
        "Accuracy": accuracy_score(labels, preds),
        "Dist_same_mean": np.mean(dist_same),
        "Dist_same_std": np.std(dist_same),
        "Dist_diff_mean": np.mean(dist_diff),
        "Dist_diff_std": np.std(dist_diff),
        "Time_mean": np.mean(times),
        "Time_std": np.std(times),
    }

# --- Точка входу ---
if __name__ == "__main__":
    entries = find_images(ROOT)
    print(f"Found {len(entries)} images")

    results = [evaluate_pipeline(model, entries) for model in MODELS]
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "face_results.csv")
    df.to_csv(csv_path, index=False)

    print("Results saved to:", csv_path)