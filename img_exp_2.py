import os
import json
import time
import random
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from deepface import DeepFace
from sklearn.metrics import accuracy_score

from core.metrics import compute_eer, compute_min_dcf, compute_far_frr
from core.cosine_similarity import cosine_similarity


DATASET_FOLDER = "dataset_img"
MODELS = ["Facenet", "ArcFace", "VGG-Face", "SFace", "OpenFace"]
RESULTS_CSV = "scores_faces.csv"
JSON_DIR = "json_embeddings"
os.makedirs(JSON_DIR, exist_ok=True)



def split_images(folder, ratio=0.6):
    imgs = [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(imgs)
    k = int(len(imgs) * ratio)
    return imgs[:k], imgs[k:]


def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def get_embedding(path, model, use_preprocess=True):
    try:
        img = preprocess_image(path) if use_preprocess else path
        rep = DeepFace.represent(
            img_path=img,
            model_name=model,
            detector_backend="mtcnn",
            enforce_detection=True,
            align=True
        )
        return np.array(rep[0]["embedding"])
    except Exception as e:
        print(f"[{model}] Error on {path}: {e}")
        return None


def build_profiles(dataset, model, ratio=0.6):
    embeddings, test_data = {}, {}

    for person in os.listdir(dataset):
        path = os.path.join(dataset, person)
        if not os.path.isdir(path):
            continue

        enroll_imgs, test_imgs = split_images(path, ratio)
        test_data[person] = test_imgs

        embs = [get_embedding(img, model) for img in enroll_imgs]
        embs = [e for e in embs if e is not None]

        if embs:
            embeddings[person] = np.mean(embs, axis=0)


    save_path = os.path.join(JSON_DIR, f"{model}_avg_embeddings.json")
    with open(save_path, "w") as f:
        json.dump({k: v.tolist() for k, v in embeddings.items()}, f)
    print(f"[{model}] Profiles created: {len(embeddings)}")
    return embeddings, test_data


def evaluate(embeddings, test_data, model):
    scores, labels, dist_same, dist_diff, times = [], [], [], [], []

    for person, imgs in tqdm(test_data.items(), desc=f"Testing {model}"):
        for img in imgs:
            emb = get_embedding(img, model)
            if emb is None:
                continue

            start = time.time()
            for profile, avg_emb in embeddings.items():
                score = cosine_similarity(emb, avg_emb)
                label = int(person == profile)
                scores.append(score)
                labels.append(label)
                (dist_same if label else dist_diff).append(1 - score)
            times.append(time.time() - start)

    eer, thr, fa, fr = compute_eer(scores, labels)
    fa_s, fr_s = compute_far_frr(scores, labels, thr)
    acc = accuracy_score(labels, [s >= thr for s in scores])
    dcf = compute_min_dcf(fr_s, fa_s)

    return {
        "model": model,
        "EER": eer,
        "Threshold": thr,
        "FAR": fa_s,
        "FRR": fr_s,
        "DCF": dcf,
        "Accuracy": acc,
        "Dist_same_mean": np.mean(dist_same) if dist_same else 0.0,
        "Dist_same_std": np.std(dist_same) if dist_same else 0.0,
        "Dist_diff_mean": np.mean(dist_diff) if dist_diff else 0.0,
        "Dist_diff_std": np.std(dist_diff) if dist_diff else 0.0,
        "Time_mean": np.mean(times) if times else 0.0,
        "Time_std": np.std(times) if times else 0.0,
    }



if __name__ == "__main__":
    enroll_ratio = 0.6
    print(f"Face Verification: {enroll_ratio*100:.0f}% enrollment / {(1-enroll_ratio)*100:.0f}% test")

    results = []
    for model in MODELS:
        embeddings, test_data = build_profiles(DATASET_FOLDER, model, enroll_ratio)
        metrics = evaluate(embeddings, test_data, model)
        results.append(metrics)
        #print(f"[{model}] ✅ EER={metrics['EER']:.4f}, Thr={metrics['Threshold']:.4f}, Acc={metrics['Accuracy']:.4f}")

    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    print(f"\nAll metrics saved → {RESULTS_CSV}")
