import os
import json
import numpy as np
import cv2
from deepface import DeepFace
from tqdm import tqdm
from typing import List, Dict, Union, Tuple

DATASET_FOLDER = "dataset_img"
MODELS = ["Facenet", "ArcFace", "VGG-Face", "SFace", "OpenFace"]
JSON_DIR = "avg_embedings" 
os.makedirs(JSON_DIR, exist_ok=True)

def preprocess_image(path: str) -> Union[np.ndarray, None]:
    img = cv2.imread(path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def get_embedding(path: str, model: str) -> Union[np.ndarray, None]:
    try:
        img_input = preprocess_image(path)
        if img_input is None: return None
            
        rep = DeepFace.represent(
            img_path=img_input,
            model_name=model,
            detector_backend="mtcnn",
            enforce_detection=True,
            align=True
        )
        emb = np.array(rep[0]["embedding"])
        return emb
    except Exception:
        return None

def find_all_images(folder: str) -> Dict[str, List[str]]:
    grouped_images = {}
    for person in os.listdir(folder):
        person_path = os.path.join(folder, person)
        if os.path.isdir(person_path):
            imgs = [os.path.join(person_path, f) 
                    for f in os.listdir(person_path) 
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            if imgs:
                grouped_images[person] = imgs
    return grouped_images

def load_grouped_dataset(folder: str) -> List[List[str]]:
    grouped_files = []
    all_person_images = find_all_images(folder) 
    for img_list in all_person_images.values():
        grouped_files.append(img_list)
    return grouped_files

def build_profiles(dataset_folder: str, model: str):
    all_person_images = find_all_images(dataset_folder)
    embeddings: Dict[str, np.ndarray] = {}
    
    print(f"[{model}] Calculating average embeddings for {len(all_person_images)} people...")

    for person, img_paths in tqdm(all_person_images.items()):
        embs = [get_embedding(img, model) for img in img_paths] 
        embs = [e for e in embs if e is not None]

        if embs:
            embeddings[person] = np.mean(embs, axis=0)

    save_path = os.path.join(JSON_DIR, f"{model}.json")
    with open(save_path, "w") as f:
        json.dump({k: v.tolist() for k, v in embeddings.items()}, f, indent=4) 
    print(f"[{model}] Profiles saved to {save_path}. Total profiles: {len(embeddings)}")
    return embeddings

if __name__ == "__main__":
    if not os.path.exists(DATASET_FOLDER):
         print(f"Error: Dataset directory '{DATASET_FOLDER}' not found.")
         print("Please ensure your dataset is structured as: dataset_img/Person_ID/image.jpg")
    else:
        for model in MODELS:
            build_profiles(DATASET_FOLDER, model)