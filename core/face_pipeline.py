import numpy as np
import json
import os
import cv2 
from typing import Callable, Tuple, Union

from deepface import DeepFace 
# Припускаємо, що ці функції доступні у вашому core
from core.cosine_similarity import cosine_similarity
from core.distance import euclidean_distance
from core.normalize import normalize

# --- Допоміжні функції для обличчя ---

def preprocess_image(path: str) -> Union[np.ndarray, None]:
    """Попередня обробка: BGR->RGB + згладжування (smoothing)."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

# --- Основний Клас Пайплайну для Обличчя ---

class FacePipeline:
    
    def __init__(
            self, 
            name: str, # Назва моделі DeepFace (e.g., 'ArcFace')
            cache_dir: str
        ):
        self.name = name
        self.cache_dir = cache_dir
        # Створення директорії кешу, якщо вона не існує
        os.makedirs(self.cache_dir, exist_ok=True)
        # Шлях до JSON-файлу кешу для цієї моделі
        self.cache_file_path = os.path.join(self.cache_dir, f"{self.name}.json")


    # --- Логіка Кешування (Читання) ---

    def _check_if_emb_exist(self, image_file: str) -> Union[np.ndarray, bool]:
        """Перевіряє, чи існує ембедінг для файлу в кеші."""
        
        if not os.path.isfile(self.cache_file_path):
            return False

        try:
            with open(self.cache_file_path, "r") as f:
                # Очікується структура: {speaker_id: {file_path: [emb_list]}}
                speakers_emd_dict = json.load(f)
        except json.JSONDecodeError:
            return False 

        # ID особи - це назва папки (передостанній елемент шляху)
        speaker = image_file.split(os.sep)[-2] 

        if speaker in speakers_emd_dict:
            speaker_files_emd_dict = speakers_emd_dict[speaker]
            if image_file in speaker_files_emd_dict:
                return np.array(speaker_files_emd_dict[image_file])
            
        return False

    # --- Логіка Кешування (Запис) ---
    
    def _cache_emb(self, image_file: str, emb: np.ndarray):
        """Зберігає ембедінг файлу в кеші."""
        speakers_emd_dict = {}

        if os.path.isfile(self.cache_file_path):
            try:
                with open(self.cache_file_path, "r") as f:
                    speakers_emd_dict = json.load(f)
            except json.JSONDecodeError:
                 speakers_emd_dict = {}

        speaker = image_file.split(os.sep)[-2]

        if speaker not in speakers_emd_dict:
            speakers_emd_dict[speaker] = {}
            
        # Зберігаємо ембедінг як список (JSON вимагає примітивних типів)
        speakers_emd_dict[speaker][image_file] = emb.tolist()
        
        with open(self.cache_file_path, "w") as f:
            json.dump(speakers_emd_dict, f, indent=4)
    
    # --- Обчислення Ембедінгу ---

    def _compute_embedding(self, image_file: str) -> Union[np.ndarray, None]:
        """Обчислює ембедінг, використовуючи DeepFace, або завантажує з кешу."""
        
        # 1. Спробувати завантажити з кешу
        cached_emb = self._check_if_emb_exist(image_file)
        if isinstance(cached_emb, np.ndarray):
            return cached_emb
        
        # 2. Якщо в кеші немає, обчислити
        try:
            img = preprocess_image(image_file)
            if img is None:
                return None
            
            rep = DeepFace.represent(
                img_path=img,
                model_name=self.name,
                detector_backend="mtcnn",
                enforce_detection=True,
                align=True
            )
            # DeepFace зазвичай повертає список результатів, беремо перший
            emb = np.array(rep[0]["embedding"]) 
            emb = normalize(emb) 
            
            # 3. Зберегти в кеш
            self._cache_emb(image_file, emb)
            
            return emb
        
        except Exception as e:
            # print(f"Error computing embedding for {image_file} using {self.name}: {e}")
            return None

    # --- Метод Виклику (__call__) для 1:1 Верифікації ---
    
    def __call__(self, file1: str, file2: str) -> Tuple[float, float]:
        """
        Порівнює два файли зображень: (Similarity, Distance).
        """
        # Отримуємо ембедінги (з кешу або обчислюємо)
        emb1 = self._compute_embedding(file1)
        emb2 = self._compute_embedding(file2)

        if emb1 is None or emb2 is None:
            # Помилка обробки або виявлення обличчя
            return -1.0, 2.0 
            
        similarity = cosine_similarity(emb1, emb2)
        # Використовуємо Евклідову дистанцію, як у вашому аудіо-коді
        distance = euclidean_distance(emb1, emb2) 

        return similarity, distance