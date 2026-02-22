import os
import random
from itertools import combinations
import pickle as pkl
from typing import List, Tuple

def find_images(root: str) -> List[str]:
    """Знаходить усі шляхи до файлів зображень."""
    entries = []
    for person in os.listdir(root):
        person_dir = os.path.join(root, person)
        if os.path.isdir(person_dir):
            for f in os.listdir(person_dir):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    entries.append(os.path.join(person_dir, f)) 
    return entries

def make_dataset(root: str) -> List[Tuple[str, str]]:
    """
    Генерує пари для 1:1 верифікації (свої та чужі).
    """
    all_files = find_images(root)
    
    # Групуємо файли за ID особи
    grouped_files = {}
    for fp in all_files:
        # ID особи - назва папки (передостанній елемент шляху)
        person_id = fp.split(os.sep)[-2] 
        if person_id not in grouped_files:
            grouped_files[person_id] = []
        grouped_files[person_id].append(fp)

    pairs = []
    ids = list(grouped_files.keys())
    
    # 1. Same-user (Свої): попарне порівняння всередині кожної особи
    for person_id in ids:
        files = grouped_files[person_id]
        if len(files) >= 2:
            same_pairs = list(combinations(files, 2))
            pairs.extend(same_pairs)
            
    # 2. Imposter (Чужі): попарне порівняння між різними особами
    # Генерація такої ж кількості чужих пар, як і своїх, для балансу
    num_same_pairs = len(pairs)
    imposter_pairs = []
    
    while len(imposter_pairs) < num_same_pairs:
        # Випадково вибираємо два різні ID
        id1, id2 = random.sample(ids, 2)
        
        # Вибираємо випадковий файл з кожного спікера
        if grouped_files[id1] and grouped_files[id2]:
            f1 = random.choice(grouped_files[id1])
            f2 = random.choice(grouped_files[id2])
            
            # Уникаємо дублікатів (хоча унікальність тут гарантується різними ID)
            pair = tuple(sorted((f1, f2)))
            if pair not in imposter_pairs:
                 imposter_pairs.append(pair)
    
    pairs.extend(imposter_pairs)
    random.shuffle(pairs)
    
    return pairs