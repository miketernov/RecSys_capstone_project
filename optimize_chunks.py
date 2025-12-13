"""
Скрипт для оптимизации загрузки рецептов:
1. Объединяет все чанки в один файл
2. Опционально: сжимает данные
"""
import json
import os
from pathlib import Path

CHUNKS_DIR = "chunks"
OUTPUT_FILE = "recipes_all.json"

def combine_chunks():
    """Объединяет все чанки в один файл"""
    print("📥 Загрузка чанков...")
    
    all_recipes = []
    chunk_files = sorted([f for f in os.listdir(CHUNKS_DIR) if f.startswith("part") and f.endswith(".json")], 
                         key=lambda x: int(x.replace("part", "").replace(".json", "")))
    
    for chunk_file in chunk_files:
        chunk_path = os.path.join(CHUNKS_DIR, chunk_file)
        print(f"  Загружаю {chunk_file}...")
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
            all_recipes.extend(chunk_data)
    
    print(f"✔ Загружено {len(all_recipes)} рецептов")
    
    # Сохраняем объединенный файл
    print(f"💾 Сохраняю в {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_recipes, f, ensure_ascii=False)
    
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✔ Готово! Размер файла: {file_size_mb:.2f} MB")
    
    return OUTPUT_FILE

if __name__ == "__main__":
    combine_chunks()

