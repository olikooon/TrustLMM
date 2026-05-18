"""
Шаг 0: Распаковка датасета и проверка структуры.
"""

import zipfile
import json
import os
from pathlib import Path

DATASET_ZIP = Path("dataset/dataset.zip")
DATASET_DIR = Path("dataset/dataset")


def extract_dataset():
    if not DATASET_ZIP.exists():
        print(f"[ERROR] Не найден файл {DATASET_ZIP}")
        print("Скачайте датасет вручную: https://github.com/HowieHwong/TrustLLM")
        return False

    print(f"Распаковываю {DATASET_ZIP} ...")
    with zipfile.ZipFile(DATASET_ZIP, "r") as z:
        z.extractall(DATASET_DIR)
    print("Готово.")
    return True


def show_structure():
    print("\n=== Структура датасета ===")
    for root, dirs, files in os.walk(DATASET_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        level = len(Path(root).relative_to(DATASET_DIR).parts)
        indent = "  " * level
        print(f"{indent}{Path(root).name}/")
        for f in sorted(files):
            if f.endswith(".json"):
                fpath = Path(root) / f
                try:
                    data = json.loads(fpath.read_text(encoding="utf-8"))
                    n = len(data) if isinstance(data, list) else "?"
                    keys = list(data[0].keys()) if isinstance(data, list) and data else []
                    print(f"{indent}  {f}  ({n} items)  keys: {keys[:6]}")
                except Exception:
                    print(f"{indent}  {f}")


def check_target_files():
    """Проверяем наличие нужных файлов для наших 3 экспериментов."""
    targets = {
        "Галлюцинации": DATASET_DIR / "truthfulness" / "hallucination.json",
        "Jailbreak":    DATASET_DIR / "safety" / "jailbreak.json",
        "Sycophancy":   DATASET_DIR / "truthfulness" / "sycophancy.json",
    }
    print("\n=== Файлы для экспериментов ===")
    all_ok = True
    for name, path in targets.items():
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            sources = set(item.get("source", "?") for item in data if isinstance(item, dict))
            print(f"  [OK] {name}: {path}  ({len(data)} items, sources={sources})")
        else:
            print(f"  [MISS] {name}: {path} — файл не найден")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    extract_dataset()
    show_structure()
    check_target_files()
    print("\nДатасет готов. Следующий шаг: python 01_generate.py")
