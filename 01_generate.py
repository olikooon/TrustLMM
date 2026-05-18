"""
Шаг 1: Генерация ответов модели на датасеты TrustLLM.
Запуск:
    python 01_generate.py --api_key gsk_xxx
    python 01_generate.py --api_key gsk_xxx --task hallucination --n 50
    python 01_generate.py --api_key gsk_xxx --model llama-3.3-70b-versatile
"""

import argparse
import json
import time
from pathlib import Path
from openai import OpenAI

# Настройки 

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Бесплатные модели на Groq (30 req/min, без лимита по токенам в сутки)
DEFAULT_MODEL = "llama-3.1-8b-instant"   # быстрая, бесплатная
ALT_MODEL     = "llama-3.3-70b-versatile"  # лучше, но 1000 req/day лимит

DATASET_DIR = Path("dataset/dataset")
RESULTS_DIR = Path("results")

TASK_FILES = {
    "hallucination": DATASET_DIR / "truthfulness" / "hallucination.json",
    "jailbreak":     DATASET_DIR / "safety" / "jailbreak.json",
    "sycophancy":    DATASET_DIR / "truthfulness" / "sycophancy.json",
}

# Запрос к API 

def get_response(client: OpenAI, prompt: str, model: str,
                 max_tokens: int = 512, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err = str(e)[:150]
            wait = 2 ** attempt
            print(f"  [attempt {attempt+1}/{retries}] Ошибка: {err}. Жду {wait}s...")
            time.sleep(wait)
    return None


# Генерация для одного файла 

def generate_for_file(task: str, client: OpenAI, model: str, n: int | None) -> Path:
    src_path = TASK_FILES[task]
    if not src_path.exists():
        raise FileNotFoundError(f"Датасет не найден: {src_path}. Сначала запустите 00_setup.py")

    data = json.loads(src_path.read_text(encoding="utf-8"))
    if n:
        data = data[:n]

    out_path = RESULTS_DIR / f"{task}_res.json"

    # Resume: если файл уже есть — используем его как основу (сохраняет стратификацию)
    if out_path.exists():
        data = json.loads(out_path.read_text(encoding="utf-8"))
        done = {i for i, item in enumerate(data) if item.get("res")}
        print(f"  Найден прогресс: {len(done)}/{len(data)} уже готово.")
    else:
        done = set()

    print(f"\n[{task.upper()}] Генерирую ответы для {len(data)} примеров (модель: {model})...")
    for i, item in enumerate(data):
        if i in done:
            continue
        prompt = item.get("prompt", "")
        if not prompt:
            continue
        res = get_response(client, prompt, model)
        item["res"] = res
        if i % 10 == 0:
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {i+1}/{len(data)} готово...")
        # Groq: ~30 req/min → минимум 2 сек между запросами
        time.sleep(2)

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    success = sum(1 for item in data if item.get("res"))
    print(f"  Сохранено: {out_path}  ({success}/{len(data)} с ответами)")
    return out_path


# Main 

def main():
    parser = argparse.ArgumentParser(description="TrustLLM: генерация ответов через Groq API")
    parser.add_argument("--api_key", required=True,
                        help="Groq API key (gsk_...) — бесплатно на console.groq.com")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Модель (default: {DEFAULT_MODEL})")
    parser.add_argument("--task", default="all", choices=["all"] + list(TASK_FILES.keys()),
                        help="Какой датасет обрабатывать (default: all)")
    parser.add_argument("--n", type=int, default=None,
                        help="Взять первые N примеров (None = весь датасет)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    client = OpenAI(base_url=GROQ_BASE_URL, api_key=args.api_key)

    tasks = list(TASK_FILES.keys()) if args.task == "all" else [args.task]

    print(f"Модель: {args.model}")
    print(f"Задачи: {tasks}")
    print(f"Примеров: {args.n or 'все'}")

    for task in tasks:
        try:
            generate_for_file(task, client, args.model, args.n)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")

    print("\nГенерация завершена. Следующий шаг: python 02_evaluate.py")


if __name__ == "__main__":
    main()
