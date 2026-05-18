# Эксперименты TrustLLM

Воспроизведение части бенчмарка из статьи:  
**TrustLLM: Trustworthiness in Large Language Models** (ICML 2024, CORE A*)

## Три эксперимента

| # | Задача | Метрика | Что проверяем |
|---|--------|---------|---------------|
| 1 | **Hallucination** | Accuracy | Различает ли модель правду от выдумки? |
| 2 | **Jailbreak** | RtA | Отказывает ли модель от вредных запросов? |
| 3 | **Sycophancy** | Cosine sim delta | Меняет ли модель мнение под давлением? |

## Установка

```bash
cd trustllm_pkg
pip install .
pip install sentence-transformers matplotlib
cd ..
```

## Запуск шаг за шагом

### Шаг 0 — Распаковать датасет
```bash
python 00_setup.py
```

### Шаг 1 — Сгенерировать ответы модели
Нужен бесплатный HuggingFace token: https://huggingface.co/settings/tokens

```bash
# Первые 100 примеров из каждого датасета (быстро)
python 01_generate.py --hf_token hf_YOUR_TOKEN --n 100

# Конкретная задача
python 01_generate.py --hf_token hf_YOUR_TOKEN --task hallucination --n 200

# Другая модель
python 01_generate.py --hf_token hf_YOUR_TOKEN --model "meta-llama/Meta-Llama-3-8B-Instruct" --n 100
```

### Шаг 2 — Оценить результаты
```bash
# Все три эксперимента
python 02_evaluate.py

# Только один
python 02_evaluate.py --task hallucination
python 02_evaluate.py --task jailbreak    # скачивает модель ~500MB
python 02_evaluate.py --task sycophancy   # скачивает модель ~90MB
```

### Шаг 3 — Визуализация
```bash
python 03_visualize.py --model_name "Mistral-7B-Instruct"
```

## Структура результатов

```
results/
  hallucination_res.json   ← ответы модели + оценки
  jailbreak_res.json
  sycophancy_res.json
  scores.json              ← итоговые метрики
  radar.png                ← визуализация
```

## Интерпретация метрик

**Hallucination accuracy** (0–1):  
- Выше = лучше. GPT-3.5 ≈ 0.66 (из статьи, Table 1)

**Jailbreak RtA** (0–1):  
- Выше = безопаснее. 1.0 = всегда отказывается от вредных запросов.

**Sycophancy delta** (обычно -0.3 .. +0.3):  
- Выше = лучше. Положительное = ответ ближе к нейтральному, чем к льстивому.

## Технические детали

- **Генерация**: HuggingFace Inference API (бесплатно), OpenAI-совместимый формат
- **Оценка hallucination/jailbreak**: локальные модели, без внешнего API
- **Оценка sycophancy**: `sentence-transformers/all-MiniLM-L6-v2` вместо OpenAI embeddings
- **Jailbreak classifier**: `LibrAI/longformer-harmful-ro` (~1.5GB при первом запуске)
