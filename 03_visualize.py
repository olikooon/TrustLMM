"""
Шаг 3: Визуализация результатов.

Строит radar chart по результатам из results/scores.json.
Также можно сравнить с референсными значениями из оригинальной статьи (GPT-3.5).

Запуск:
    python 03_visualize.py
    python 03_visualize.py --model_name "Mistral-7B"
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Sans"

RESULTS_DIR = Path("results")
SCORES_FILE = RESULTS_DIR / "scores.json"

# Референсные значения GPT-3.5 из оригинальной статьи TrustLLM (Table 1 / Figure 3)
GPT35_REFERENCE = {
    "Hallucination (avg)": 0.657,
    "Jailbreak RtA":       0.625,
    "Sycophancy":          0.042,  # delta sim (из статьи)
}


def load_scores() -> dict:
    if not SCORES_FILE.exists():
        raise FileNotFoundError(
            f"Файл {SCORES_FILE} не найден. Сначала запустите 02_evaluate.py"
        )
    return json.loads(SCORES_FILE.read_text(encoding="utf-8"))


def extract_radar_values(scores: dict) -> dict[str, float]:
    """Нормализуем все метрики к шкале [0, 1] для radar chart."""
    values = {}

    # Hallucination: среднее из словаря
    if "hallucination" in scores and isinstance(scores["hallucination"], dict):
        values["Hallucination\n(avg acc)"] = scores["hallucination"].get("avg", 0)

    # Jailbreak: RtA напрямую [0,1]
    if "jailbreak_rta" in scores and scores["jailbreak_rta"] is not None:
        values["Jailbreak\n(RtA)"] = scores["jailbreak_rta"]

    # Sycophancy: delta может быть отрицательной, нормализуем к [0,1]
    if "sycophancy" in scores and scores["sycophancy"] is not None:
        raw = scores["sycophancy"]
        # Типичный диапазон: [-0.3, +0.3] → нормализуем к [0, 1]
        normalized = (raw + 0.3) / 0.6
        normalized = max(0.0, min(1.0, normalized))
        values["Sycophancy\n(нейтральность)"] = normalized

    return values


def radar_chart(model_values: dict[str, float], model_name: str,
                ref_values: dict[str, float] | None = None, out_path: Path = None):
    categories = list(model_values.keys())
    n = len(categories)
    if n < 2:
        print("[WARN] Недостаточно категорий для radar chart. Строю bar chart.")
        bar_chart(model_values, model_name, out_path)
        return

    scores = list(model_values.values())
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    scores_closed = scores + [scores[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    ax.plot(angles_closed, scores_closed, "o-", linewidth=2,
            color="steelblue", label=model_name)
    ax.fill(angles_closed, scores_closed, alpha=0.15, color="steelblue")

    if ref_values:
        ref_scores = [ref_values.get(c, 0) for c in categories]
        ref_closed = ref_scores + [ref_scores[0]]
        ax.plot(angles_closed, ref_closed, "s--", linewidth=1.5,
                color="tomato", label="GPT-3.5 (ref)")
        ax.fill(angles_closed, ref_closed, alpha=0.08, color="tomato")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title(f"TrustLLM: профиль надёжности\n{model_name}", pad=20, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    save_path = out_path or RESULTS_DIR / "radar.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Radar chart сохранён: {save_path}")
    plt.close()


def bar_chart(model_values: dict[str, float], model_name: str, out_path: Path = None):
    labels = list(model_values.keys())
    vals = list(model_values.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, vals, color="steelblue", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score (нормализованный к [0,1])")
    ax.set_title(f"TrustLLM: {model_name}", fontsize=12)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    save_path = out_path or RESULTS_DIR / "bar.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Bar chart сохранён: {save_path}")
    plt.close()


def print_table(model_name: str, model_values: dict, scores: dict):
    print(f"\n{'='*55}")
    print(f"  Модель: {model_name}")
    print(f"{'='*55}")
    print(f"  {'Метрика':<30} {'Значение':>10}  {'GPT-3.5':>10}")
    print(f"  {'-'*50}")

    ref_map = {
        "Hallucination\n(avg acc)":      0.657,
        "Jailbreak\n(RtA)":              0.625,
        "Sycophancy\n(нейтральность)":   0.550,  # нормализованный ref
    }
    for k, v in model_values.items():
        ref = ref_map.get(k, "—")
        ref_str = f"{ref:.3f}" if isinstance(ref, float) else ref
        clean_k = k.replace("\n", " ")
        print(f"  {clean_k:<30} {v:>10.3f}  {ref_str:>10}")
    print(f"{'='*55}")

    # Детали hallucination
    if "hallucination" in scores and isinstance(scores["hallucination"], dict):
        h = scores["hallucination"]
        print("\n  Hallucination (детали):")
        for sub in ["halu_qa", "halu_summ", "halu_dial", "mc"]:
            if sub in h:
                print(f"    {sub:<20} {h[sub]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="TrustLLM: визуализация")
    parser.add_argument("--model_name", default="Mistral-7B-Instruct",
                        help="Имя модели для заголовка графика")
    parser.add_argument("--no_ref", action="store_true",
                        help="Не показывать референсные значения GPT-3.5")
    args = parser.parse_args()

    scores = load_scores()
    model_values = extract_radar_values(scores)

    print_table(args.model_name, model_values, scores)

    ref = None if args.no_ref else {
        k: GPT35_REFERENCE.get(k.replace("\n", " ").split("(")[0].strip(), 0)
        for k in model_values
    }

    radar_chart(model_values, args.model_name, ref_values=ref)


if __name__ == "__main__":
    main()
