import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]
STRATEGIES = ["QxAx", "QxAen", "QxTenAen"]
LANGS = ["bn", "de", "es", "fr", "ja", "ru", "th"]
DEFAULT_ACC = 0.8
DEFAULT_CI = 0.05
STRAT_COLORS = ["red", "orange", "green"]

def load_summary(path):
    if not os.path.exists(path):
        return {lang: (DEFAULT_ACC, DEFAULT_CI) for lang in LANGS + ["en"]}
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["language"]
            acc = float(row["accuracy"])
            ci = float(row["ci_radius"])
            scores[lang] = (acc, ci)
    return scores

def plot_model_radar(model):
    model_dir = os.path.join("output", model)
    theta = np.linspace(0, 2*np.pi, len(LANGS), endpoint=False).tolist()
    theta += theta[:1]  # 封闭

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    # 每个策略
    for i, strat in enumerate(STRATEGIES):
        summary_file = os.path.join(model_dir, strat, "summary.csv")
        summary = load_summary(summary_file)

        values = [summary[lang][0] for lang in LANGS]
        ci = [summary[lang][1] for lang in LANGS]

        values += values[:1]
        ci += ci[:1]

        ax.plot(theta, values, color=STRAT_COLORS[i], linewidth=2, label=strat)
        lower = np.array(values) - np.array(ci)
        upper = np.array(values) + np.array(ci)
        ax.fill_between(theta, lower, upper, color=STRAT_COLORS[i], alpha=0.2)

    # 英语均值多边形
    en_values = []
    en_ci_list = []
    for strat in STRATEGIES:
        summary_file = os.path.join(model_dir, strat, "summary.csv")
        summary = load_summary(summary_file)
        en_values.append(summary["en"][0])
        en_ci_list.append(summary["en"][1])
    en_mean = np.mean(en_values)
    en_ci = np.mean(en_ci_list) / math.sqrt(len(STRATEGIES))
    en_polygon = [en_mean]*len(LANGS)
    en_polygon += en_polygon[:1]
    ax.plot(theta, en_polygon, color="blue", linestyle="--", linewidth=2, label="en mean")
    ax.fill_between(theta, [en_mean-en_ci]*len(theta), [en_mean+en_ci]*len(theta), color="blue", alpha=0.2)

    # 设置雷达图
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(LANGS)
    ax.set_yticklabels([])  # 不显示 y 数值
    ax.set_ylim(0,1)
    ax.set_title(f"Model: {model}", fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2,1.1))

    out_file = os.path.join(model_dir, "radar.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"[Saved] {out_file}")

def main():
    for model in MODELS:
        plot_model_radar(model)

if __name__ == "__main__":
    main()
