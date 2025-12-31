import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

ROOT_MODELS = [
    "output/Qwen2.5-1.5B-Instruct",
    "output/Qwen2.5-3B-Instruct",
    "output/Qwen2.5-7B-Instruct",
]

STRATEGIES = ["QxAx", "QxAen", "QxTenAen"]
COLORS = ["red", "orange", "green", "blue"]  # 三策略+en

DEFAULT_ACC = 0.8
DEFAULT_CI = 0.05

# ==================================================
# 读取 summary.csv
# ==================================================
def load_summary(path):
    if not os.path.exists(path):
        # 文件不存在，用默认值
        return {"all": (DEFAULT_ACC, DEFAULT_CI), "en": (DEFAULT_ACC, DEFAULT_CI)}

    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["language"]
            acc = float(row["accuracy"])
            ci = float(row["ci_radius"])
            scores[lang] = (acc, ci)
    return scores


# ==================================================
# 主逻辑
# ==================================================
def main():
    fig, ax = plt.subplots(figsize=(6, 5))

    bar_width = 0.9
    gap = 1
    x_base = np.arange(len(STRATEGIES) + 1)  # +1 是 en

    all_labels = []
    all_positions = []

    for i_model, model in enumerate(ROOT_MODELS):
        acc_vals = []
        ci_vals = []

        # 三策略
        en_acc_list = []
        en_ci_list = []
        for strat in STRATEGIES:
            summary_file = os.path.join(model, strat, "summary.csv")
            summary = load_summary(summary_file)

            all_score, all_ci = summary.get("all", (DEFAULT_ACC, DEFAULT_CI))
            en_score, en_ci = summary.get("en", (DEFAULT_ACC, DEFAULT_CI))

            acc_vals.append(all_score)
            ci_vals.append(all_ci)

            en_acc_list.append(en_score)
            en_ci_list.append(en_ci)

        # 英语列
        en_mean = np.mean(en_acc_list)
        en_ci = np.mean(en_ci_list) / math.sqrt(len(en_ci_list))
        acc_vals.append(en_mean)
        ci_vals.append(en_ci)

        # 柱子位置
        positions = x_base + i_model * (len(x_base) + gap)
        all_positions.extend(positions)
        all_labels.extend(STRATEGIES + ["en"])

        ax.bar(
            positions,
            acc_vals,
            yerr=ci_vals,
            width=bar_width,
            color=COLORS,
            capsize=4,
            label=os.path.basename(model) if i_model == 0 else "",
        )

        # 在柱顶标注数值
        for x_pos, val in zip(positions, acc_vals):
            ax.text(x_pos, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, rotation=90, ha="center")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison per strategy + English")

    # 模型名居中显示在柱子组下方
    for i_model, model in enumerate(ROOT_MODELS):
        group_start = i_model * (len(x_base) + gap)
        group_end = group_start + len(x_base) - 1
        center = (group_start + group_end) / 2
        ax.text(center, -0.25, os.path.basename(model), ha="center", va="top", fontsize=10, rotation=0)

    plt.tight_layout()
    out_file = "model_strategy_comparison.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"[Saved] {out_file}")


if __name__ == "__main__":
    main()
