import os
import csv
import json
import numpy as np
import matplotlib

# 非 GUI 后端
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "output/Qwen2.5-3B-Instruct/QxTenAen-2step-PTI"

SUMMARY_FILE = os.path.join(ROOT, "summary.csv")
ATTN_FILE = os.path.join(ROOT, "attn_result.jsonl")
OUT_FILE = os.path.join(ROOT, "costp_vs_accuracy_flipped.png")


# ==================================================
# 读取 summary.csv -> accuracy
# ==================================================
def load_scores(path):
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["language"]] = float(row["accuracy"])
    return scores


# ==================================================
# 读取 attn_result.jsonl -> mean cos_tp
# ==================================================
def load_cos_tp_means(path):
    buf = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            lang = item["lang"]
            cos_tp = np.array(item["cos_tp"], dtype=np.float32)
            buf.setdefault(lang, []).append(cos_tp.mean())

    return {k: float(np.mean(v)) for k, v in buf.items()}


# ==================================================
# 主逻辑
# ==================================================
def main():
    scores = load_scores(SUMMARY_FILE)
    cos_tp_means = load_cos_tp_means(ATTN_FILE)

    # 只保留交集，并按 cos_tp 排序
    langs = sorted(
        set(scores) & set(cos_tp_means),
        key=lambda l: cos_tp_means[l],
        reverse=True,
    )

    cos_vals = np.array([cos_tp_means[l] for l in langs])
    acc_vals = np.array([scores[l] for l in langs])

    x = np.arange(len(langs))
    width = 0.8

    # ==================================================
    # 绘图：上下两个 subplot
    # ==================================================
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 6), sharex=True
    )

    # 上图：cos_tp
    ax1.bar(x, cos_vals, width=width, color="steelblue")
    ax1.set_ylabel("Mean cos_tp")
    ax1.set_title("Mean cos_tp (top) & Accuracy (bottom, flipped) per language")
    for i, v in enumerate(cos_vals):
        ax1.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # 下图：accuracy 倒置显示
    ax2.bar(x, acc_vals, width=width, color="orange")
    ax2.set_ylabel("Accuracy")
    ax2.invert_yaxis()  # 倒置 y 轴

    for i, v in enumerate(acc_vals):
        ax2.text(i, v - 0.002, f"{v:.3f}", ha="center", va="top", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(langs, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=200)
    plt.close()
    print(f"[Saved] {OUT_FILE}")


if __name__ == "__main__":
    main()
