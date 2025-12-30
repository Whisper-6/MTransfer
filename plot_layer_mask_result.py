import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt


ROOT = "./output/Qwen2.5-1.5B-Instruct/QxTenAen-2step-hf-mask"
TOTAL_LAYERS = 28


def load_cover_results(root):
    """
    读取所有 L[l, r] / None 文件夹
    返回:
        cover_results[(l, r)] = (mean_acc, mean_ci)
        其中 None -> (None, None)
    """
    cover_results = {}

    for d in os.listdir(root):
        if d == "None":
            l, r = None, None
        else:
            if not (d.startswith("L[") and d.endswith("]")):
                continue
            inside = d[2:-1].strip()
            l, r = map(int, inside.split(","))

        csv_path = os.path.join(root, d, "summary.csv")
        if not os.path.exists(csv_path):
            continue

        accs = []
        cis = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                accs.append(float(row["accuracy"]))
                cis.append(float(row["ci_radius"]))

        if len(accs) == 0:
            continue

        mean_acc = np.mean(accs)
        mean_ci = np.mean(cis) / math.sqrt(len(accs))

        cover_results[(l, r)] = (mean_acc, mean_ci)

    return cover_results


# ==================================================
# 2. 提取前向 / 后向覆盖趋势
# ==================================================
def extract_trends(cover_results, total_layers):
    """
    返回:
        prefix: [(x, acc, ci)]  对应 None + L[0, x]
        suffix: [(x, acc, ci)]  对应 L[x, total_layers] + None
    """
    if (None, None) not in cover_results:
        raise ValueError("Missing baseline result")

    base_acc, base_ci = cover_results[(None, None)]

    # ---------- 前向覆盖 ----------
    prefix = [(0, base_acc, base_ci)]
    for (l, r), (acc, ci) in cover_results.items():
        if l == 0:
            prefix.append((r, acc, ci))

    prefix = sorted(set(prefix), key=lambda x: x[0])

    # ---------- 后向覆盖 ----------
    suffix = [(total_layers, base_acc, base_ci)]
    for (l, r), (acc, ci) in cover_results.items():
        if r == total_layers:
            suffix.append((l, acc, ci))

    suffix = sorted(set(suffix), key=lambda x: x[0])

    return prefix, suffix


# ==================================================
# 3. 绘图：同一张图展示两条趋势
# ==================================================
def plot_cover_trends(prefix, suffix, root):
    plt.figure(figsize=(8, 5.5))

    # -------- 前向覆盖 --------
    x1, y1, ci1 = zip(*prefix)
    plt.errorbar(
        x1, y1,
        yerr=ci1,
        marker="o",
        linestyle="-",
        linewidth=2,
        capsize=3,
        label="Cover from front: L[0, x]"
    )

    # -------- 后向覆盖 --------
    x2, y2, ci2 = zip(*suffix)
    plt.errorbar(
        x2, y2,
        yerr=ci2,
        marker="o",
        linestyle="-",
        linewidth=2,
        capsize=3,
        label="Cover from back: L[x, 28]"
    )

    plt.xlabel("Coverage Boundary Layer")
    plt.ylabel("Mean Accuracy")
    plt.title("Accuracy vs Coverage Range")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    out_path = os.path.join(root, "coverage_trends.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved:", out_path)


# ==================================================
# 4. main
# ==================================================
if __name__ == "__main__":
    cover_results = load_cover_results(ROOT)
    prefix, suffix = extract_trends(cover_results, TOTAL_LAYERS)
    plot_cover_trends(prefix, suffix, ROOT)
