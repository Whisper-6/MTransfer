import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict

ROOT = "./output/Qwen2.5-1.5B-Instruct/QxTenAen-2step-hf-mask"


# ==================================================
# 1. 读取所有层 & 语言结果
# ==================================================
def load_layer_results(root):
    layer_dirs = [
        d for d in os.listdir(root)
        if d.startswith("L") and d[1:].isdigit()
    ]
    layer_dirs = sorted(layer_dirs, key=lambda x: int(x[1:]))

    layers = []
    layer_lang_scores = {}           # layer -> [(lang, acc)]
    lang_layer_scores = defaultdict(dict)  # lang -> {layer: acc}
    layer_ci = {}                    # layer -> mean_ci

    for layer_name in layer_dirs:
        layer_idx = int(layer_name[1:])
        csv_path = os.path.join(root, layer_name, "summary.csv")

        accs = []
        cis = []
        lang_scores = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lang = row["language"]
                acc = float(row["accuracy"])
                ci = float(row["ci_radius"])

                accs.append(acc)
                cis.append(ci)
                lang_scores.append((lang, acc))
                lang_layer_scores[lang][layer_idx] = acc

        layers.append(layer_idx)
        layer_lang_scores[layer_idx] = lang_scores
        layer_ci[layer_idx] = np.mean(cis) / math.sqrt(len(accs))

    layers = np.array(sorted(layers))
    return layers, layer_lang_scores, lang_layer_scores, layer_ci


# ==================================================
# 2. 图 1：所有语言均值 vs Layer
# ==================================================
def plot_layer_mean(layers, layer_lang_scores, layer_ci, root):
    mean_acc = []
    mean_ci = []

    for l in layers:
        accs = [acc for _, acc in layer_lang_scores[l]]
        mean_acc.append(np.mean(accs))
        mean_ci.append(layer_ci[l])

    mean_acc = np.array(mean_acc)
    mean_ci = np.array(mean_ci)

    plt.figure(figsize=(7, 5))
    plt.plot(layers, mean_acc, linewidth=2)
    plt.fill_between(
        layers,
        mean_acc - mean_ci,
        mean_acc + mean_ci,
        alpha=0.2
    )

    plt.xlabel("Layer")
    plt.ylabel("Mean Accuracy")
    plt.title("Mean Accuracy Across Languages vs Layer")
    plt.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(root, "layer_mean_accuracy.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved:", out_path)


# ==================================================
# 3. 图 2：每个语言的分层折线
# ==================================================
def plot_per_language(layers, lang_layer_scores, root):
    plt.figure(figsize=(8, 6))

    cmap = cm.get_cmap("tab10", len(lang_layer_scores))

    for i, (lang, layer_scores) in enumerate(sorted(lang_layer_scores.items())):
        ys = [layer_scores[l] for l in layers]
        plt.plot(
            layers,
            ys,
            label=lang,
            linewidth=2,
            color=cmap(i)
        )

    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Per-Language Accuracy vs Layer")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(root, "per_language_layer_accuracy.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved:", out_path)


# ==================================================
# 4. main
# ==================================================
if __name__ == "__main__":
    layers, layer_lang_scores, lang_layer_scores, layer_ci = load_layer_results(ROOT)

    plot_layer_mean(layers, layer_lang_scores, layer_ci, ROOT)
    plot_per_language(layers, lang_layer_scores, ROOT)
