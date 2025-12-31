import json
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置 =================
input_file = "output/Qwen2.5-3B-Instruct/QxTenAen-2step-PTI/attn_result.jsonl"
output_root = "output/Qwen2.5-3B-Instruct/QxTenAen-2step-PTI/plots"
os.makedirs(output_root, exist_ok=True)

# ================= 读取数据 =================
data = defaultdict(list)

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        lang = item["lang"]
        data[lang].append({
            "attn_p": item["attn_p"],             # [layer]
            "attn_t": item["attn_t"],             # [layer]
            "delta_p_len": item["delta_p_len"],   # [layer][head]
            "delta_t_len": item["delta_t_len"],   # [layer][head]
            "cos_tp": item["cos_tp"],             # [layer][head]
            "dot_tp": item["dot_tp"],             # [layer][head]
        })

# ================= 通用绘图函数（layer-head） =================
def plot_layer_head_curve(
    values, layers, title, ylabel, out_path, color_main="black", color_head="gray"
):
    num_layers, num_heads = values.shape
    mean_head = values.mean(axis=1)
    plt.figure(figsize=(10, 5))
    for h in range(num_heads):
        plt.plot(layers, values[:, h], color=color_head, alpha=0.3, linewidth=1)
    plt.plot(layers, mean_head, "-o", color=color_main, linewidth=2, label="head mean")
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ================= 主循环 =================
all_attn_p = []
all_attn_t = []

for lang, samples in data.items():
    lang_dir = os.path.join(output_root, lang)
    os.makedirs(lang_dir, exist_ok=True)

    # ---------- attn_p / attn_t ----------
    attn_p = np.array([s["attn_p"] for s in samples]).mean(axis=0)
    attn_t = np.array([s["attn_t"] for s in samples]).mean(axis=0)
    all_attn_p.append(attn_p)
    all_attn_t.append(attn_t)

    layers = np.arange(1, len(attn_p) + 1)

    # 图 1：绝对值
    plt.figure(figsize=(10, 5))
    plt.plot(layers, attn_p, "-o", color="blue", label="attn_p")
    plt.plot(layers, attn_t, "-o", color="orange", label="attn_t")
    plt.xlabel("Layer")
    plt.ylabel("Attention Ratio")
    plt.title(f"{lang} - Attention Absolute Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(lang_dir, f"{lang}_attn_abs.png"))
    plt.close()

    # 图 2：相对占比
    denom = attn_p + attn_t + 1e-8
    rel_p = attn_p / denom
    rel_t = attn_t / denom
    plt.figure(figsize=(10, 5))
    plt.plot(layers, rel_p, "-o", color="blue", label="p / (p + t)")
    plt.plot(layers, rel_t, "-o", color="orange", label="t / (p + t)")
    plt.xlabel("Layer")
    plt.ylabel("Relative Ratio")
    plt.title(f"{lang} - Attention Relative Ratio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(lang_dir, f"{lang}_attn_relative.png"))
    plt.close()

    # ---------- delta ----------
    delta_p = np.array([s["delta_p_len"] for s in samples]).mean(axis=0)
    delta_t = np.array([s["delta_t_len"] for s in samples]).mean(axis=0)

    plt.figure(figsize=(10, 5))
    for h in range(delta_p.shape[1]):
        plt.plot(layers, delta_p[:, h], color="blue", alpha=0.25, linewidth=1)
        plt.plot(layers, delta_t[:, h], color="orange", alpha=0.25, linewidth=1)
    plt.plot(layers, delta_p.mean(axis=1), "-o", color="blue", linewidth=2, label="delta_p_len (mean)")
    plt.plot(layers, delta_t.mean(axis=1), "-o", color="orange", linewidth=2, label="delta_t_len (mean)")
    plt.xlabel("Layer")
    plt.ylabel("Delta Length")
    plt.title(f"{lang} - Delta Length (p vs t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(lang_dir, f"{lang}_delta_p_t.png"))
    plt.close()

    plot_layer_head_curve(
        delta_t - delta_p,
        layers,
        title=f"{lang} - Delta Difference (p - t)",
        ylabel="Delta Difference",
        out_path=os.path.join(lang_dir, f"{lang}_delta_diff.png"),
    )

    # ---------- cos_tp ----------
    cos_tp = np.array([s["cos_tp"] for s in samples]).mean(axis=0)
    plot_layer_head_curve(
        cos_tp, layers,
        title=f"{lang} - cos_tp across layers",
        ylabel="cos_tp",
        out_path=os.path.join(lang_dir, f"{lang}_cos_tp.png"),
        color_main="purple",
        color_head="purple"
    )

    # ---------- dot_tp ----------
    dot_tp = np.array([s["dot_tp"] for s in samples]).mean(axis=0)
    plot_layer_head_curve(
        dot_tp, layers,
        title=f"{lang} - dot_tp across layers",
        ylabel="dot_tp",
        out_path=os.path.join(lang_dir, f"{lang}_dot_tp.png"),
        color_main="green",
        color_head="green"
    )

# ================= 汇总 total 图 =================
total_dir = os.path.join(output_root, "total")
os.makedirs(total_dir, exist_ok=True)

# ---------- attn_p / attn_t ----------
all_attn_p = []
all_attn_t = []

# ---------- delta / cos / dot ----------
all_delta_p = []
all_delta_t = []
all_cos_tp = []
all_dot_tp = []

for lang, samples in data.items():
    # attn
    attn_p = np.array([s["attn_p"] for s in samples]).mean(axis=0)
    attn_t = np.array([s["attn_t"] for s in samples]).mean(axis=0)
    all_attn_p.append(attn_p)
    all_attn_t.append(attn_t)

    # delta
    delta_p = np.array([s["delta_p_len"] for s in samples]).mean(axis=0)
    delta_t = np.array([s["delta_t_len"] for s in samples]).mean(axis=0)
    all_delta_p.append(delta_p)
    all_delta_t.append(delta_t)

    # cos / dot
    cos_tp = np.array([s["cos_tp"] for s in samples]).mean(axis=0)
    dot_tp = np.array([s["dot_tp"] for s in samples]).mean(axis=0)
    all_cos_tp.append(cos_tp)
    all_dot_tp.append(dot_tp)

layers = np.arange(1, all_attn_p[0].shape[0] + 1)

# ----------- 绘制 attn ----------- #
def plot_total_curve(all_values, ylabel, filename, color_main, color_head=None, delta=False):
    """
    all_values: list[np.array] 每个元素 shape: [layers] 或 [layers, heads]
    delta: True -> 有 head 维度
    """
    plt.figure(figsize=(10, 5))
    all_values = np.array(all_values)
    if delta:
        # head均值
        mean_values = all_values.mean(axis=(0,2))  # total mean across langs and heads
        # 各语言均值
        for lv in all_values:
            plt.plot(layers, lv.mean(axis=1), "--", color=color_main, alpha=0.5)
    else:
        mean_values = all_values.mean(axis=0)
        for lv in all_values:
            plt.plot(layers, lv, "--", color=color_main, alpha=0.5)
    plt.plot(layers, mean_values, "-o", color=color_main, linewidth=2, label="total mean")
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(f"Total - {ylabel} Across Languages")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(total_dir, filename))
    plt.close()

# attn 绝对值
plot_total_curve(all_attn_p, "attn_p", "attn_p_total.png", "blue")
plot_total_curve(all_attn_t, "attn_t", "attn_t_total.png", "orange")
# 相对占比
rel_p = np.array(all_attn_p) / (np.array(all_attn_p) + np.array(all_attn_t) + 1e-8)
rel_t = np.array(all_attn_t) / (np.array(all_attn_p) + np.array(all_attn_t) + 1e-8)
plot_total_curve(rel_p, "p/(p+t)", "attn_relative_p_total.png", "blue")
plot_total_curve(rel_t, "t/(p+t)", "attn_relative_t_total.png", "orange")

# ----------- delta ---------- #
plot_total_curve(all_delta_p, "delta_p_len", "delta_p_len_total.png", "blue", delta=True)
plot_total_curve(all_delta_t, "delta_t_len", "delta_t_len_total.png", "orange", delta=True)
# 差值
delta_diff = [dt - dp for dp, dt in zip(all_delta_p, all_delta_t)]
plot_total_curve(delta_diff, "delta_diff", "delta_diff_total.png", "black", delta=True)

# ----------- cos_tp / dot_tp ----------- #
plot_total_curve(all_cos_tp, "cos_tp", "cos_tp_total.png", "purple", delta=True)
plot_total_curve(all_dot_tp, "dot_tp", "dot_tp_total.png", "green", delta=True)


print(f"All plots saved under {output_root}/, including total summary.")
