import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# 参数
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--result-dir",
    type=str,
    default=".",
    help="Directory containing experiment result folders (default: current directory)"
)
parser.add_argument("--min",type=float,default=0.0)
parser.add_argument("--max",type=float,default=1.0)
parser.add_argument("--scale",type=float,default=0.1)
args = parser.parse_args()

root_dir = args.result_dir
radar_min, radar_max = args.min, args.max  # 雷达图刻度范围
# ----------------------
# 统计所有文件夹 en 总体正确率
# ----------------------
en_correct_total = 0
en_trials_total = 0

folders = sorted([
    f for f in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, f))
])

for folder in folders:
    csv_path = os.path.join(root_dir, folder, "result.csv")
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    en_row = df[(df["language"] == "en") & (df["source"] == "total")]
    if not en_row.empty:
        en_correct_total += en_row["correct"].values[0]
        en_trials_total += en_row["total"].values[0]

if en_trials_total > 0:
    en_acc_global = en_correct_total / en_trials_total
else:
    en_acc_global = 0.0

print(f"Global en accuracy across all folders: {en_acc_global:.4f}")

# ----------------------
# 遍历每个文件夹生成雷达图
# ----------------------
for folder in folders:
    csv_path = os.path.join(root_dir, folder, "result.csv")
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    df_total = df[df["source"] == "total"]

    # ----------------------
    # 其他语言（排除 en，字典序排序）
    # ----------------------
    lang_rows = df_total[df_total["language"] != "en"].copy()
    if lang_rows.empty:
        continue

    lang_rows = lang_rows.sort_values("language")
    languages = lang_rows["language"].tolist()
    accuracies = lang_rows["accuracy"].tolist()
    ci_radii = lang_rows["ci_radius"].tolist()

    # ----------------------
    # 雷达图准备
    # ----------------------
    N = len(languages)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    values = accuracies + accuracies[:1]
    ci_upper = [min(v + c, 1.0) for v, c in zip(accuracies, ci_radii)]
    ci_lower = [max(v - c, 0.0) for v, c in zip(accuracies, ci_radii)]
    ci_upper.append(ci_upper[0])
    ci_lower.append(ci_lower[0])

    # ----------------------
    # 绘图
    # ----------------------
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # global en 背景多边形
    en_polygon = [en_acc_global] * N
    en_polygon += en_polygon[:1]
    ax.fill(
        np.linspace(0, 2 * np.pi, N + 1),
        en_polygon,
        alpha=0.2,
        label=f"Global en={en_acc_global:.3f}"
    )

    # 语言准确率
    ax.plot(angles, values, 'o-', linewidth=2, label='Accuracy')
    ax.fill_between(angles, ci_lower, ci_upper, alpha=0.3, label='CI radius')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(languages)
    ax.set_ylim(radar_min, radar_max)
    ax.set_yticks(np.linspace(radar_min, radar_max, int((radar_max-radar_min+1e-6)//args.scale) + 1))
    ax.set_title(f"Language Accuracy Comparison ({folder})", size=14)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15))

    # 保存图片
    out_path = os.path.join(root_dir, folder, "radar.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
