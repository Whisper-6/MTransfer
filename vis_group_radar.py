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
    help="Root directory containing subfolders with result.csv (default: current directory)"
)
parser.add_argument("--min", type=float, default=0.0)
parser.add_argument("--max", type=float, default=1.0)
parser.add_argument("--scale", type=float, default=0.1)
args = parser.parse_args()

root_dir = args.result_dir
radar_min, radar_max = args.min, args.max

subfolders_to_plot = ["QxAx", "QxAen", "QxTenAen"]


# ----------------------
# 收集数据
# ----------------------
data_dict = {}          # folder -> {lang: (acc, ci)}
all_languages = set()

for folder in subfolders_to_plot:
    csv_path = os.path.join(root_dir, folder, "result.csv")
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    df_total = df[df["source"] == "total"]

    lang_acc = {}
    for _, row in df_total.iterrows():
        lang = row["language"]
        acc = row["accuracy"]
        ci = row["ci_radius"]
        lang_acc[lang] = (acc, ci)
        if lang != "en":
            all_languages.add(lang)

    data_dict[folder] = lang_acc

if not data_dict:
    print("No data found.")
    exit(0)


# ----------------------
# 雷达图维度
# ----------------------
languages = sorted(list(all_languages))
N = len(languages)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]


# ======================
# 雷达图
# ======================
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors = ['r', 'g', 'b', 'orange', 'purple']

for i, folder in enumerate(subfolders_to_plot):
    if folder not in data_dict:
        continue

    lang_acc = data_dict[folder]

    values = [lang_acc[l][0] for l in languages]
    values += values[:1]

    ci_upper = [min(lang_acc[l][0] + lang_acc[l][1], 1.0) for l in languages]
    ci_lower = [max(lang_acc[l][0] - lang_acc[l][1], 0.0) for l in languages]
    ci_upper += ci_upper[:1]
    ci_lower += ci_lower[:1]

    ax.plot(
        angles, values, 'o-',
        linewidth=2,
        color=colors[i % len(colors)],
        label=folder
    )
    ax.fill_between(
        angles,
        ci_lower,
        ci_upper,
        color=colors[i % len(colors)],
        alpha=0.2
    )

# ----------------------
# global en 背景
# ----------------------
en_correct_total = 0
en_trials_total = 0

for folder in subfolders_to_plot:
    csv_path = os.path.join(root_dir, folder, "result.csv")
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    en_row = df[(df["language"] == "en") & (df["source"] == "total")]
    if not en_row.empty:
        en_correct_total += en_row["correct"].values[0]
        en_trials_total += en_row["total"].values[0]

en_acc_global = en_correct_total / en_trials_total if en_trials_total > 0 else 0.0
ax.fill(
    np.linspace(0, 2 * np.pi, N + 1),
    [en_acc_global] * (N + 1),
    alpha=0.2,
    label=f"Global en={en_acc_global:.3f}"
)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(languages)
ax.set_ylim(radar_min, radar_max)
ax.set_yticks(
    np.linspace(
        radar_min, radar_max,
        int((radar_max - radar_min + 1e-6) // args.scale) + 1
    )
)
ax.set_title("Language Accuracy Comparison", size=16)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

radar_path = os.path.join(root_dir, "radar_comparison.png")
plt.savefig(radar_path, bbox_inches="tight")
plt.close(fig)


# ======================
# 新增：均值条形图 + CI + 表格（按要求修改）
# ======================
folders = []
mean_accs = []
mean_cis = []

# 1) 非 en 任务的均值和 CI
for folder in subfolders_to_plot:
    if folder not in data_dict:
        continue

    # 只统计非 en 语言
    accs = [data_dict[folder][l][0] for l in languages]
    cis = [data_dict[folder][l][1] for l in languages]

    folders.append(folder)
    mean_accs.append(np.mean(accs))      # 还在 0-1 区间
    mean_cis.append(np.mean(cis))        # 同样在 0-1 区间

# 2) en 的 CI：每个 en 源头的 CI 的均值再除以 sqrt(源的个数)
en_cis = []
en_accs = []
en_sources = 0

for folder in subfolders_to_plot:
    csv_path = os.path.join(root_dir, folder, "result.csv")
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    df_en = df[df["language"] == "en"]

    if df_en.empty:
        continue

    # 对 en 的所有 source 统计 accuracy 和 ci_radius
    accs_en = df_en["accuracy"].values
    cis_en = df_en["ci_radius"].values

    if len(accs_en) > 0:
        en_accs.append(np.mean(accs_en))    # 各 task 的 en 平均 acc
        en_cis.append(np.mean(cis_en))      # 各 task 的 en 平均 CI
        en_sources += 1

if en_sources > 0:
    en_acc_mean = np.mean(en_accs)
    en_ci_mean = np.mean(en_cis) / np.sqrt(en_sources)
else:
    en_acc_mean = 0.0
    en_ci_mean = 0.0

folders.append("en_global")
mean_accs.append(en_acc_mean)
mean_cis.append(en_ci_mean)

# 3) 所有分数和 CI 乘 100，变成百分数
mean_accs = np.array(mean_accs) * 100.0
mean_cis = np.array(mean_cis) * 100.0

# 4) 绘制条形图：大小 (4,5)，bar width=0.8，每个棒不同颜色
fig, ax = plt.subplots(figsize=(3, 5))

x = np.arange(len(folders))
width = 0.8

# 几种颜色循环使用
bar_colors = plt.cm.tab10(np.linspace(0, 1, len(folders)))

ax.bar(
    x,
    mean_accs,
    yerr=mean_cis,
    capsize=5,
    width=width,
    color=bar_colors,
    edgecolor='black',
)

ax.set_xticks(x)
ax.set_xticklabels(folders, rotation=45, ha='right')
ax.set_ylim(radar_min * 100.0, radar_max * 100.0)
ax.set_ylabel("Mean Accuracy (%)")
ax.set_title("Mean Accuracy Across Languages")
ax.grid(axis="y", linestyle="--", alpha=0.5)

bar_path = os.path.join(root_dir, "mean_accuracy_bar.png")
plt.tight_layout()
plt.savefig(bar_path, bbox_inches="tight")
plt.close(fig)

# 5) 输出简易表格：task 名称，score (xx.xx +- yy.yy)
rows = []
for name, acc, ci in zip(folders, mean_accs, mean_cis):
    rows.append(f"{name:20s} {acc:6.2f} +- {ci:6.2f}")

table_text = "\n".join(rows)
table_path = os.path.join(root_dir, "mean_accuracy_summary.txt")
with open(table_path, "w", encoding="utf-8") as f:
    f.write(table_text + "\n")

print(f"Radar plot saved to {radar_path}")
print(f"Bar chart saved to {bar_path}")
print("Summary table:")
print(table_text)
print(f"Summary table saved to {table_path}")