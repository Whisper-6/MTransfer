import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# 配置
# ----------------------
root_dir = "."  # 大文件夹路径
radar_min, radar_max = 0.5, 0.9  # 雷达图刻度范围
# 只比较指定子文件夹
subfolders_to_plot = ["QxAen", "QxTenAen", "QxAx"]

# ----------------------
# 收集各子文件夹数据
# ----------------------
data_dict = {}  # folder -> dict(language -> (acc, ci))
all_languages = set()

for folder in subfolders_to_plot:
    csv_path = os.path.join(root_dir, folder, "result.csv")
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    df_total = df[df["source"]=="total"]
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
    print("No data found for the specified subfolders.")
    exit(0)

# ----------------------
# 确定雷达图语言维度（去掉 en，按字典序）
# ----------------------
languages = sorted(list(all_languages))
N = len(languages)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# ----------------------
# 绘图
# ----------------------
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))

colors = ['r', 'g', 'b', 'orange', 'purple']
for i, folder in enumerate(subfolders_to_plot):
    if folder not in data_dict:
        continue
    lang_acc = data_dict[folder]
    values = [lang_acc[lang][0] if lang in lang_acc else radar_min for lang in languages]
    values += values[:1]

    ci_upper = [min(lang_acc[lang][0]+lang_acc[lang][1], 1.0) if lang in lang_acc else radar_min for lang in languages]
    ci_lower = [max(lang_acc[lang][0]-lang_acc[lang][1], 0.0) if lang in lang_acc else radar_min for lang in languages]
    ci_upper += ci_upper[:1]
    ci_lower += ci_lower[:1]

    # 绘制折线
    ax.plot(angles, values, 'o-', linewidth=2, color=colors[i % len(colors)], label=folder)
    ax.fill_between(angles, ci_lower, ci_upper, color=colors[i % len(colors)], alpha=0.2)

# ----------------------
# 添加 en 背景线
# ----------------------
# 计算全局 en accuracy
en_correct_total = 0
en_trials_total = 0
for folder in subfolders_to_plot:
    csv_path = os.path.join(root_dir, folder, "result.csv")
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    en_row = df[(df["language"]=="en") & (df["source"]=="total")]
    if not en_row.empty:
        en_correct_total += en_row["correct"].values[0]
        en_trials_total += en_row["total"].values[0]
if en_trials_total > 0:
    en_acc_global = en_correct_total / en_trials_total
else:
    en_acc_global = 0.0

en_polygon = [en_acc_global]*N
en_polygon += en_polygon[:1]
ax.fill(np.linspace(0, 2*np.pi, N+1), en_polygon, color='lightblue', alpha=0.2, label=f"Global en={en_acc_global:.3f}")

# ----------------------
# 雷达图设置
# ----------------------
ax.set_xticks(angles[:-1])
ax.set_xticklabels(languages)
ax.set_ylim(radar_min, radar_max)
ax.set_yticks(np.linspace(radar_min, radar_max, 5))
ax.set_title("Language Accuracy Comparison", size=16)
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# ----------------------
# 保存图片
# ----------------------
out_path = os.path.join(root_dir, "radar_comparison.png")
plt.savefig(out_path, bbox_inches='tight')
plt.close(fig)

print(f"Radar plot saved to {out_path}")
