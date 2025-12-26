#!/usr/bin/env python3
"""
Unified visualization script for all experiment configurations.
Generates:
1. Individual radar plots for each configuration
2. Comparison radar plot for standard configurations
3. Comparison radar plot for mask configurations (if exists)
4. Unified comparison plot (standard + mask)
5. Bar charts for mean accuracies
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Parse arguments
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--result-dir",
    type=str,
    required=True,
    help="Root directory containing experiment results"
)
parser.add_argument("--min", type=float, default=0.0, help="Radar plot minimum value")
parser.add_argument("--max", type=float, default=1.0, help="Radar plot maximum value")
parser.add_argument("--scale", type=float, default=0.1, help="Radar plot tick scale")
args = parser.parse_args()

root_dir = args.result_dir
radar_min, radar_max = args.min, args.max

print("=" * 60)
print("🎨 Unified Radar Visualization")
print("=" * 60)
print(f"📁 Result directory: {root_dir}")
print(f"📊 Radar range: [{radar_min}, {radar_max}], scale: {args.scale}\n")

# ----------------------
# Get en baseline (unified)
# ----------------------
def get_en_baseline(root_dir):
    """Get en baseline accuracy from standard configurations."""
    baseline_candidates = ["QxTenAen", "QxTenAen-2step", "QxAen", "QxAx"]
    for candidate in baseline_candidates:
        csv_path = os.path.join(root_dir, candidate, "result.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            en_row = df[(df["language"] == "en") & (df["source"] == "total")]
            if not en_row.empty:
                en_acc = en_row["accuracy"].values[0]
                print(f"✓ Using en baseline from {candidate}: {en_acc:.4f}")
                return en_acc
    print("⚠ Warning: No en baseline found in standard configs, using 0.0")
    return 0.0

en_baseline = get_en_baseline(root_dir)

# ----------------------
# Collect standard configurations
# ----------------------
print("\n" + "=" * 60)
print("📊 [1/4] Collecting standard configurations...")
print("=" * 60)

standard_configs = ["QxAx", "QxAen", "QxTenAen", "QxTenAen-2step", "QxTenAen-2step-v2"]
standard_data = {}
all_languages = set()

for config in standard_configs:
    csv_path = os.path.join(root_dir, config, "result.csv")
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    df_total = df[df["source"] == "total"]

    lang_acc = {}
    for _, row in df_total.iterrows():
        lang = row["language"]
        acc = row["accuracy"]
        ci = row["ci_radius"]
        total = row["total"]
        correct = row["correct"]
        lang_acc[lang] = (acc, ci, total, correct)
        if lang != "en":
            all_languages.add(lang)

    standard_data[config] = lang_acc
    print(f"  ✓ {config}: {len(lang_acc)} languages")

# ----------------------
# Collect mask configurations
# ----------------------
print("\n" + "=" * 60)
print("📊 [2/4] Collecting mask configurations...")
print("=" * 60)

mask_dir = os.path.join(root_dir, "QxTenAen_mask")
mask_data = {}
layer_folders = []

if os.path.exists(mask_dir):
    layer_folders = sorted([
        f for f in os.listdir(mask_dir)
        if os.path.isdir(os.path.join(mask_dir, f)) and f.startswith("layer_")
    ])

    for folder in layer_folders:
        csv_path = os.path.join(mask_dir, folder, "result.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df_total = df[df["source"] == "total"]

        lang_acc = {}
        for _, row in df_total.iterrows():
            lang = row["language"]
            acc = row["accuracy"]
            ci = row["ci_radius"]
            total = row["total"]
            correct = row["correct"]
            lang_acc[lang] = (acc, ci, total, correct)
            if lang != "en":
                all_languages.add(lang)

        mask_data[folder] = lang_acc
        print(f"  ✓ {folder}: {len(lang_acc)} languages")
else:
    print("  ℹ No QxTenAen_mask directory found")

# ----------------------
# Generate individual radar plots
# ----------------------
print("\n" + "=" * 60)
print("🎨 [3/4] Generating individual radar plots...")
print("=" * 60)

def generate_individual_radar(config_name, lang_acc, output_path, is_mask=False):
    """Generate individual radar plot for a configuration."""
    # Filter out English
    lang_data = {k: v for k, v in lang_acc.items() if k != "en"}
    if not lang_data:
        return False
    
    languages = sorted(lang_data.keys())
    accuracies = [lang_data[l][0] for l in languages]
    ci_radii = [lang_data[l][1] for l in languages]  # Use per-language CI for individual radar
    
    N = len(languages)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    values = accuracies + accuracies[:1]
    ci_upper = [min(v + c, 1.0) for v, c in zip(accuracies, ci_radii)]
    ci_lower = [max(v - c, 0.0) for v, c in zip(accuracies, ci_radii)]
    ci_upper.append(ci_upper[0])
    ci_lower.append(ci_lower[0])
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # en baseline background
    if en_baseline > 0:
        en_polygon = [en_baseline] * (N + 1)
        ax.fill(
            np.linspace(0, 2 * np.pi, N + 1),
            en_polygon,
            alpha=0.2,
            color='gray',
            label=f"en baseline={en_baseline:.3f}"
        )
    
    # Language accuracies
    ax.plot(angles, values, 'o-', linewidth=2, color='steelblue', label='Accuracy')
    ax.fill_between(angles, ci_lower, ci_upper, alpha=0.3, color='steelblue', label='CI')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(languages, fontsize=10)
    ax.set_ylim(radar_min, radar_max)
    ax.set_yticks(np.linspace(radar_min, radar_max, int((radar_max-radar_min)/args.scale) + 1))
    ax.set_title(f"Language Accuracy: {config_name}", size=14, pad=20)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return True

# Generate for standard configs
for config, lang_acc in standard_data.items():
    output_path = os.path.join(root_dir, config, "radar.png")
    if generate_individual_radar(config, lang_acc, output_path):
        print(f"  ✓ {config}")

# Generate for mask configs
for folder, lang_acc in mask_data.items():
    output_path = os.path.join(mask_dir, folder, "radar.png")
    if generate_individual_radar(folder, lang_acc, output_path, is_mask=True):
        print(f"  ✓ {folder}")

# ----------------------
# Generate comparison plots
# ----------------------
print("\n" + "=" * 60)
print("🎨 [4/4] Generating comparison plots...")
print("=" * 60)

languages = sorted(list(all_languages))
N = len(languages)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# ----------------------
# 1. Standard configs comparison
# ----------------------
if standard_data:
    print("\n  📊 Standard configurations comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (config, lang_acc) in enumerate(standard_data.items()):
        values = []
        ci_upper = []
        ci_lower = []
        
        for lang in languages:
            if lang in lang_acc:
                acc, ci, _, _ = lang_acc[lang]
                values.append(acc)
                ci_upper.append(min(acc + ci, 1.0))
                ci_lower.append(max(acc - ci, 0.0))
            else:
                values.append(0.0)
                ci_upper.append(0.0)
                ci_lower.append(0.0)
        
        values += values[:1]
        ci_upper += ci_upper[:1]
        ci_lower += ci_lower[:1]
        
        color = colors[i % len(colors)]
        ax.plot(angles, values, 'o-', linewidth=2, label=config, color=color)
        ax.fill_between(angles, ci_lower, ci_upper, alpha=0.1, color=color)
    
    # en baseline
    if en_baseline > 0:
        ax.fill(
            np.linspace(0, 2 * np.pi, N + 1),
            [en_baseline] * (N + 1),
            alpha=0.15,
            color='gray',
            label=f"en baseline={en_baseline:.3f}"
        )
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(languages, fontsize=12)
    ax.set_ylim(radar_min, radar_max)
    ax.set_yticks(np.linspace(radar_min, radar_max, int((radar_max-radar_min)/args.scale) + 1))
    ax.set_title("Standard Configurations Comparison", size=16, pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.1), fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(root_dir, "radar_comparison.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    ✓ {output_path}")
    
    # Mean accuracy bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    config_names = list(standard_data.keys())
    mean_accs = []
    mean_cis = []
    
    for config in config_names:
        lang_acc = standard_data[config]
        # Sum up total and correct across all non-English languages
        total_sum = sum(lang_acc[lang][2] for lang in languages if lang in lang_acc)
        correct_sum = sum(lang_acc[lang][3] for lang in languages if lang in lang_acc)
        if total_sum > 0:
            acc = correct_sum / total_sum
            ci = 1.96 * np.sqrt(acc * (1 - acc) / total_sum)
        else:
            acc, ci = 0.0, 0.0
        mean_accs.append(acc)
        mean_cis.append(ci)
    
    bars = ax.bar(range(len(config_names)), mean_accs, yerr=mean_cis, capsize=5,
                   color='steelblue', alpha=0.7, edgecolor='black')
    
    if en_baseline > 0:
        ax.axhline(y=en_baseline, color='r', linestyle='--', linewidth=2,
                   label=f'en baseline={en_baseline:.3f}')
    
    for i, (bar, acc) in enumerate(zip(bars, mean_accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + mean_cis[i] + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Mean Accuracy (non-English)", fontsize=12)
    ax.set_title("Standard Configurations: Mean Accuracy", fontsize=14, pad=15)
    ax.set_ylim(radar_min, radar_max)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(root_dir, "mean_accuracy_bar.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    ✓ {output_path}")

# ----------------------
# 2. Mask configs comparison
# ----------------------
if mask_data:
    print("\n  📊 Mask configurations comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(mask_data)))
    
    for i, (folder, lang_acc) in enumerate(mask_data.items()):
        values = []
        ci_upper = []
        ci_lower = []
        
        for lang in languages:
            if lang in lang_acc:
                acc, ci, _, _ = lang_acc[lang]
                values.append(acc)
                ci_upper.append(min(acc + ci, 1.0))
                ci_lower.append(max(acc - ci, 0.0))
            else:
                values.append(0.0)
                ci_upper.append(0.0)
                ci_lower.append(0.0)
        
        values += values[:1]
        ci_upper += ci_upper[:1]
        ci_lower += ci_lower[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=folder, color=colors_map[i])
        ax.fill_between(angles, ci_lower, ci_upper, alpha=0.15, color=colors_map[i])
    
    # en baseline
    if en_baseline > 0:
        ax.fill(
            np.linspace(0, 2 * np.pi, N + 1),
            [en_baseline] * (N + 1),
            alpha=0.15,
            color='gray',
            label=f"en baseline={en_baseline:.3f}"
        )
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(languages, fontsize=12)
    ax.set_ylim(radar_min, radar_max)
    ax.set_yticks(np.linspace(radar_min, radar_max, int((radar_max-radar_min)/args.scale) + 1))
    ax.set_title("Mask Configurations: Layer-wise Comparison", size=16, pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.1), fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(mask_dir, "radar_comparison.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    ✓ {output_path}")
    
    # Mean accuracy bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by layer number
    sorted_items = sorted(mask_data.items(), 
                         key=lambda x: int(x[0].split("_")[1]) if "_" in x[0] else 0)
    folder_names = [f for f, _ in sorted_items]
    mean_accs = []
    mean_cis = []
    
    for folder, lang_acc in sorted_items:
        # Sum up total and correct across all non-English languages
        total_sum = sum(lang_acc[lang][2] for lang in languages if lang in lang_acc)
        correct_sum = sum(lang_acc[lang][3] for lang in languages if lang in lang_acc)
        if total_sum > 0:
            acc = correct_sum / total_sum
            ci = 1.96 * np.sqrt(acc * (1 - acc) / total_sum)
        else:
            acc, ci = 0.0, 0.0
        mean_accs.append(acc)
        mean_cis.append(ci)
    
    bars = ax.bar(range(len(folder_names)), mean_accs, yerr=mean_cis, capsize=5,
                   color='coral', alpha=0.7, edgecolor='black')
    
    if en_baseline > 0:
        ax.axhline(y=en_baseline, color='r', linestyle='--', linewidth=2,
                   label=f'en baseline={en_baseline:.3f}')
    
    for i, (bar, acc) in enumerate(zip(bars, mean_accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + mean_cis[i] + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(range(len(folder_names)))
    ax.set_xticklabels(folder_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Mean Accuracy (non-English)", fontsize=12)
    ax.set_title("Mask Configurations: Mean Accuracy by Layer", fontsize=14, pad=15)
    ax.set_ylim(radar_min, radar_max)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(mask_dir, "mean_accuracy_bar.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    ✓ {output_path}")

# ----------------------
# 3. Unified comparison (standard + mask)
# ----------------------
if standard_data or mask_data:
    print("\n  📊 Unified comparison (Standard + Mask)...")
    
    # Combine all data
    unified_data = {}
    unified_data.update(standard_data)
    for folder, lang_acc in mask_data.items():
        display_name = f"mask_{folder.replace('layer_', 'L')}"
        unified_data[display_name] = lang_acc
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 
              'brown', 'pink', 'gray', 'olive', 'navy', 'teal', 'maroon']
    
    for i, (config, lang_acc) in enumerate(unified_data.items()):
        values = []
        ci_upper = []
        ci_lower = []
        
        for lang in languages:
            if lang in lang_acc:
                acc, ci, _, _ = lang_acc[lang]
                values.append(acc)
                ci_upper.append(min(acc + ci, 1.0))
                ci_lower.append(max(acc - ci, 0.0))
            else:
                values.append(0.0)
                ci_upper.append(0.0)
                ci_lower.append(0.0)
        
        values += values[:1]
        ci_upper += ci_upper[:1]
        ci_lower += ci_lower[:1]
        
        color = colors[i % len(colors)]
        line_style = '--' if config.startswith('mask_') else '-'
        ax.plot(angles, values, 'o', linewidth=2, label=config, color=color, 
                linestyle=line_style, markersize=4)
        ax.fill_between(angles, ci_lower, ci_upper, alpha=0.08, color=color)
    
    # en baseline
    if en_baseline > 0:
        ax.fill(
            np.linspace(0, 2 * np.pi, N + 1),
            [en_baseline] * (N + 1),
            alpha=0.15,
            color='black',
            label=f"en baseline={en_baseline:.3f}"
        )
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(languages, fontsize=12)
    ax.set_ylim(radar_min, radar_max)
    ax.set_yticks(np.linspace(radar_min, radar_max, int((radar_max-radar_min)/args.scale) + 1))
    ax.set_title("Unified Comparison: All Configurations", size=16, pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.1), fontsize=9, ncol=1)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(root_dir, "unified_radar_comparison.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    ✓ {output_path}")
    
    # Unified bar chart
    fig, ax = plt.subplots(figsize=(max(14, len(unified_data) * 0.9), 6))
    
    config_names = list(unified_data.keys())
    mean_accs = []
    mean_cis = []
    bar_colors = []
    
    for config in config_names:
        lang_acc = unified_data[config]
        # Sum up total and correct across all non-English languages
        total_sum = sum(lang_acc[lang][2] for lang in languages if lang in lang_acc)
        correct_sum = sum(lang_acc[lang][3] for lang in languages if lang in lang_acc)
        if total_sum > 0:
            acc = correct_sum / total_sum
            ci = 1.96 * np.sqrt(acc * (1 - acc) / total_sum)
        else:
            acc, ci = 0.0, 0.0
        mean_accs.append(acc)
        mean_cis.append(ci)
        bar_colors.append('coral' if config.startswith('mask_') else 'steelblue')
    
    bars = ax.bar(range(len(config_names)), mean_accs, yerr=mean_cis, capsize=5,
                   color=bar_colors, alpha=0.7, edgecolor='black')
    
    if en_baseline > 0:
        ax.axhline(y=en_baseline, color='r', linestyle='--', linewidth=2,
                   label=f'en baseline={en_baseline:.3f}')
    
    for i, (bar, acc) in enumerate(zip(bars, mean_accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + mean_cis[i] + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Mean Accuracy (non-English)", fontsize=12)
    ax.set_title("Unified Comparison: Mean Accuracy", fontsize=14, pad=15)
    ax.set_ylim(radar_min, radar_max)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Legend for bar colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Standard Config'),
        Patch(facecolor='coral', alpha=0.7, label='Mask Config')
    ]
    if en_baseline > 0:
        from matplotlib.lines import Line2D
        legend_elements.append(Line2D([0], [0], color='r', linestyle='--', 
                                     linewidth=2, label=f'en baseline={en_baseline:.3f}'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(root_dir, "unified_mean_accuracy_bar.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    ✓ {output_path}")

# ----------------------
# Summary
# ----------------------
print("\n" + "=" * 60)
print("✅ All visualizations completed!")
print("=" * 60)
print(f"📁 Output directory: {root_dir}")
print(f"📊 Standard configs: {len(standard_data)}")
print(f"📊 Mask configs: {len(mask_data)}")
print(f"📈 en baseline: {en_baseline:.4f}")
print("=" * 60)

