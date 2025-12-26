import os
import json
import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_attention_dist(s):
    """
    把 attention_dist 字符串解析成 Python list[dict]
    兼容 JSON 字符串和 Python literal 字符串两种格式。
    """
    s = str(s).strip()
    if not s:
        return []
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


def get_attention_by_layer(csv_path, language, source, fields):
    """
    从 CSV 中取出某个 language + source 的 attention_dist，
    转成 DataFrame，并且只保留你关心的字段（不归一化）。

    返回:
        attn_df: DataFrame，index = layer，columns = fields
        num_layers: 层数
    """
    df = pd.read_csv(csv_path)
    sub = df[(df["language"] == language) & (df["source"] == source)]
    if sub.empty:
        raise ValueError(f"No row found for language={language}, source={source}")
    row = sub.iloc[0]

    attn_list = parse_attention_dist(row["attention_dist"])
    if not attn_list:
        raise ValueError(f"attention_dist is empty or parse failed for {language}, {source}")

    attn_df = pd.DataFrame(attn_list)   # 每一行是一个 layer 的 dict

    # 只保留存在的字段
    valid_fields = [f for f in fields if f in attn_df.columns]
    if len(valid_fields) < len(fields):
        missing = set(fields) - set(valid_fields)
        print(f"Warning: missing fields in attention_dist for {language}: {missing}")

    attn_df = attn_df[valid_fields]
    num_layers = len(attn_df)
    return attn_df, num_layers


def plot_language_averaged_with_individuals_norm(
    csv_path,
    languages,
    source="total",
    fields=("problem", "translation"),
    output_dir="output",
    show=False,
    vertical_layer=None,  # 比如 10，在该 layer 画一条竖线（1-based）
    model_name="model_name",
    rev=True
):
    """
    每层先对各语言的 problem / translation 做归一化：
        p_norm = problem / (problem + translation)
        t_norm = translation / (problem + translation)
    然后在语言维度上求平均，画两条粗线（avg-problem, avg-translation），
    同时把各语言的归一化曲线画成细且透明的背景线。

    仅适用于 fields=("problem", "translation") 这种两字段情况。
    """
    assert len(fields) == 2 and "problem" in fields and "translation" in fields, \
        "当前实现假设 fields 只包含 'problem' 和 'translation' 两个字段"

    os.makedirs(output_dir, exist_ok=True)

    per_lang_matrices = []  # 每个语言的 (num_layers, 2) 矩阵（已按层归一化）
    num_layers_ref = None
    field_order = None

    # 为 problem / translation 各指定一种基础颜色
    base_colors = {
        "problem": "tab:blue",
        "translation": "tab:orange",
    }

    plt.figure(figsize=(10, 6))

    # 先画每种语言的细线（归一化后的），作为背景
    for lang in languages:
        try:
            attn_df, num_layers = get_attention_by_layer(
                csv_path,
                language=lang,
                source=source,
                fields=fields,
            )
        except Exception as e:
            print(f"Skip language {lang} due to error: {e}")
            continue

        if num_layers_ref is None:
            num_layers_ref = num_layers
            field_order = list(attn_df.columns)
        else:
            if num_layers != num_layers_ref:
                print(f"Warning: {lang} has different num_layers ({num_layers}) vs {num_layers_ref}")
            # 防止顺序不一致
            attn_df = attn_df.reindex(columns=field_order)

        # fillna 防止出现 NaN
        attn_df = attn_df.fillna(0.0)

        # === 关键：对每一层做 p/(p+t), t/(p+t) 的归一化 ===
        # 这里假设 field_order = ["problem", "translation"] 或者相反
        p = attn_df[field_order[0]].to_numpy()
        t = attn_df[field_order[1]].to_numpy()
        denom = p + t

        # 避免除以 0，如果某层 p+t=0，就保持 0
        denom_safe = np.where(denom == 0, 1.0, denom)
        p_norm = np.where(denom == 0, 0.0, p / denom_safe)
        t_norm = np.where(denom == 0, 0.0, t / denom_safe)

        norm_mat = np.stack([p_norm, t_norm], axis=1)  # (num_layers, 2)
        per_lang_matrices.append(norm_mat)

        x = range(1, num_layers + 1)

        # 每种语言、每个 field 画一根很细、透明的线
        for field_idx, field in enumerate(field_order):
            y = norm_mat[:, field_idx]
            if rev:
                y = y[::-1]
            plt.plot(
                x,
                y,
                color=base_colors.get(field, "gray"),
                alpha=0.4,          # 透明度高一点，让它当背景
                linewidth=0.7,       # 线细一点
                label=None,          # 不单独加到 legend 里
            )

    if not per_lang_matrices:
        print("No valid language data found.")
        return

    # 堆起来: (num_lang, num_layers, 2)
    per_lang_matrices = np.stack(per_lang_matrices, axis=0)
    # 按语言平均: (num_layers, 2)
    avg_over_langs = per_lang_matrices.mean(axis=0)

    x = range(1, num_layers_ref + 1)

    # 再画两条粗的跨语言平均线
    for field_idx, field in enumerate(field_order):
        y = avg_over_langs[:, field_idx]
        if rev:
            y = y[::-1]
        plt.plot(
            x,
            y,
            color=base_colors.get(field, "gray"),
            linewidth=2.5,
            marker="o",
            label=f"avg-{field}",
        )

    # 可选：画一条竖线标某个 layer（1-based）
    if vertical_layer is not None:
        plt.axvline(
            x=vertical_layer,
            color="red",
            linestyle=":",
            linewidth=1.5,
            label=f"Layer {vertical_layer}",
        )

    plt.xlabel("Layer (1 = shallowest)")
    plt.ylabel("Normalized attention proportion (per language per layer)")
    plt.ylim(0, 1)  # 归一化后在 [0,1] 范围内
    plt.title(
        f"Per-layer normalized (p/(p+t)) attention with cross-language average\n"
        f"fields={fields}, source={source}, model={model_name}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_name = f"avg_over_langs_norm_{source}_{'_'.join(fields)}_{model_name}.png"
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    csv_path = "./output/Qwen2.5-1.5B-Instruct/QxTenAen-2step-attn_eval/result.csv"

    languages = ["ja", "de", "ru", "fr", "bn", "es", "th"]

    plot_language_averaged_with_individuals_norm(
        csv_path=csv_path,
        languages=languages,
        source="total",
        fields=("problem", "translation"),
        output_dir="output/attn_plot/Qwen2.5-1.5B",     # 保存到 ./output
        show=False,
        model_name="Qwen2.5-1.5B",
        vertical_layer=18,
    )