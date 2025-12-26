
import os
import json
import ast
import pandas as pd
import matplotlib.pyplot as plt

def parse_attention_dist(s):
    s = s.strip()
    if not s:
        return []
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)

def plot_attention_from_csv(
    csv_path,
    language="ja",
    source="mgsm",
    fields=("problem", "translation"),
    output_dir="output",
    filename=None,
    show=False,
    model_name="model_name",
    rev=True
):
    df = pd.read_csv(csv_path)
    sub = df[(df["language"] == language) & (df["source"] == source)]
    if sub.empty:
        raise ValueError(f"No row found for language={language}, source={source}")
    row = sub.iloc[0]

    attn_list = parse_attention_dist(row["attention_dist"])
    if not attn_list:
        raise ValueError("attention_dist is empty or parse failed")

    attn_df = pd.DataFrame(attn_list)
    num_layers = len(attn_df)

    plt.figure(figsize=(10, 6))
    x = range(num_layers)

    # 只用 attn_df 中实际存在的字段
    valid_fields = [f for f in fields if f in attn_df.columns]
    if len(valid_fields) < len(fields):
        missing = set(fields) - set(valid_fields)
        print(f"Warning: missing fields in attention_dist: {missing}")

    df = attn_df[valid_fields] # .div(layer_sum_replaced, axis=0)

    for field in valid_fields:
        re = df[field]
        if rev:
            re = df[field][::-1]
        plt.plot(x, re, marker="o", label=field)

    layer_to_mark = 16  # 想标的层号（0-based）

    plt.axvline(
        x=layer_to_mark,
        color='red',      # 线的颜色
        linestyle='--',   # 虚线
        linewidth=1.5,    # 线宽
        label=f'Layer {layer_to_mark}'
    )
    plt.xlabel("Layer")
    plt.ylabel("Average attention")
    plt.title(f"Attention distribution by layer\nlanguage={language}, source={source}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # === 保存到 output/ ===
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"attn_{language}_{source}_{'_'.join(fields)}_{model_name}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    
    csv_path = "./output/Qwen2.5-1.5B-Instruct/QxTenAen-2step-attn_eval/result.csv"  # 替换成你的 CSV 路径
    lang = ["ja", "de", "ru", "fr", "bn", "es", "th"]
    for l in lang: 
        plot_attention_from_csv(
            csv_path,
            language=l,
            source="total",
            fields=("problem", "translation"),
            output_dir="output/attn_plot/Qwen2.5-1.5B",     # 保存到 ./output
            show=False,               # 不弹窗，只保存
            model_name="Qwen2.5-1.5B"
        )