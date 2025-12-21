#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制多语言数学推理准确率雷达图
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw radar chart for multilingual math-reasoning accuracy."
    )
    parser.add_argument(
        "--result-csv",
        required=True,
        help="Path to the input CSV file (must contain columns: language, source, accuracy).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the PNG image will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------ 
    # 读取数据
    # ------------------------------
    df = pd.read_csv(args.result_csv)

    # 提取非 en 语言 total 正确率，按字典序
    langs = sorted([l for l in df["language"].unique() if l != "en"])

    values = [
        df[(df["language"] == l) & (df["source"] == "total")]["accuracy"].values[0]
        for l in langs
    ]

    # en 的 total 正确率，用作背景
    en_value = df[(df["language"] == "en") & (df["source"] == "total")][
        "accuracy"
    ].values[0]

    # ------------------------------ 
    # 角度设置
    # ------------------------------
    N = len(langs)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 封闭曲线

    values += values[:1]
    en_polygon = [en_value] * N
    en_polygon += en_polygon[:1]

    # ------------------------------ 
    # 绘图
    # ------------------------------
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)  # 从顶部开始
    ax.set_theta_direction(-1)  # 顺时针

    # 绘制 en 背景多边形
    ax.fill(angles, en_polygon, color="lightblue", alpha=0.3, label="en")

    # 绘制非 en 模型数据曲线
    ax.plot(angles, values, color="red", linewidth=2, marker="o", label="model")
    ax.fill(angles, values, color="red", alpha=0.2)

    # 设置语言标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(langs)

    # 半径刻度（从 0.5 开始）
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks(np.linspace(0.5, 1.0, 5))
    ax.set_yticklabels([f"{v:.2f}" for v in np.linspace(0.5, 1.0, 5)])

    ax.set_title("Model Math Reasoning Accuracy per Language", pad=20)
    ax.legend(loc="upper right")

    # ------------------------------ 
    # 保存到本地图片
    # ------------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Radar chart saved to: {args.output}")


if __name__ == "__main__":
    main()