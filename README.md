# MTransfer

Investigating Multilingual Knowledge Transfer in LLMs

# Environment

conda activate nlp # 在并行智算云上

# NetWork

AutoDL 访问 HuggingFace 需要加速：`source /etc/network_turbo`
并行智算云加速：`source /workspace/v2ray/network_setup.sh`

# DataSet

> 数据集已经上传 github，无需再下载

选用的数据集包括

- mgsm:   bn de en es fr ja ru sw te th zh
- MSVAMP: bn de en es fr ja ru sw th zh

最终选择的测试语言: bn de es fr ja ru th (sw 表现太差，删去)

运行 `down_datasets.sh` 生成数据集

数据集存储在 `eval_data/mmath/`，命名为 `fr.jsonl` 等

共 bn, de, es, fr, ja, ru, th 七种小语言，再加上 en 一种主要语言

每条信息包括 source, query(英语), m_query(对应语言), answer

# Model

运行 `down_models.sh` 下载模型到 `~/autodl-tmp/local_model/`

默认有 Qwen2.5-{0.5,1.5,3,7}B-Instruct

# Eval

`configs/` 里用 yaml 配置了每种 eval 方案对应的 prompt 格式，`--config` 后填写对应 yaml 的名称

## 基本用法

```bash
python eval.py \
    --model Qwen2.5-7B-Instruct \
    --config QxAen \
    --num-samples 8
```

参数解释：

| 名称             | 含义           | 默认值                      |
| ---------------- | -------------- | --------------------------- |
| `--model`        | 模型名称       |                             |
| `--config`       | 配置名称       |                             |
| `--num-gpus`     | 推理用 gpu 数  | 所有 gpu                    |
| `--num-samples`  | 每题重复采样数 | 1                           |
| `--keep-correct` | 正确回答保留数 | 32                          |
| `--keep-wrong`   | 错误回答保留数 | 32                          |
| `--output-dir`   | 输出路径       | `output/模型名/配置名`      |
| `--data-dir`     | 数据集路径     | `eval_data/mmath`           |
| `--model-dir`    | 模型父文件夹   | `~/autodl-tmp/local_model/` |
| `--max_tokens`   | 最长输出长度   | 2048                        |
| `--temperature`  | 推理温度       | 0.3                         |


## 配置模式

| 配置       | 问题语言 | 指令语言        | 回答语言 | 说明                  |
| ---------- | -------- | --------------- | -------- | --------------------- |
| `QxAx`     | X语言    | X语言           | X语言    | 原生语言能力基线      |
| `QxAen`    | X语言    | 英语            | 英语     | 小语言问题 → 英语思维 |
| `QxTenAen` | X语言    | 英语 (显式翻译) | 英语     | 显式翻译链式推理      |
| `QxTenAx`  | X语言    | 英语 (显式翻译) | X语言    |                       |
| `QenAx`    | 英语     | 英语            | X语言    |                       |

## 输出结果

输出包括总分（result.csv）和每个语言回答的抽样（形如 fr_samples.jsonl）

## 可视化

绘制雷达图展示各语言的准确率：

```bash
# 示例1: default 配置
python draw_radar.py \
    --result-csv output/Qwen2.5-7B-Instruct/default/result.csv \
    --output radar/Qwen2.5-7B-Instruct/default.png

# 示例2: QenAx 配置（英文问题 + 多语言回答）
python draw_radar.py \
    --result-csv output/Qwen2.5-7B-Instruct/QenAx/result.csv \
    --output radar/Qwen2.5-7B-Instruct/QenAx.png
```

雷达图将展示各小语言（bn, de, es, fr, ja, ru, th）的准确率，以英语准确率作为参考背景。