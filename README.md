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
    --config QxAx \
    --num-samples 16
```

参数解释：

| 名称             | 含义           | 默认值                      |
| ---------------- | -------------- | --------------------------- |
| `--model`        | 模型名称       |                             |
| `--config`       | 配置名称       |                             |
| `--num-gpus`     | 推理用 gpu 数  | 所有 gpu                    |
| `--num-samples`  | 每题重复采样数 | 1                           |
| `--output-dir`   | 输出路径       | `output/模型名/配置名`      |
| `--data-dir`     | 数据集路径     | `eval_data/mmath`           |
| `--model-dir`    | 模型父文件夹   | `~/autodl-tmp/local_model/` |
| `--max_tokens`   | 最长输出长度   | 2048                        |
| `--temperature`  | 推理温度       | 0.3                         |

## QxTenAen-2step

```bash
python eval_QxTenAen.py \
    --model Qwen2.5-1.5B-Instruct \
    --num-samples 8
```

## QxTenAen_mask（分层 Attention Masking）

实现带有分层 attention masking 的评估，研究模型前 N 层"看不见"翻译时的解题能力。包含两个原子操作：

### Step 1: 翻译（执行一次）

```bash
python eval_QxTenAen_mask.py \
    --operation translate \
    --model Qwen2.5-7B-Instruct \
    --num-samples 1
```

翻译结果保存在 `tmp/{model}/` 目录，包含：source (唯一标识), original_question, translation, answer

### Step 2: 解题（可测试不同 mask 层数）

```bash
# 测试前 8 层不能看到翻译
python eval_QxTenAen_mask.py \
    --operation solve \
    --model Qwen2.5-7B-Instruct \
    --num-mask-layers 8
```

**说明**：
- solve 操作无需指定 `--num-samples`（自动从翻译结果推断）
- 结果保存在 `output/{model}/QxTenAen_mask/layer_{N}/`
- `--num-mask-layers 0`：所有层可见翻译（相当于 QxTenAen-2step-v2）
- 翻译使用 vLLM（速度快），solve 使用 transformers（支持自定义 attention hook）

### 批量测试示例

```bash
# 翻译一次
python eval_QxTenAen_mask.py --operation translate --model Qwen2.5-7B-Instruct --num-samples 1

# 测试多个 mask 层数（Qwen2.5-7B 有 28 层）
for layers in 0 7 14 21 28; do
    python eval_QxTenAen_mask.py --operation solve --model Qwen2.5-7B-Instruct --num-mask-layers $layers
done
```

## 配置模式

| 配置       | 问题语言 | 指令语言        | 回答语言 | 说明                  |
| ---------- | -------- | --------------- | -------- | --------------------- |
| `QxAx`     | X语言    | X语言           | X语言    | 原生语言能力基线      |
| `QxAen`    | X语言    | 英语            | 英语     | 小语言问题 → 英语思维 |
| `QxTenAen` | X语言    | 英语 (显式翻译) | 英语     | 显式翻译链式推理      |
| `QxTenAx`  | X语言    | 英语 (显式翻译) | X语言    |                       |
| `QenAx`    | 英语     | 英语            | X语言    |                       |

## 输出结果

输出包括总分（result.csv）和每个语言的回答（形如 fr.jsonl）

## 可视化

使用统一的可视化脚本生成所有图表（标准配置 + mask 配置）：

```bash
python vis_radar.py --result-dir output/Qwen2.5-7B-Instruct
```

**可选参数**：

```bash
python vis_radar.py --result-dir output/Qwen2.5-7B-Instruct --min 0.0 --max 1.0 --scale 0.1
```

**生成的图表**：
- 每个配置的单独雷达图（`{config}/radar.png`）
- 标准配置对比图（`radar_comparison.png`、`mean_accuracy_bar.png`）
- Mask 配置对比图（`QxTenAen_mask/radar_comparison.png`、`mean_accuracy_bar.png`）
- 统一对比图（`unified_radar_comparison.png`、`unified_mean_accuracy_bar.png`）

雷达图展示各小语言（bn, de, es, fr, ja, ru, th）的准确率和置信区间，以英语准确率作为统一 baseline（灰色背景）。