# MTransfer

Investigating Multilingual Knowledge Transfer in LLMs

# Environment

conda activate nlp # 在并行智算云上

# NetWork

AutoDL 访问 HuggingFace 需要加速：`source /etc/network_turbo`

# DataSet

> 数据集已经上传 github，无需再下载

选用的数据集包括

- mgsm:   bn de en es fr ja ru sw te th zh
- MSVAMP: bn de en es fr ja ru sw th zh

最终选择的测试语言: bn de es fr ja ru th (sw 表现太差，删去)

运行 `down_datasets.sh` 生成数据集

数据集存储在 `eval_data/mmath/`，命名为 `fr.jsonl` 等

共 bn, de, es, fr, ja, ru, sw, th 八种小语言，再加上 en 一种主要语言

每条信息包括 source, query(英语), m_query(对应语言), answer

# Model

运行 `down_models.sh` 下载模型到 `~/autodl-tmp/local_model/`

默认有 Qwen2.5-{0.5,1.5,3,7}B-Instruct

# Eval

`configs/` 里用 yaml 配置了每种 eval 方案对应的 prompt 格式，`--config` 后填写对应 yaml 的名称

## 基本用法

**推荐方式（batch-size=all，让 vLLM 自动批处理，速度更快）：**

```bash
python eval.py \
  --model Qwen2.5-7B-Instruct \
  --num-gpus 8 \
  --batch-size all \
  --config default \
  --output-dir output/Qwen2.5-7B-Instruct/default
```

**传统方式（手动指定批处理大小）：**

```bash
python eval.py \
  --model Qwen2.5-7B-Instruct \
  --num-gpus 8 \
  --batch-size 8 \
  --config default \
  --output-dir output/Qwen2.5-7B-Instruct/default
```

## 模型路径说明

`--model` 参数支持多种格式：

1. **模型名称**（自动从 `~/autodl-tmp/local_model/` 加载）：

   ```bash
   --model Qwen2.5-7B-Instruct
   ```
2. **绝对路径**（推荐，适用于自定义模型位置）：

   ```bash
   --model /workspace/NLP_PROJECT/Qwen2.5-7B-Instruct
   ```
3. **相对名称 + 自定义目录**：

   ```bash
   --model Qwen2.5-7B-Instruct --model-dir /your/custom/path
   ```

## 批处理选项

`--batch-size` 参数支持两种模式：

- `--batch-size all`：让 vLLM 自动处理所有数据（**推荐**，速度快 1.5-2 倍）
- `--batch-size N`：手动设置批处理大小，N 为数字（如 8、16、32，默认 8，更稳定但较慢）

> 💡 **性能提示**：使用 `--batch-size all` 可充分利用 vLLM 的 Continuous Batching 技术，大幅提升推理速度。

## 配置模式

| 配置         | 问题语言       | 指令语言                       | 回答语言        | 说明                           |
| ------------ | -------------- | ------------------------------ | --------------- | ------------------------------ |
| `default`  | X语言          | X语言                          | X语言           | 原生语言能力基线               |
| `Aen`      | X语言          | 英语                           | 英语            | 小语言问题 → 英语思维         |
| `TenAen`   | X语言          | 英语 (显式翻译)                | 英语            | 显式翻译链式推理               |
| `TenAx`    | X语言          | 英语 (显式翻译)                | X语言           | 英语思维 → 小语言输出         |
| `QenAx`    | **英语** | **X语言**                | **X语言** | 英语问题 → 各语言指令         |
| `QenAxPen` | **英语** | **英语** (要求用X语言答) | **X语言** | 英语指令明确要求用指定语言回答 |

## 输出结果

输出包括每个语言的回答（形如 fr.csv）和总分（result.csv）

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
