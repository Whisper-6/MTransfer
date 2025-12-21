# MTransfer

Investigating Multilingual Knowledge Transfer in LLMs

# Environment

```
pip install requirements.txt
```

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

```
python eval.py \
  --model Qwen2.5-7B-Instruct \
  --num-gpus 8 \
  --batch-size 8 \
  --config default \
  --output-dir output/Qwen2.5-7B-Instruct/default
```

输出包括每个语言的回答（形如 fr.csv）和总分（result.csv）

```
python draw_radar.py \
    --result-csv output/Qwen2.5-7B-Instruct/default/result.csv \
    --output radar/Qwen2.5-7B-Instruct/default.png
```

可绘制雷达图
