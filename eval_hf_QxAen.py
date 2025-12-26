import os
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from utils import last_number_from_text, build_chat_prompt
from multiprocessing import Process, Manager
import csv

# ----------------------
# 命令行参数
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--model-dir", type=str, default="~/autodl-tmp/local_model/")
parser.add_argument("--translation-dir", type=str, default="./translation/")
parser.add_argument("--out-dir", type=str, default="./hf_output/")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
parser.add_argument("--do-sample", action="store_true", default=True, help="是否开启随机采样生成")
parser.add_argument("--temperature", type=float, default=0.3, help="生成温度")
args = parser.parse_args()

# ----------------------
# 固定设置
# ----------------------
languages = ["bn", "de", "es", "fr", "ja", "ru", "th", "en"]

SOLVE_PROMPT = (
    "Problem: {question}\n\n"
    "Solve the above problem in English, and enclose the final number at the end of the response in $\\boxed{{}}$."
)

model_path = os.path.expanduser(os.path.join(args.model_dir, args.model))
translation_path = os.path.expanduser(os.path.join(args.translation_dir, args.model))
output_path = os.path.expanduser(os.path.join(args.out_dir, args.model, "QxAen"))
os.makedirs(output_path, exist_ok=True)

# ----------------------
# 读取翻译文件，安全过滤缺字段
# ----------------------
def load_all_problems(languages, translation_path):
    all_problems = []
    for lang in languages:
        file_path = os.path.join(translation_path, f"{lang}.jsonl")
        if not os.path.exists(file_path):
            print(f"[WARN] {file_path} 不存在，跳过")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    if "original_question" not in data or "translation" not in data or "answer" not in data:
                        print(f"[WARN] {file_path} 第 {line_num} 行缺字段，跳过")
                        continue
                    data["language"] = lang
                    all_problems.append(data)
                except json.JSONDecodeError:
                    print(f"[WARN] {file_path} 第 {line_num} 行 JSON 错误，跳过")
    return all_problems

all_problems = load_all_problems(languages, translation_path)
print(f"[INFO] 共加载 {len(all_problems)} 个问题")

# ----------------------
# 异步批量生成函数
# ----------------------
@torch.inference_mode()
def generate_batch(model, tokenizer, batch, device, temperature=0.3, do_sample=False):
    results = []
    user_prompts = [build_chat_prompt(tokenizer, x["prompt"]) for x in batch]

    # 编码
    inputs = tokenizer(
        user_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    gen_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=do_sample,
        temperature=temperature,
        top_p=0.9,
        top_k=50
    )

    outputs = model.generate(**inputs, generation_config=gen_config)
    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    for item, response in zip(batch, decoded):
        predicted_number = last_number_from_text(response)
        correct = predicted_number == item.get("answer")
        results.append({
            "language": item["language"],
            "source": item["source"],
            "prompt": item["prompt"],
            "response": response,
            "predicted": predicted_number,
            "answer": item.get("answer"),
            "correct": correct
        })
    return results

# ----------------------
# 多 GPU 异步推理（每 GPU 自己显示进度条）
# ----------------------
def run_on_gpu(gpu_id, problems, batch_size, temperature, do_sample, return_list):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] GPU-{gpu_id} 使用 {device} 加载模型")

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    # GPU 内部生成 prompt
    for p in problems:
        p["prompt"] = SOLVE_PROMPT.format(
            question=p["original_question"],
            translation=p["translation"]
        )

    # 按 prompt 长度排序
    problems.sort(key=lambda x: len(x["prompt"]))

    dataloader = DataLoader(
        problems,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x  # 保持 list of dict
    )

    for batch in tqdm(dataloader, desc=f"GPU-{gpu_id}", position=gpu_id):
        batch_results = generate_batch(model, tokenizer, batch, device, temperature, do_sample)
        return_list.extend(batch_results)

# ----------------------
# 主流程
# ----------------------
manager = Manager()
return_list = manager.list()

# 数据按 GPU 均分
gpu_splits = [[] for _ in range(args.num_gpus)]
for i, p in enumerate(all_problems):
    gpu_splits[i % args.num_gpus].append(p)

# 启动进程
processes = []
for gpu_id in range(args.num_gpus):
    p = Process(target=run_on_gpu, args=(
        gpu_id, gpu_splits[gpu_id], args.batch_size,
        args.temperature, args.do_sample,
        return_list
    ))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

# ----------------------
# 保存结果
# ----------------------
results = list(return_list)
results_by_lang = {lang: [] for lang in languages}
for r in results:
    lang = r.pop("language")
    results_by_lang[lang].append(r)

# 输出 jsonl
for lang, items in results_by_lang.items():
    out_file = os.path.join(output_path, f"{lang}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 输出 csv 和正确率
csv_file = os.path.join(output_path, "result.csv")
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["language", "accuracy"])
    writer.writeheader()
    for lang, items in results_by_lang.items():
        acc = sum(x["correct"] for x in items) / len(items) if items else 0
        print(f"{lang} 正确率: {acc:.2%}")
        writer.writerow({"language": lang, "accuracy": acc})

print(f"[INFO] 所有结果已保存到 {output_path}")
