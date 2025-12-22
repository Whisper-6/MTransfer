import os
import json
import re
import csv
import argparse
import math
import multiprocessing as mp
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
import yaml
from transformers import AutoTokenizer

# ----------------------
# 数字转换函数
# ----------------------
BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"

def convert_to_arabic_digits(s):
    return ''.join(str(BENGALI_DIGITS.index(c)) if c in BENGALI_DIGITS else c for c in s)

def last_number_from_text(text):
    text = convert_to_arabic_digits(text)
    nums = re.findall(r"[-+]?\d+", text)
    return int(nums[-1]) if nums else None

# ----------------------
# 单进程函数（每 GPU 一个常驻进程）
# ----------------------
def worker_process(rank, world_size, args, rank_lang_data, config, return_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)

    # 处理模型路径：如果是绝对路径直接使用，否则拼接默认路径
    if os.path.isabs(args.model):
        model_path = args.model
    else:
        model_path = os.path.join(args.model_dir, args.model)

    # 初始化模型
    llm = LLM(
        model=model_path,
        dtype="half",
    )

    # 初始化 tokenizer（用于 chat template）
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    worker_results = {}

    for lang, data_slice in rank_lang_data.items():
        total = 0
        correct = 0
        results = []

        keyword_stats = {kw: {"total": 0, "correct": 0} for kw in args.sources}
        yaml_prompt = config["languages"][lang]["prompt"]
        
        # 检查配置文件是否指定使用英文问题
        question_field = config.get("question_field", "m_query")

        # ---------- 优化方案：一次性构造所有 prompts ----------
        if args.use_vllm_batch_all:
            # 方案1: 让 vLLM 一次性处理所有数据（推荐）
            all_prompts = []
            for ex in data_slice:
                user_content = yaml_prompt.format(question=ex[question_field])
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_content},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                all_prompts.append(prompt)
            
            # 一次性生成所有结果
            outputs = llm.generate(
                all_prompts,
                SamplingParams(
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    stop=["<|user|>"],
                ),
                use_tqdm=True,
            )
            
            # 处理结果
            for ex, out in zip(data_slice, outputs):
                text = out.outputs[0].text.strip()
                pred = last_number_from_text(text)
                try:
                    ans = int(float(convert_to_arabic_digits(str(ex["answer"]))))
                except:
                    ans = None
                is_correct = pred == ans
                total += 1
                correct += int(is_correct)
                for kw in args.sources:
                    if kw in ex["source"]:
                        keyword_stats[kw]["total"] += 1
                        keyword_stats[kw]["correct"] += int(is_correct)
                results.append({
                    "source": ex["source"],
                    "full_response": text,
                    "pred_number": pred if pred is not None else "",
                    "answer": ans if ans is not None else "",
                })
        else:
            # 方案2: 手动分批（原方案，兼容性好）
            pbar = tqdm(
                total=len(data_slice),
                desc=f"[GPU {rank}] {lang}",
                position=rank,
                leave=False,
            )

            for i in range(0, len(data_slice), args.batch_size):
                batch = data_slice[i:i + args.batch_size]

                # ---------- 构造 chat-template prompt ----------
                prompts = []
                for ex in batch:
                    user_content = yaml_prompt.format(question=ex[question_field])

                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {
                            "role": "user",
                            "content": user_content,
                        },
                    ]

                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    prompts.append(prompt)
                # -----------------------------------------------

                outputs = llm.generate(
                    prompts,
                    SamplingParams(
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        stop=["<|user|>"],  # 保险：防止继续对话
                    ),
                    use_tqdm=False,
                )

                for ex, out in zip(batch, outputs):
                    text = out.outputs[0].text.strip()
                    pred = last_number_from_text(text)

                    try:
                        ans = int(float(convert_to_arabic_digits(str(ex["answer"]))))
                    except:
                        ans = None

                    is_correct = pred == ans
                    total += 1
                    correct += int(is_correct)

                    for kw in args.sources:
                        if kw in ex["source"]:
                            keyword_stats[kw]["total"] += 1
                            keyword_stats[kw]["correct"] += int(is_correct)

                    results.append({
                        "source": ex["source"],
                        "full_response": text,
                        "pred_number": pred if pred is not None else "",
                        "answer": ans if ans is not None else "",
                    })

                    pbar.update(1)

            pbar.close()

        worker_results[lang] = {
            "total": total,
            "correct": correct,
            "results": results,
            "keyword_stats": keyword_stats,
        }

    return_dict[rank] = worker_results

    # 清理
    del llm
    torch.cuda.empty_cache()

# ----------------------
# 主函数
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name or path (absolute path or relative to --model-dir)")
    parser.add_argument("--model-dir", default="/root/autodl-tmp/local_model", help="Default directory for models (when --model is not absolute)")
    parser.add_argument("--batch-size", default="8", help="Batch size: a number (e.g., 8) or 'all' to let vLLM handle all data at once (faster)")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--data-dir", default="eval_data/mmath")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    # 解析 batch_size 参数
    if args.batch_size.lower() == "all":
        args.use_vllm_batch_all = True
        args.batch_size = None  # 不需要数值
    else:
        args.use_vllm_batch_all = False
        try:
            args.batch_size = int(args.batch_size)
        except ValueError:
            parser.error(f"--batch-size must be a number or 'all', got: {args.batch_size}")

    # 加载 YAML 配置
    config_path = "./configs/" + args.config + ".yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    args.sources = ["mgsm", "polymath-low"]
    langs = list(config["languages"].keys())

    os.makedirs(args.output_dir, exist_ok=True)

    # 预加载并切分所有语言数据
    all_lang_data = {}
    for lang in langs:
        path = os.path.join(args.data_dir, f"{lang}.jsonl")
        if not os.path.exists(path):
            continue

        with open(path, encoding="utf-8") as f:
            data = [json.loads(l) for l in f]

        chunk_size = math.ceil(len(data) / args.num_gpus)
        all_lang_data[lang] = [
            data[i * chunk_size:(i + 1) * chunk_size]
            for i in range(args.num_gpus)
        ]

    # 启动 GPU 常驻进程
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for rank in range(args.num_gpus):
        rank_lang_data = {
            lang: all_lang_data[lang][rank]
            for lang in all_lang_data
        }

        p = mp.Process(
            target=worker_process,
            args=(rank, args.num_gpus, args, rank_lang_data, config, return_dict)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 汇总 & 保存
    summary_rows = []

    for lang in all_lang_data:
        total = 0
        correct = 0
        results = []
        keyword_stats = {kw: {"total": 0, "correct": 0} for kw in args.sources}

        for v in return_dict.values():
            lang_res = v[lang]
            total += lang_res["total"]
            correct += lang_res["correct"]
            results.extend(lang_res["results"])

            for kw in args.sources:
                keyword_stats[kw]["total"] += lang_res["keyword_stats"][kw]["total"]
                keyword_stats[kw]["correct"] += lang_res["keyword_stats"][kw]["correct"]

        out_csv = os.path.join(args.output_dir, f"{lang}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["source", "full_response", "pred_number", "answer"]
            )
            writer.writeheader()
            writer.writerows(results)

        acc = correct / max(1, total)
        print(f"\n✅ {lang}: {acc:.4f} ({correct}/{total})")

        for kw in args.sources:
            k_total = keyword_stats[kw]["total"]
            k_correct = keyword_stats[kw]["correct"]
            k_acc = k_correct / max(1, k_total)
            print(f"  {kw}: {k_acc:.4f}")

            summary_rows.append({
                "language": lang,
                "source": kw,
                "total": k_total,
                "correct": k_correct,
                "accuracy": round(k_acc, 4)
            })

        summary_rows.append({
            "language": lang,
            "source": "total",
            "total": total,
            "correct": correct,
            "accuracy": round(acc, 4)
        })

    summary_csv = os.path.join(args.output_dir, "result.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["language", "source", "total", "correct", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
