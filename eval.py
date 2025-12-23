import os
import json
import csv
import argparse
import math
import multiprocessing as mp
import time
import torch
from vllm import LLM, SamplingParams
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import last_number_from_text

import warnings
warnings.filterwarnings("ignore", message=".*destroy_process_group.*")

# ---------------------- worker_process ----------------------
def worker_process(rank, args, rank_lang_data, config, return_dict, progress):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)

    model_path = os.path.join(args.model_dir, args.model)
    llm = LLM(model=model_path, dtype="half")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    worker_results = {}

    for lang, data_slice in rank_lang_data.items():
        if not data_slice:
            continue

        yaml_prompt = config["languages"][lang]["prompt"]
        question_field = config.get("question_field", "m_query")

        # ---------- 构造 prompts（每题 num_samples 次） ----------
        all_prompts, prompt_meta, prompt_texts = [], [], []
        per_source_qids = {src: [] for src in args.sources}

        for ex_id, ex in enumerate(data_slice):
            for src in args.sources:
                if src in ex["source"]:
                    per_source_qids[src].append(ex_id)

            user_content = yaml_prompt.format(question=ex[question_field])
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            for _ in range(args.num_samples):
                all_prompts.append(prompt)
                prompt_meta.append(ex_id)
                prompt_texts.append(prompt)

        # ---------- 一次性生成 ----------
        outputs = llm.generate(
            all_prompts,
            SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stop=["<|user|>"],
                seed=None,
            ),
            use_tqdm=False,
        )

        # ---------- 更新进度 ----------
        try:
            with progress.get_lock():
                progress.value += 1
        except Exception:
            pass

        # ---------- 统计 ----------
        per_example_correct = [0] * len(data_slice)
        all_records = []

        for ex_id, prompt, out in zip(prompt_meta, prompt_texts, outputs):
            ex = data_slice[ex_id]
            text = out.outputs[0].text.strip()
            pred = last_number_from_text(text)
            ans = ex["answer"]

            is_correct = (pred == ans)
            per_example_correct[ex_id] += int(is_correct)

            record = {
                "language": lang,
                "source": ex["source"],
                "prompt": prompt,
                "full_response": text,
                "pred_number": pred if pred is not None else "",
                "answer": ans if ans is not None else "",
                "is_correct": int(is_correct),
            }
            all_records.append(record)

        k = args.num_samples
        N = len(data_slice)
        total_correct = sum(per_example_correct)
        total_trials = N * k

        source_stats = {}
        for src in args.sources:
            ids = per_source_qids[src]
            if not ids:
                continue
            correct_src = sum(per_example_correct[i] for i in ids)
            total_src = len(ids) * k
            source_stats[src] = {
                "num_questions": len(ids),
                "total_correct": correct_src,
                "total_trials": total_src,
            }

        worker_results[lang] = {
            "num_questions": N,
            "total_correct": total_correct,
            "total_trials": total_trials,
            "sources": source_stats,
            "all_records": all_records,  # 存全部样本
        }

    return_dict[rank] = worker_results
    del llm
    torch.cuda.empty_cache()


# ---------------------- main ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default="eval_data/mmath")
    parser.add_argument("--model-dir", default="/root/autodl-tmp/local_model")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)

    args = parser.parse_args()
    args.sources = ["mgsm", "polymath-low"]

    if args.output_dir is None:
        args.output_dir = os.path.join("output", args.model, args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join("configs", f"{args.config}.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)

    langs = list(config["languages"].keys())

    # ---------- 加载数据 ----------
    all_lang_data = {}
    for lang in langs:
        path = os.path.join(args.data_dir, f"{lang}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = [json.loads(l) for l in f]

        chunk = math.ceil(len(data) / args.num_gpus)
        all_lang_data[lang] = [data[i * chunk:(i + 1) * chunk] for i in range(args.num_gpus)]

    # ---------- 多进程 ----------
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    total_batches = 0
    for lang in all_lang_data:
        for r in range(args.num_gpus):
            if all_lang_data[lang][r]:
                total_batches += 1

    progress = mp.Value('i', 0)

    for rank in range(args.num_gpus):
        rank_data = {lang: all_lang_data[lang][rank] for lang in all_lang_data}
        p = mp.Process(
            target=worker_process,
            args=(rank, args, rank_data, config, return_dict, progress),
        )
        p.start()
        processes.append(p)

    if total_batches > 0:
        with tqdm(total=total_batches, desc="generate batches") as pbar:
            last = 0
            while True:
                with progress.get_lock():
                    cur = progress.value
                if cur > last:
                    pbar.update(cur - last)
                    last = cur
                if cur >= total_batches:
                    break
                time.sleep(0.1)

    for p in processes:
        p.join()

    # ---------- 汇总结果 ----------
    summary_rows = []

    for lang in all_lang_data:
        total_correct = 0
        total_trials = 0
        all_samples = []

        source_agg = {src: {"correct": 0, "total": 0} for src in args.sources}

        for v in return_dict.values():
            if lang not in v:
                continue
            res = v[lang]
            total_correct += res["total_correct"]
            total_trials += res["total_trials"]
            all_samples.extend(res["all_records"])

            for src, sres in res["sources"].items():
                source_agg[src]["correct"] += sres["total_correct"]
                source_agg[src]["total"] += sres["total_trials"]

        # ---------- source-level CI ----------
        for src in args.sources:
            agg = source_agg[src]
            if agg["total"] == 0:
                continue
            acc = agg["correct"] / agg["total"]
            stderr = math.sqrt(acc * (1 - acc) / agg["total"])
            ci_radius = 1.96 * stderr
            summary_rows.append({
                "language": lang,
                "source": src,
                "total": agg["total"],
                "correct": agg["correct"],
                "accuracy": round(acc, 4),
                "ci_radius": round(ci_radius, 4),
            })

        # ---------- total CI ----------
        if total_trials > 0:
            acc_total = total_correct / total_trials
            stderr_total = math.sqrt(acc_total * (1 - acc_total) / total_trials)
            ci_radius_total = 1.96 * stderr_total
        else:
            acc_total, ci_radius_total = 0.0, 0.0

        summary_rows.append({
            "language": lang,
            "source": "total",
            "total": total_trials,
            "correct": total_correct,
            "accuracy": round(acc_total, 4),
            "ci_radius": round(ci_radius_total, 4),
        })

        # ---------- 保存样本 CSV ----------
        if all_samples:
            out_path = os.path.join(args.output_dir, f"{lang}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["language", "source", "prompt", "full_response", "pred_number", "answer", "is_correct"]
                )
                writer.writeheader()
                writer.writerows(all_samples)

        print(f"✅ {lang}: acc={acc_total:.4f}, ci_radius={ci_radius_total:.4f}")

    # ---------- 保存 summary CSV ----------
    out_csv = os.path.join(args.output_dir, "result.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["language", "source", "total", "correct", "accuracy", "ci_radius"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()