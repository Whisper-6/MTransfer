import os
import json
import csv
import argparse
import math
import multiprocessing as mp
import time
import torch
import yaml
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import last_number_from_text, build_chat_prompt, save_results

langs = ["bn", "de", "es", "fr", "ja", "ru", "th"]

SOLVE_PROMPT = (
    "Problem: {problem}\n\n"
    "English Translation: {translation}\n\n"
    "Solve the problem in English and enclose the final number at the end of the response in $\\boxed{{}}$.\n\n"
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--model-dir", default="/root/autodl-tmp/local_model")
    args = parser.parse_args()
    args.output_dir = os.path.join("output", args.model, "QxTenAen-2step-PTI")
    args.data_dir = os.path.join("output", args.model, "translation")
    args.top_k = 64
    args.top_p = 0.9
    args.max_tokens = 1024
    args.temperature = 0.3
    return args

# ---------------------- worker_process ----------------------
def worker_process(rank, args, data_batches, return_dict, progress):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)

    from vllm import LLM, SamplingParams

    try:
        model_path = os.path.join(args.model_dir, args.model)
        llm = LLM(model=model_path, dtype="half")

        records = []

        for batch in data_batches:
            batch_prompts = [p["prompt"] for p in batch]

            outputs = llm.generate(
                batch_prompts,
                SamplingParams(
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    stop=["<|user|>"],
                    top_k=args.top_k,
                    top_p=args.top_p,
                    seed=None,
                ),
                use_tqdm=False,
            )

            for ex, ex_out in zip(batch, outputs):
                ans = ex["answer"]

                for out in ex_out.outputs:
                    response = out.text.strip()
                    pred = last_number_from_text(response)
                    is_correct = (pred == ans)

                    records.append({
                        "lang": ex["lang"],
                        "source": ex["source"],
                        "prompt": ex["prompt"],
                        "response": response,
                        "pred": pred if pred is not None else "",
                        "answer": ans,
                        "is_correct": int(is_correct),
                    })

            with progress.get_lock():
                progress.value += 1     

        return_dict[rank] = records
        del llm
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error in worker {rank}: {e}")

# ---------------------- main ----------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- 加载数据 ----------
    data = []
    for lang in langs:
        path = os.path.join(args.data_dir, f"{lang}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            lang_data = [json.loads(l) for l in f]
        for ex in lang_data:
            ex["lang"] = lang
        data.extend(lang_data)

    model_path = os.path.join(args.model_dir, args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    for ex in data:
        lang = ex["lang"]
        user_content = SOLVE_PROMPT.format(problem=ex["problem"], translation=ex["translation"])
        prompt = build_chat_prompt(tokenizer, user_content)
        ex["prompt"] = prompt

    data.sort(key=lambda x: len(x["prompt"]))
        
    # 将 data 划分为 batches, 随机打乱，均分给各个 gpu
    data_batches = []
    for i in range(0, len(data), args.batch_size):
        batch = data[i : i + args.batch_size]
        data_batches.append(batch)
    random.shuffle(data_batches)
    total_batches = len(data_batches)
    
    # ---------- 多进程 ----------
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    progress = mp.Value('i', 0)

    for rank in range(args.num_gpus):
        rank_data_batches = data_batches[rank::args.num_gpus]
        p = mp.Process(
            target=worker_process,
            args=(rank, args, rank_data_batches, return_dict, progress),
        )
        p.start()
        processes.append(p)

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

    lang_results = {lang: [] for lang in langs}
    for rank in range(args.num_gpus):
        for record in return_dict[rank]:
            lang_results[record["lang"]].append(record)
    
    save_results(args, langs, lang_results)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()