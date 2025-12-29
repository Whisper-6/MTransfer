import os
import json
import argparse
import torch
import random
import time
from tqdm import tqdm
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import build_chat_prompt, last_number_from_text, save_results

langs = ["bn", "de", "es", "fr", "ja", "ru", "th"]
# langs = ["bn"]

SOLVE_PROMPT = (
    "Problem: {problem}\n\n"
    "English Translation: {translation}\n\n"
    "Solve the problem in English and enclose the final number at the end of the response in $\\boxed{{}}$.\n\n"
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--model-dir", default="/root/autodl-tmp/local_model")
    args = parser.parse_args()
    args.output_dir = os.path.join("output", args.model, "QxTenAen-2step-hf")
    args.data_dir = os.path.join("output", args.model, "translation")
    args.top_k = 64
    args.top_p = 0.9
    args.max_tokens = 1024
    args.temperature = 0.3
    return args

import torch
from tqdm import tqdm

def worker_process(rank, args, data, return_dict, progress):
    import os
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # ---------- 加载模型 ----------
    model_path = os.path.join(args.model_dir, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16
    )
    model.to(device)
    model.eval()

    records = []

    def run_generate(data, max_new_tokens, batch_size, data_unfinished):

        # 按照 prompt 长度排序，减少 padding
        data.sort(key=lambda x: len(x["input_ids"]))

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            B = len(batch)

            prompt_len = max([len(ex["input_ids"]) for ex in batch])
            padded_input_ids = torch.full((B, prompt_len), tokenizer.pad_token_id,
                                        dtype=torch.long, device=device)
            attention_mask = torch.zeros((B, prompt_len), dtype=torch.long, device=device)

            spans = []
            for i, ex in enumerate(batch):
                input_ids = torch.tensor(ex["input_ids"], device=device)
                input_len = len(input_ids)
                left_pad = prompt_len - input_len

                padded_input_ids[i, left_pad:prompt_len] = input_ids
                attention_mask[i, left_pad:prompt_len] = 1

                p_start, p_end = ex["p_span"]
                t_start, t_end = ex["t_span"]
                p_start += left_pad
                p_end += left_pad
                t_start += left_pad
                t_end += left_pad
                spans.append((p_start, p_end, t_start, t_end))

            # ---------- Prefill prompt (分块) ----------
            past = None
            block_size = 32
            for start in range(0, prompt_len-1, block_size):
                end = min(start + block_size, prompt_len-1)
                input_ids_step = padded_input_ids[:, start:end]
                mask_step = attention_mask[:, :end]
                with torch.inference_mode():
                    outputs = model(
                        input_ids=input_ids_step,
                        attention_mask=mask_step,
                        past_key_values=past,
                        use_cache=True,
                        output_attentions=False
                    )
                past = outputs.past_key_values

            # ---------- Generate response ----------
            outputs = model.generate(
                input_ids=padded_input_ids,        
                attention_mask=attention_mask,     
                past_key_values=past,              
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
            )

            finished_count = 0

            for i, ex in enumerate(batch):
                response_ids = outputs[i, prompt_len:].cpu()
                eos_idx = (response_ids == tokenizer.eos_token_id).nonzero()
                if data_unfinished is not None and len(eos_idx) == 0:
                    data_unfinished.append(ex)
                    continue

                gen_len = eos_idx[0].item()
                response = tokenizer.decode(response_ids[:gen_len], skip_special_tokens=False).strip()

                pred = last_number_from_text(response)
                ans = ex["answer"]
                is_correct = (pred == ans)
                
                records.append({
                    "lang": ex["lang"],
                    "source": ex["source"],
                    "problem": ex["problem"],
                    "translation": ex["translation"],
                    "response": response,
                    "pred": pred if pred is not None else "",
                    "answer": ans,
                    "is_correct": int(is_correct),
                })

                finished_count += 1

            with progress.get_lock():
                progress.value += finished_count

    data_unfinished = []
    run_generate(data, args.max_tokens // 2, args.batch_size * 2, data_unfinished)
    print("unfinished samples:", len(data_unfinished))
    run_generate(data_unfinished, args.max_tokens, args.batch_size, None)

    return_dict[rank] = records


def build_token_spans(ex, tokenizer):

    problem = ex["problem"]
    translation = ex["translation"]
    user_content = SOLVE_PROMPT.format(problem=problem, translation=translation)
    prompt = build_chat_prompt(tokenizer, user_content)

    encoding = tokenizer(
        prompt,
        return_offsets_mapping=True,
        add_special_tokens=True
    )
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    # ---------------- char span -> token span ----------------
    def find_span(start_str, end_str):
        s = prompt.find(start_str) + len(start_str)
        e = prompt.find(end_str, s)
        return s, e

    p_char_start, p_char_end = find_span("Problem: ", "\n\n")
    t_char_start, t_char_end = find_span("English Translation: ", "\n\n")

    starts = torch.tensor([s for s, _ in offsets])
    ends   = torch.tensor([e for _, e in offsets])

    def char2token_span(char_start, char_end):
        token_start = torch.searchsorted(ends, char_start, right=True).item()
        token_end   = torch.searchsorted(starts, char_end, right=False).item()
        return token_start, token_end

    p_span = char2token_span(p_char_start, p_char_end)
    t_span = char2token_span(t_char_start, t_char_end)

    return {
        "input_ids": input_ids,
        "p_span": p_span,
        "t_span": t_span,
    }


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
        random.shuffle(lang_data)
        lang_data = lang_data[:500]
        for ex in lang_data:
            ex["lang"] = lang
        data.extend(lang_data)

    model_path = os.path.join(args.model_dir, args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # ---------- 加载 tokenizer ----------
    model_path = os.path.join(args.model_dir, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ---------- 生成 tokenized 和 span ----------
    for ex in tqdm(data, desc="Tokenizing samples"):
        span_info = build_token_spans(ex, tokenizer)
        ex.update(span_info)
    random.shuffle(data)
    total_samples = len(data)

    # ---------- 多 GPU 处理 ----------
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    progress = mp.Value('i', 0)

    for rank in range(args.num_gpus):
        rank_data = data[rank::args.num_gpus]
        p = mp.Process(
            target=worker_process,
            args=(rank, args, rank_data, return_dict, progress),
        )
        p.start()
        processes.append(p)

    with tqdm(total=total_samples, desc="Generating samples") as pbar:
        last = 0
        while True:
            with progress.get_lock():
                cur = progress.value
            if cur > last:
                pbar.update(cur - last)
                last = cur
            if cur >= total_samples:
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
    mp.set_start_method("spawn")
    main()