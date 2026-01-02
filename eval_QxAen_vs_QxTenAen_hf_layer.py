import os
import json
import argparse
import torch
import random
import time
from tqdm import tqdm
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from utils import build_chat_prompt, last_number_from_text, save_results

langs = ["bn", "de", "es", "fr", "ja", "ru", "th"]

TRANSLATION_PROMPT = "Problem in English: "

SOLVE_PROMPT = (
    "Problem in {language}: {problem}\n\n" +
    TRANSLATION_PROMPT + "{translation}\n\n" +
    "Solve the problem using English and enclose the final number at the end of the response in $\\boxed{{}}$."
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--model-dir", default="/root/autodl-tmp/local_model")
    parser.add_argument(
        "--mask-layers",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Mask layers in [START, END)"
    )
    args = parser.parse_args()
    mask_set = f"L{args.mask_layers}" if args.mask_layers is not None else "None"
    args.output_dir = os.path.join("output", args.model, "QxTenAen-2step-hf-mask", mask_set)
    args.data_dir = os.path.join("output", args.model, "translation")
    args.top_k = 64
    args.top_p = 0.9
    args.max_tokens = 800
    args.temperature = 0.3
    return args

import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    TopKLogitsWarper,
)

@torch.no_grad()
def prefill(
    model,
    input_ids,                 # [B, L_prompt]
    attention_mask,            # [B, L_prompt], left padding = 0
    block_size=32,
):
    B, L = input_ids.shape
    past = None
    for i in range(0, L, block_size):
        step_ids = input_ids[:, i: i + block_size]           # [B, block_size]
        step_mask = attention_mask[:, : i + block_size]      # [B, i + block_size]
        outputs = model(
            input_ids=step_ids,
            attention_mask=step_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        past = outputs.past_key_values

    return past

@torch.no_grad()
def generate(
    model,
    input_ids,                 # [B, L_prompt]
    attention_mask,            # [B, L_prompt], left padding = 0
    eos_token_id,
    max_new_tokens,
    temperature=1.0,
    top_p=1.0,
    top_k=None,
    past_key_values=None,
):
    device = input_ids.device
    B, L = input_ids.shape

    generated = input_ids.clone()
    past = past_key_values

    logits_processor = LogitsProcessorList()
    if temperature != 1.0:
        logits_processor.append(TemperatureLogitsWarper(temperature))
    if top_p < 1.0:
        logits_processor.append(TopPLogitsWarper(top_p))
    if top_k != None:
        logits_processor.append(TopKLogitsWarper(top_k))

    # 1. Prefill prompt（如果没有给 past
    if past is None:
        past = prefill(model, input_ids, attention_mask)

    # 2. Autoregressive generation
    cur_len = generated.size(1)
    B = input_ids.size(0)
    finish_flags = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_new_tokens):
        step_ids = generated[:, -1:]
        step_mask = torch.cat(
            [attention_mask, torch.ones(B, cur_len - attention_mask.size(1) + 1, device=device)],
            dim=1
        )[:, : cur_len + 1]

        outputs = model(
            input_ids=step_ids,
            attention_mask=step_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        logits = outputs.logits[:, -1, :]
        logits = logits_processor(generated, logits)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B,1]

        generated = torch.cat([generated, next_token], dim=1)
        past = outputs.past_key_values

        cur_len += 1

        finish_flags = finish_flags | (next_token.squeeze(1) == eos_token_id)
        if finish_flags.all():
            break

    return generated

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

    empty_token_id = tokenizer("                ")["input_ids"][0]

    def run_generate(data, max_new_tokens, batch_size, data_unfinished):

        # 按照 prompt 长度排序，减少 padding
        data.sort(key=lambda x: len(x["input_ids"]))

        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
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

            prompt_kv_cache = prefill(
                model,
                padded_input_ids,
                attention_mask,
            )

            padded_input_ids_wo_t = padded_input_ids.clone()
            for i, (p_start, p_end, t_start, t_end) in enumerate(spans):
                # 将 t_span 的内容替换为 pad_token_id
                padded_input_ids_wo_t[i, t_start:t_end] = empty_token_id

            prompt_kv_cache_wo_t = prefill(
                model,
                padded_input_ids_wo_t,
                attention_mask,
            )

            # 将 mask_layers 的 kv 替换为没有 translation 的 kv
            if args.mask_layers is not None:
                for layer in range(args.mask_layers[0], args.mask_layers[1]):
                    prompt_kv_cache[layer][0][:,:,:,:] = prompt_kv_cache_wo_t[layer][0][:,:,:,:]
                    prompt_kv_cache[layer][1][:,:,:,:] = prompt_kv_cache_wo_t[layer][1][:,:,:,:]
            
            outputs = generate(
                model,
                padded_input_ids,
                attention_mask,
                tokenizer.eos_token_id,
                max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                past_key_values=prompt_kv_cache
            )

            finished_count = 0

            for i, ex in enumerate(batch):
                response_ids = outputs[i, prompt_len:].cpu()
                eos_idx = (response_ids == tokenizer.eos_token_id).nonzero()
                if data_unfinished is not None and len(eos_idx) == 0:
                    data_unfinished.append(ex)
                    continue

                eos_idx = eos_idx[0].item() if len(eos_idx) > 0 else len(response_ids)
                response = tokenizer.decode(response_ids[:eos_idx], skip_special_tokens=False).strip()

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
    run_generate(data_unfinished, args.max_tokens, args.batch_size, None)

    return_dict[rank] = records


def build_token_spans(ex, tokenizer):

    problem = ex["problem"]
    translation = ex["translation"]
    user_content = SOLVE_PROMPT.format(
        problem=problem,
        language=ex["lang"],
        translation=translation
    )
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
        s = prompt.find(start_str)
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