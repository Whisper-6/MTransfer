import os
import json
import argparse
import torch
import random
import time
from tqdm import tqdm
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import build_batches

MAX_GEN_LEN = 1000

langs = ["bn", "de", "es", "fr", "ja", "ru", "th"]
# langs = ["bn"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--model-dir", default="/root/autodl-tmp/local_model")
    args = parser.parse_args()
    args.data_dir = os.path.join("output", args.model, args.config)
    args.output_dir = args.data_dir
    return args

import torch
from tqdm import tqdm

def worker_process(rank, args, data_batches, return_dict, progress):
    import os
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # ---------- 加载模型 ----------
    model_path = os.path.join(args.model_dir, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()
    model.set_attn_implementation("eager")

    num_layers = model.config.num_hidden_layers
    num_heads  = model.config.num_attention_heads
    kv_heads   = model.config.num_key_value_heads
    head_group_size = num_heads // kv_heads
    head_dim   = model.config.hidden_size // num_heads

    records = []

    for batch in data_batches:
        B = len(batch)

        # 统计量用 fp32
        attn_p_stats = torch.zeros((B, num_layers), device=device, dtype=torch.float32)
        attn_t_stats = torch.zeros((B, num_layers), device=device, dtype=torch.float32)
        delta_p_len_stats = torch.zeros((B, num_layers, num_heads), device=device, dtype=torch.float32)
        delta_t_len_stats = torch.zeros((B, num_layers, num_heads), device=device, dtype=torch.float32)
        dot_tp_stats = torch.zeros((B, num_layers, num_heads), device=device, dtype=torch.float32)
        cos_tp_stats = torch.zeros((B, num_layers, num_heads), device=device, dtype=torch.float32)

        gen_start_idxs = [ex["gen_start_idx"] for ex in batch]
        prompt_len = max(gen_start_idxs)

        gen_lens = [len(ex["tokenized"]) - ex["gen_start_idx"] for ex in batch]
        max_gen_len = max(gen_lens)
        total_len = prompt_len + max_gen_len

        # ---------- 准备输入 ----------
        padded_input_ids = torch.full((B, total_len), tokenizer.pad_token_id,
                                      dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, total_len), device=device)
        p_span_mask = torch.zeros((B, prompt_len), device=device, dtype=torch.bfloat16)
        t_span_mask = torch.zeros((B, prompt_len), device=device, dtype=torch.bfloat16)

        spans = []

        for i, ex in enumerate(batch):
            ids = torch.tensor(ex["tokenized"], device=device)
            g = ex["gen_start_idx"]
            seq_len = len(ids)
            left_pad = prompt_len - g

            padded_input_ids[i, left_pad:left_pad + seq_len] = ids
            attention_mask[i, left_pad:left_pad + seq_len] = 1

            p_start, p_end = ex["p_span"]
            t_start, t_end = ex["t_span"]
            p_start += left_pad
            p_end += left_pad
            t_start += left_pad
            t_end += left_pad
            spans.append((left_pad, p_start, p_end, t_start, t_end))
            
            p_span_mask[i, p_start:p_end] = 1.0
            t_span_mask[i, t_start:t_end] = 1.0

        # ---------- prompt 部分 ----------
        past = None
        for step in range(prompt_len):
            input_ids_step = padded_input_ids[:, step:step+1]
            mask_step = torch.ones(B, step+1, device=device)

            for b, ex in enumerate(batch):
                left_pad, p_start, p_end, t_start, t_end = spans[b]
                mask_step[b, :left_pad] = 0
                if p_start < t_start:
                    if t_start <= step and step < t_end:
                        mask_step[b, p_start:p_end] = 0
                else:
                    if p_start <= step and step < p_end:
                        mask_step[b, t_start:t_end] = 0

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids_step,
                    attention_mask=mask_step,
                    past_key_values=past,
                    use_cache=True,
                    output_attentions=False
                )
            past = outputs.past_key_values

        # ---------- response 部分 ----------
        for step in range(prompt_len, total_len):
            input_ids_step = padded_input_ids[:, step:step+1]
            mask_step = attention_mask[:, :step+1]
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids_step,
                    attention_mask=mask_step,
                    past_key_values=past,
                    use_cache=True,
                    output_attentions=True
                )

            past = outputs.past_key_values
            attentions = outputs.attentions

            fin_mask = torch.tensor([step >= gen_lens[b] + prompt_len for b in range(B)],
                                    device=device, dtype=torch.bool)
            mask_exp = (~fin_mask).to(torch.bfloat16).unsqueeze(-1).unsqueeze(-1)  # [B,1,1]

            for layer_idx in range(num_layers):
                for kv_head_idx in range(kv_heads):
                    h_start = kv_head_idx * head_group_size
                    h_end = (kv_head_idx + 1) * head_group_size

                    attn_lg = attentions[layer_idx][:, h_start:h_end, -1, :prompt_len]
                    v_lg = past[layer_idx][1][:, kv_head_idx, :prompt_len, :]

                    attn_lg_masked = attn_lg * mask_exp

                    attn_p = attn_lg_masked * p_span_mask.unsqueeze(1)      # [B, 1, L]
                    delta_p = torch.einsum("bgl,bld->bgd", attn_p, v_lg)    # [B, G, D]
                    attn_t = attn_lg_masked * t_span_mask.unsqueeze(1)      # [B, 1, L]
                    delta_t = torch.einsum("bgl,bld->bgd", attn_t, v_lg)    # [B, G, D]

                    attn_p_stats[:, layer_idx] += torch.sum(attn_p, dim=(1,2)).to(torch.float32)
                    attn_t_stats[:, layer_idx] += torch.sum(attn_t, dim=(1,2)).to(torch.float32)

                    delta_p_len_lg = torch.norm(delta_p, dim=-1)    # [B, G]
                    delta_t_len_lg = torch.norm(delta_t, dim=-1)    # [B, G]
                    dot_tp = torch.sum(delta_p * delta_t, dim=-1)   # [B, G]
                    cos_tp = dot_tp / (delta_p_len_lg * delta_t_len_lg + 1e-8)

                    dot_tp_stats[:, layer_idx, h_start:h_end] += dot_tp.to(torch.float32)
                    cos_tp_stats[:, layer_idx, h_start:h_end] += cos_tp.to(torch.float32)
                    delta_p_len_stats[:, layer_idx, h_start:h_end] += delta_p_len_lg.to(torch.float32)
                    delta_t_len_stats[:, layer_idx, h_start:h_end] += delta_t_len_lg.to(torch.float32)
        
        for i, ex in enumerate(batch):
            attn_p = attn_p_stats[i].cpu() / (gen_lens[i] * num_heads)
            attn_t = attn_t_stats[i].cpu() / (gen_lens[i] * num_heads)
            delta_p_len = delta_p_len_stats[i].cpu() / gen_lens[i]
            delta_t_len = delta_t_len_stats[i].cpu() / gen_lens[i]
            dot_tp = dot_tp_stats[i].cpu() / gen_lens[i]
            cos_tp = cos_tp_stats[i].cpu() / gen_lens[i]
            
            records.append({
                "lang": ex["lang"],
                "attn_p": attn_p.tolist(),
                "attn_t": attn_t.tolist(),
                "delta_p_len": delta_p_len.tolist(),
                "delta_t_len": delta_t_len.tolist(),
                "dot_tp": dot_tp.tolist(),
                "cos_tp": cos_tp.tolist(),
            })


        with progress.get_lock():
            progress.value += 1

    return_dict[rank] = records


def build_token_spans(ex, tokenizer):

    prompt = ex["prompt"]
    response = ex["response"]
    text = prompt + response

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True
    )
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    # ---------------- char span -> token span ----------------
    def find_span(start_str, end_str):
        s = text.find(start_str) + len(start_str)
        e = text.find(end_str, s)
        return s, e

    p_char_start, p_char_end = find_span("Problem: ", "\n\n")
    t_char_start, t_char_end = find_span("English Translation: ", "\n\n")
    gen_start = len(prompt)

    starts = torch.tensor([s for s, _ in offsets])
    ends   = torch.tensor([e for _, e in offsets])

    def char2token_span(char_start, char_end):
        token_start = torch.searchsorted(ends, char_start, right=True).item()
        token_end   = torch.searchsorted(starts, char_end, right=False).item()
        return token_start, token_end

    p_span = char2token_span(p_char_start, p_char_end)
    t_span = char2token_span(t_char_start, t_char_end)
    gen_start_idx = torch.searchsorted(ends, gen_start, right=True).item()

    return {
        "tokenized": input_ids,
        "p_span": p_span,
        "t_span": t_span,
        "gen_start_idx": gen_start_idx,
        "gen_len": len(input_ids) - gen_start_idx,
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
        lang_data = lang_data[:250]
        for ex in lang_data:
            ex["lang"] = lang
            for k in ["pred", "answer", "is_correct"]:
                ex.pop(k, None)
            data.append(ex)

    print(f"Total loaded {len(data)} samples from {args.data_dir}")

    # ---------- 加载 tokenizer ----------
    model_path = os.path.join(args.model_dir, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ---------- 生成 tokenized 和 span ----------
    for ex in tqdm(data, desc="Tokenizing samples"):
        span_info = build_token_spans(ex, tokenizer)
        ex.update(span_info)

    # 抛弃回答过长的数据（基本都是 repeating）
    data = [ex for ex in data if len(ex["tokenized"]) - ex["gen_start_idx"] <= MAX_GEN_LEN]

    print(f"Total {len(data)} samples after filtering long generations")

    # ---------- 贪心划分 batch ----------
    data_batches = build_batches(gen_cost_ratio=3, data=data, B=args.batch_size, max_mass=2*MAX_GEN_LEN)
    total_batches = len(data_batches)
    random.shuffle(data_batches)

    # ---------- 多 GPU 处理 ----------
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

    result = []
    
    for lang, records in lang_results.items():
        total = len(records)
        attn_p = sum([torch.tensor(r["attn_p"]) for r in records]) / total
        attn_t = sum([torch.tensor(r["attn_t"]) for r in records]) / total
        delta_p_len = sum([torch.tensor(r["delta_p_len"]) for r in records]) / total
        delta_t_len = sum([torch.tensor(r["delta_t_len"]) for r in records]) / total
        dot_tp = sum([torch.tensor(r["dot_tp"]) for r in records]) / total
        cos_tp = sum([torch.tensor(r["cos_tp"]) for r in records]) / total
        
        result.append({
            "lang": lang,
            "attn_p": attn_p.tolist(),
            "attn_t": attn_t.tolist(),
            "delta_p_len": delta_p_len.tolist(),
            "delta_t_len": delta_t_len.tolist(),
            "dot_tp": dot_tp.tolist(),
            "cos_tp": cos_tp.tolist(),
        })

    output_path = os.path.join(args.output_dir, "attn_result.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in result:
            f.write(json.dumps(r) + "\n")
        

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()