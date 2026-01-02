import os
import json
import argparse
import torch
import random
import time
from tqdm import tqdm
import multiprocessing as mp
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    TopKLogitsWarper,
)

from utils import build_chat_prompt, last_number_from_text, save_results

# ===================== config =====================
langs = ["bn", "de", "es", "fr", "ja", "ru", "th"]
# langs = ["ja"]

LANGUAGE = {
    "bn": "Bengali",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "ru": "Russian",
    "th": "Thai",
}

M_QUERY_PROMPT = "Problem in {language}: {problem}\n"
EN_QUERY_PROMPT = "Problem in English: {translation}\n"
INSTRUCTION = "\nSolve the problem using English and enclose the final number at the end of the response in $\\boxed{}$."

SPECIAL_TOKEN = "<|EN_SPLIT|>"

# ===================== args =====================
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
    )
    args = parser.parse_args()

    mask_set = f"L{args.mask_layers}" if args.mask_layers else "None"
    args.output_dir = os.path.join("output", args.model, "QxTenAen-QenAen", mask_set)
    args.data_dir = os.path.join("output", args.model, "translation")

    args.top_k = 64
    args.top_p = 0.9
    args.max_tokens = 800
    args.temperature = 0.3
    return args


# ===================== generation utils =====================
@torch.no_grad()
def prefill(model, input_ids, attention_mask, block_size=32):
    B, L = input_ids.shape
    past = None
    for i in range(0, L, block_size):
        step_ids = input_ids[:, i:i + block_size]
        step_mask = attention_mask[:, :i + block_size]
        out = model(
            input_ids=step_ids,
            attention_mask=step_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
    return past


@torch.no_grad()
def generate(
    model,
    input_ids,
    attention_mask,
    eos_token_id,
    max_new_tokens,
    temperature=1.0,
    top_p=1.0,
    top_k=None,
    past_key_values=None,
):
    device = input_ids.device
    B = input_ids.size(0)
    generated = input_ids.clone()
    past = past_key_values
    cur_len = generated.size(1)

    processors = LogitsProcessorList()
    if temperature != 1.0:
        processors.append(TemperatureLogitsWarper(temperature))
    if top_p < 1.0:
        processors.append(TopPLogitsWarper(top_p))
    if top_k is not None:
        processors.append(TopKLogitsWarper(top_k))

    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        step_ids = generated[:, -1:]
        step_mask = torch.cat(
            [attention_mask, torch.ones(B, 1, device=device)],
            dim=1
        )

        out = model(
            input_ids=step_ids,
            attention_mask=step_mask,
            past_key_values=past,
            use_cache=True,
        )
        logits = processors(generated, out.logits[:, -1])
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        generated = torch.cat([generated, next_token], dim=1)
        past = out.past_key_values
        attention_mask = step_mask

        finished |= (next_token.squeeze(1) == eos_token_id)
        if finished.all():
            break

    return generated


# ===================== worker =====================
def worker_process(rank, args, data, return_dict, progress):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model_path = os.path.join(args.model_dir, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
    ).to(device).eval()

    records = []

    def run_generate(data, max_new_tokens, batch_size, data_unfinished):

        # 按照 prompt 长度排序，减少 padding
        data.sort(key=lambda x: len(x["input_ids_gt"]))

        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            B = len(batch)

            prompt_len = max(len(ex["input_ids_gt"]) for ex in batch)
            ids_gt = torch.full((B, prompt_len), tokenizer.pad_token_id,
                                dtype=torch.long, device=device)
            ids_model = ids_gt.clone()
            attention_mask = torch.zeros_like(ids_gt, dtype=torch.long, device=device)

            for i, ex in enumerate(batch):
                seq_len = len(ex["input_ids_gt"])
                ids_gt[i, -seq_len:] = torch.tensor(ex["input_ids_gt"])
                ids_model[i, -seq_len:] = torch.tensor(ex["input_ids_model"])
                attention_mask[i, -seq_len:] = 1

            kv_cache = prefill(model, ids_gt, attention_mask)
            
            if args.mask_layers:
                kv_cache_model = prefill(model, ids_model, attention_mask)
                for layer in range(args.mask_layers[0], args.mask_layers[1]):
                    kv_cache[layer][0][:,:,:,:] = kv_cache_model[layer][0][:,:,:,:]
                    kv_cache[layer][1][:,:,:,:] = kv_cache_model[layer][1][:,:,:,:]

            outputs = generate(
                model,
                ids_gt,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                past_key_values=kv_cache
            )

            finished_count = 0

            for i, ex in enumerate(batch):
                response_ids = outputs[i, prompt_len:].cpu()
                eos_idx = (response_ids == tokenizer.eos_token_id).nonzero()
                if len(eos_idx) == 0:
                    if data_unfinished is not None:
                        data_unfinished.append(ex)
                    continue

                eos_idx = eos_idx[0].item()
                response = tokenizer.decode(response_ids[:eos_idx], skip_special_tokens=False).strip()
                pred = last_number_from_text(response)
                ans = ex["answer"]
                is_correct = (pred == ans)

                records.append({
                    "lang": ex["lang"],
                    "problem": ex["problem"],
                    "translation": ex["translation"],
                    "problem_en": ex["problem_en"],
                    "response": response,
                    "pred": pred if pred is not None else "",
                    "answer": ans,
                    "is_correct": is_correct,
                })

                finished_count += 1

            with progress.get_lock():
                progress.value += B if data_unfinished is None else finished_count

    data_unfinished = []
    run_generate(data, args.max_tokens // 2, args.batch_size * 2, data_unfinished)
    run_generate(data_unfinished, args.max_tokens, args.batch_size, None)

    return_dict[rank] = records


# ===================== tokenize helpers =====================
def build_dual_input_ids(ex, tokenizer):
    # ----- strings -----
    m_problem = M_QUERY_PROMPT.format(
        language=LANGUAGE[ex["lang"]],
        problem=ex["problem"],
    )

    user_prompt = m_problem + SPECIAL_TOKEN + INSTRUCTION
    chat_prompt = build_chat_prompt(tokenizer, user_prompt)
    cut= chat_prompt.find(SPECIAL_TOKEN)
    prefix = chat_prompt[:cut]
    suffix = chat_prompt[cut + len(SPECIAL_TOKEN):]

    enc_prefix = tokenizer(prefix)["input_ids"]
    enc_suffix = tokenizer(suffix)["input_ids"]
    translation_gt = EN_QUERY_PROMPT.format(translation=ex["problem_en"])
    translation_model = EN_QUERY_PROMPT.format(translation=ex["translation"])
    enc_translation_gt = tokenizer(translation_gt)["input_ids"]
    enc_translation_model = tokenizer(translation_model)["input_ids"]
    space_id = tokenizer("                ")["input_ids"][0]

    # 检查 decode 结果
    # print(f"Prefix decodes to: '{tokenizer.decode(enc_prefix)}'")
    # print(f"Suffix decodes to: '{tokenizer.decode(enc_suffix)}'")
    # print(f"Translation GT decodes to: '{tokenizer.decode(enc_translation_gt)}'")
    # print(f"Translation Model decodes to: '{tokenizer.decode(enc_translation_model)}'")
    # print(f"Space ID {space_id} decodes to: '{tokenizer.decode([space_id])}'")

    len_T_gt = len(enc_translation_gt)
    len_T_model = len(enc_translation_model)

    if len_T_gt < len_T_model:
        enc_translation_gt += [space_id] * (len_T_model - len_T_gt)
    elif len_T_model < len_T_gt:
        enc_translation_model += [space_id] * (len_T_gt - len_T_model)

    input_ids_gt = enc_prefix + enc_translation_gt + enc_suffix
    input_ids_model = enc_prefix + enc_translation_model + enc_suffix

    # 输出两者的完整 decode
    # print(f"GT Input IDs decode to: '{tokenizer.decode(input_ids_gt)}'")
    # print(f"Model Input IDs decode to: '{tokenizer.decode(input_ids_model)}'")
    # exit(0)

    ex["input_ids_gt"] = input_ids_gt
    ex["input_ids_model"] = input_ids_model


# ===================== main =====================
def main():
    args = parse_args()

    data = []
    for lang in langs:
        path = os.path.join(args.data_dir, f"{lang}.jsonl")
        if not os.path.exists(path):
            continue
        lang_data = []
        with open(path, encoding="utf-8") as f:
            for l in f:
                ex = json.loads(l)
                ex["lang"] = lang
                lang_data.append(ex)
        # random.shuffle(lang_data)
        # lang_data = lang_data[:500]
        data.extend(lang_data)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_dir, args.model),
        trust_remote_code=True,
    )

    for ex in tqdm(data, desc="Tokenizing"):
        build_dual_input_ids(ex, tokenizer)

    random.shuffle(data)

    manager = mp.Manager()
    return_dict = manager.dict()
    progress = mp.Value("i", 0)
    procs = []

    for r in range(args.num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(r, args, data[r::args.num_gpus], return_dict, progress),
        )
        p.start()
        procs.append(p)

    with tqdm(total=len(data)) as pbar:
        last = 0
        while last < len(data):
            with progress.get_lock():
                cur = progress.value
            pbar.update(cur - last)
            last = cur
            time.sleep(0.1)

    for p in procs:
        p.join()

    lang_results = {l: [] for l in langs}
    for r in return_dict:
        for rec in return_dict[r]:
            lang_results[rec["lang"]].append(rec)

    save_results(args, langs, lang_results)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
