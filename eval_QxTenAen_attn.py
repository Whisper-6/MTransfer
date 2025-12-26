import os
import json
import csv
import argparse
import math
import multiprocessing as mp
import time
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import last_number_from_text

# ====================== 固定 prompt ======================

language = {
    "bn": "Bengali",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "ru": "Russian",
    "th": "Thai",
}

TRANSLATE_PROMPT = (
    "Problem: {question}\n\n"
    "Translate the problem from {language} to English, Output only the translation without any extra text."
)

SOLVE_PROMPT = (
    "Problem: {question}\n\n"
    "Translation: {translation}\n\n"
    "Solve the above problem and enclose the final number at the end of the response in $\\boxed{{}}$."
)


# ---------------------- worker_process ----------------------
def build_chat_prompt(tokenizer, user_content):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def worker_process(rank, args, rank_lang_data, return_dict, progress):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)
    is_debug = isinstance(progress, list)
    model_path = os.path.join(args.model_dir, args.model)
    
    # ============================================
    from transformers import AutoTokenizer, AutoModelForCausalLM

    BATCH_SIZE = 3  # 减少以节省内存
        
    PROBLEM_MARK = "<|problem|>"
    TRANSLATION_MARK = "<|translation|>"
    GEN_MARK = "<|gen|>"
    special_tokens = {
        "additional_special_tokens": [
            PROBLEM_MARK,
            TRANSLATION_MARK,
            GEN_MARK,
        ]
    }
        
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    hf_tokenizer.add_special_tokens(special_tokens)
    
    SOLVE_PROMPT_MARKED = (
        f"{PROBLEM_MARK}\n"
        "Problem: {question}\n\n"
        f"{TRANSLATION_MARK}\n"
        "Translation: {translation}\n\n"
        f"{GEN_MARK}\n"
        "Solve the above problem and enclose the final number at the end "
        "of the response in $\\boxed{{}}$."
    )
    default_segname = {"problem", "translation", "generation"}
    default_stats = {}
    for seg in default_segname:
        default_stats[seg] = 0.0
        default_stats[f"{seg}_uni"] = 0.0
    # ============================================
    
    llm = LLM(model=model_path, dtype="half")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    worker_results = {}
    print(f"Worker {rank} processing languages: {list(rank_lang_data.keys())}")
    # ====================== 生成阶段 ======================
    all_generated_data = {}
    for lang, data_slice in rank_lang_data.items():
        if not data_slice:
            continue

        question_field = "m_query"

        # ====================== Step 1: 翻译（集中 batch，每题 num_samples 次） ======================
        translate_prompts = []
        translate_meta = []   # (ex_id)

        for ex_id, ex in enumerate(data_slice):
            user_content = TRANSLATE_PROMPT.format(
                question=ex[question_field],
                language=language[lang],
            )
            prompt = build_chat_prompt(tokenizer, user_content)

            # 同一道题，复制 num_samples 份 prompt
            for _ in range(args.num_samples):
                translate_prompts.append(prompt)
                translate_meta.append(ex_id)

        translate_outputs = llm.generate(
            translate_prompts,
            SamplingParams(
                temperature=args.translate_temperature,
                max_tokens=512,
                seed=None,
            ),
            use_tqdm=False,
        )

        translated_questions = [
            out.outputs[0].text.strip()
            for out in translate_outputs
        ]

        # ====================== Step 2: 解题（集中 batch） ======================
        solve_prompts = []
        prompt_meta = []      # ex_id，对应每一次 trial
        per_source_qids = {src: [] for src in args.sources}

        # source → question 映射（一次即可）
        for ex_id, ex in enumerate(data_slice):
            for src in args.sources:
                if src in ex["source"]:
                    per_source_qids[src].append(ex_id)

        for ex_id, tq in zip(translate_meta, translated_questions):
            solve_user_content = SOLVE_PROMPT.format(
                question=data_slice[ex_id][question_field],
                translation=tq
            )
            solve_prompt = build_chat_prompt(tokenizer, solve_user_content)

            solve_prompts.append(solve_prompt)
            prompt_meta.append(ex_id)

        solve_outputs = llm.generate(
            solve_prompts,
            SamplingParams(
                temperature=args.solve_temperature,
                max_tokens=args.max_tokens,
                stop=["<|user|>"],
                seed=None,
            ),
            use_tqdm=False,
        )
        if is_debug:
            print("processing on ", lang, "len", len(data_slice))
        def chunked(iterable, chunk_size):
            for i in range(0, len(iterable), chunk_size):
                yield iterable[i:i + chunk_size]
                
            
        SOLVE_PROMPT_MARKED = (
            f"{PROBLEM_MARK}\n"
            "Problem: {question}\n\n"
            f"{TRANSLATION_MARK}\n"
            "Translation: {translation}\n\n"
            f"{GEN_MARK}\n"
            "Solve the above problem and enclose the final number at the end "
            "of the response in $\\boxed{{}}$."
        )
        
        full_texts = []
        for ex_id, tq, out in zip(translate_meta, translated_questions, solve_outputs):
            solve_user_content = SOLVE_PROMPT_MARKED.format(
                question=data_slice[ex_id][question_field],
                translation=tq
            )
            solve_prompt = build_chat_prompt(hf_tokenizer, solve_user_content)
            gen_text = out.outputs[0].text
            full_texts.append(solve_prompt + gen_text)

        all_generated_data[lang] = {
            'data_slice': data_slice,
            'translated_questions': translated_questions,
            'solve_outputs': solve_outputs,
            'full_texts': full_texts,
            'prompt_meta': prompt_meta,
            'translate_meta': translate_meta,
            'per_source_qids': per_source_qids,
        }

    # 删除 vLLM 以节省内存
    del llm
    del tokenizer
    torch.cuda.empty_cache()

    # 加载 HF model 用于注意力计算
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",  # 必须
    ).eval()
    hf_model.resize_token_embeddings(len(hf_tokenizer))
    print(f"Worker {rank} loaded HF model for attention computation.")
    # ====================== 注意力计算阶段 ======================
    for lang, gen_data in all_generated_data.items():
        data_slice = gen_data['data_slice']
        translated_questions = gen_data['translated_questions']
        solve_outputs = gen_data['solve_outputs']
        full_texts = gen_data['full_texts']
        prompt_meta = gen_data['prompt_meta']
        translate_meta = gen_data['translate_meta']
        per_source_qids = gen_data['per_source_qids']
        print(f"Worker {rank} computing attention for language: {lang}, num examples: {len(data_slice)}")
        # ====================== Step 3: 计算 attn ======================
        def chunked(iterable, chunk_size):
            for i in range(0, len(iterable), chunk_size):
                yield iterable[i:i + chunk_size]
                
                
        def find_marker_positions(tokenizer, full_text):
            ids = tokenizer(
                full_text,
                add_special_tokens=False
            )["input_ids"]

            def find(marker):
                marker_id = tokenizer(marker, add_special_tokens=False)["input_ids"]
                for i in range(len(ids)):
                    if ids[i:i+len(marker_id)] == marker_id:
                        return i + len(marker_id)
                raise ValueError(f"Marker {marker} not found")

            p = find(PROBLEM_MARK)
            t = find(TRANSLATION_MARK)
            g = find(GEN_MARK)

            return {
                "problem": (p, t),
                "translation": (t, g),
                "generation": (g, len(ids)),
                "full_ids": (0, ids),
            }
            
        def compute_attention_batch(model, tokenizer, texts):
            """
            texts: List[str]
            return: attn tensor (L, B, H, T, T)
            """
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True,
                    use_cache=False,
                )

            # outputs.attentions: tuple of (B, H, T, T)
            attn = torch.stack(outputs.attentions)  # (L, B, H, T, T)

            return attn, inputs["attention_mask"]

        def compute_region_attention_per_layer(
            attn,               # (L, B, H, T, T)
            attention_mask,     # (B, T)
            regions_list        # list of region dicts
        ):
            """
            return:
            results[b][layer] = {
                "problem": float,
                "translation": float,
                "generation": float,
            }
            """
            L, B, H, T, _ = attn.shape
            results = []

            # 先对 head 平均
            # (L, B, T, T)
            attn_layer = attn.mean(dim=2)

            for b in range(B):
                regions = regions_list[b]
                g_start, g_end = regions["generation"]
                valid_len = attention_mask[b].sum().item()

                per_layer_stats = []

                for l in range(L):
                    stats = default_stats.copy()

                    layer_attn = attn_layer[l, b]  # (T, T)

                    for tgt in range(g_start, g_end):
                        if tgt >= valid_len:
                            continue

                        row = layer_attn[tgt, :valid_len]
                        total = row.sum().item()
                        if total == 0:
                            continue

                        for name, (s, e) in regions.items():
                            if name == "full_ids":
                                continue
                            S = row[s:e].sum().item() / total
                            stats[name] += S
                            stats[f"{name}_uni"] += S / (e - s + 1)

                    num_gen = max(1, g_end - g_start)
                    for k in stats:
                        stats[k] /= num_gen

                    per_layer_stats.append(stats)

                results.append(per_layer_stats)

            return results # shape (B, L, dict)
        
        layerwise_results = []
        iteration = 0
        for full_text_batch in chunked(full_texts, BATCH_SIZE):
            regions_list = [
                find_marker_positions(hf_tokenizer, text)
                for text in full_text_batch
            ]
            if is_debug:
                print(f"computing attention batch {iteration}, batch size {len(full_text_batch)}")
            iteration += 1
            attn, attention_mask = compute_attention_batch(
                hf_model,
                hf_tokenizer,
                full_text_batch
            )
            batch_layerwise = compute_region_attention_per_layer(
                attn,
                attention_mask,
                regions_list
            )
            layerwise_results.extend(batch_layerwise)
            if iteration % 10 == 0 or iteration == math.ceil(len(full_texts) / BATCH_SIZE):
                print(f"Worker {rank} attention batches computed: {iteration}/{math.ceil(len(full_texts) / BATCH_SIZE)}")
                print(f"Mem allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")

            del attn, attention_mask
            if is_debug:
                print(f"attention batch computed: {batch_layerwise}")
                print(f"Mem allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
            

        # ---------- 更新进度 ----------
        try:
            if isinstance(progress, list):
                progress[0] += 1
            else:
                with progress.get_lock():
                    progress.value += 1
        except Exception:
            pass

        # ====================== 统计 ======================
        per_example_correct = [0] * len(data_slice)
        all_records = []

        L = len(layerwise_results[0])
        aggregated = [[
            {"problem": 0.0, "translation": 0.0, "generation": 0.0}
            for _ in range(L)
        ] for _ in range(len(data_slice))]
        
        for ex_id, tq, out, lay_attn in zip(prompt_meta, translated_questions, solve_outputs, layerwise_results):
            ex = data_slice[ex_id]
            text = out.outputs[0].text.strip()
            pred = last_number_from_text(text)
            ans = ex["answer"]

            is_correct = (pred == ans)
            per_example_correct[ex_id] += int(is_correct)
            if is_debug:
                print(aggregated[ex_id], lay_attn)
            for l in range(L):
                for k, _ in aggregated[ex_id][l].items():
                        aggregated[ex_id][l][k] += lay_attn[l][k]

            record = {
                "language": lang,
                "source": ex["source"],
                "translated_question": tq,
                "full_response": text,
                "pred_number": pred if pred is not None else "",
                "answer": ans if ans is not None else "",
                "is_correct": int(is_correct),
                "attention_dist": lay_attn,
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
            aggregated_src = [
                {"problem": 0.0, "translation": 0.0, "generation": 0.0}
                for _ in range(L)
            ]
            
            for i in ids:
                for l in range(L):
                    for key, _ in aggregated_src[l].items():
                        aggregated_src[l][key] += aggregated[i][l][key]
            
            source_stats[src] = {
                "num_questions": len(ids),
                "total_correct": correct_src,
                "total_trials": total_src,
                "attention_dist": aggregated_src,
            }

        worker_results[lang] = {
            "num_questions": N,
            "total_correct": total_correct,
            "total_trials": total_trials,
            "sources": source_stats,
            "all_records": all_records,
        }

    return_dict[rank] = worker_results
    del hf_model
    torch.cuda.empty_cache()

# ---------------------- main ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default="eval_data/mmath")
    parser.add_argument("--model-dir", default="/root/autodl-tmp/local_model")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--solve-temperature", type=float, default=0.3)
    parser.add_argument("--translate-temperature", type=float, default=0.3)

    args = parser.parse_args()
    args.sources = ["mgsm", "polymath-low"]

    # 固定 config 名，仅用于路径
    config_name = "QxTenAen-2step-attn_eval"

    if args.output_dir is None:
        args.output_dir = os.path.join("output", args.model, config_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- 加载数据 ----------
    langs = []
    all_lang_data = {}

    for fname in os.listdir(args.data_dir):
        if fname.endswith(".jsonl") and not fname == "en.jsonl":
            lang = fname.replace(".jsonl", "")
            langs.append(lang)
            with open(os.path.join(args.data_dir, fname), encoding="utf-8") as f:
                data = [json.loads(l) for l in f]

            chunk = math.ceil(len(data) / args.num_gpus)
            all_lang_data[lang] = [
                data[i * chunk:(i + 1) * chunk]
                for i in range(args.num_gpus)
            ]

    # ---------- 多进程 ----------
    if args.num_gpus == 1:
        print("----- test mode on single GPU, no multiprocessing. ------")
        return_dict = {}
        progress = [0]
        rank_data = {lang: all_lang_data[lang][0] for lang in all_lang_data}
        worker_process(0, args, rank_data, return_dict, progress)
        total_batches = 1
        with tqdm(total=total_batches, desc="generate batches") as pbar:
            pbar.update(1)
    else:
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        total_batches = sum(
            1
            for lang in all_lang_data
            for r in range(args.num_gpus)
            if all_lang_data[lang][r]
        )

        progress = mp.Value('i', 0)

        for rank in range(args.num_gpus):
            rank_data = {lang: all_lang_data[lang][rank] for lang in all_lang_data}
            p = mp.Process(
                target=worker_process,
                args=(rank, args, rank_data, return_dict, progress),
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

    # ---------- 汇总 & 保存（保持你原逻辑） ----------
    summary_rows = []

    for lang in all_lang_data:
        total_correct = 0
        total_trials = 0
        all_samples = []

        source_agg = {src: {"correct": 0, "total": 0, "attention_dist": None} for src in args.sources}

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
                if source_agg[src]["attention_dist"] is None:
                    source_agg[src]["attention_dist"] = sres["attention_dist"]
                else:
                    for l in range(len(sres["attention_dist"])):
                        for key, _ in sres["attention_dist"][l].items():
                            source_agg[src]["attention_dist"][l][key] += sres["attention_dist"][l][key]

        attn_total = None
        
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
                "attention_dist": json.dumps([
                    {region: value / agg["total"] 
                    for region, value in layer.items()} 
                    for layer in agg["attention_dist"]
                ] if agg["total"] else None),
            })
            if attn_total is None:
                attn_total = agg["attention_dist"]
            else:
                for l in range(len(agg["attention_dist"])):
                    for key, _ in agg["attention_dist"][l].items():
                        attn_total[l][key] += agg["attention_dist"][l][key]

        acc_total = total_correct / total_trials if total_trials else 0.0
        stderr_total = math.sqrt(acc_total * (1 - acc_total) / total_trials) if total_trials else 0.0
        ci_radius_total = 1.96 * stderr_total

        summary_rows.append({
            "language": lang,
            "source": "total",
            "total": total_trials,
            "correct": total_correct,
            "accuracy": round(acc_total, 4),
            "ci_radius": round(ci_radius_total, 4),
            "attention_dist": json.dumps([
                    {region: value / total_trials 
                    for region, value in layer.items()} 
                    for layer in attn_total
                ] if total_trials else None),
        })

        if all_samples:
            out_path = os.path.join(args.output_dir, f"{lang}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "language",
                        "source",
                        "translated_question",
                        "full_response",
                        "pred_number",
                        "answer",
                        "is_correct",
                        "attention_dist",
                    ]
                )
                writer.writeheader()
                writer.writerows(all_samples)

        print(f"✅ {lang}: acc={acc_total:.4f}, ci_radius={ci_radius_total:.4f}")

    out_csv = os.path.join(args.output_dir, "result.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["language", "source", "total", "correct", "accuracy", "ci_radius", "attention_dist"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
