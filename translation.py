import os
import json
import math
import multiprocessing as mp
import time
import argparse
import sys
import traceback
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from utils import build_chat_prompt

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
    "Problem: {question}\n"
    "Translate the math problem from {language} into English, preserving all details."
    "Output only the translation without any additional text.\n"
    "English Translation:"
)

# ====================== Operation: Translate ======================

def worker_translate(rank, args, rank_lang_data, return_dict, progress):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)

    # Import vllm inside worker to avoid issues in main process or if not needed
    from vllm import LLM, SamplingParams

    model_path = os.path.join(args.model_dir, args.model)
    print(f"[Rank {rank}] Loading model with vLLM from {model_path} on GPU {os.environ['CUDA_VISIBLE_DEVICES']}...")
    
    llm = LLM(model=model_path, dtype="half", gpu_memory_utilization=0.8)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    worker_results = {}

    for lang, data_slice in rank_lang_data.items():
        if not data_slice:
            continue

        question_field = "m_query"

        # 构造翻译 prompts（每题 num_samples 次）
        translate_prompts = []
        translate_meta = []  # 记录原始题目

        for ex in data_slice:
            user_content = TRANSLATE_PROMPT.format(
                question=ex[question_field],
                language=language[lang],
            )
            prompt = build_chat_prompt(tokenizer, user_content)

            for sample_idx in range(args.num_samples):
                translate_prompts.append(prompt)
                translate_meta.append(ex)  # 直接保存原题信息

        # 使用 vLLM 批量翻译
        translate_outputs = llm.generate(
            translate_prompts,
            SamplingParams(
                temperature=args.translate_temperature,
                max_tokens=512,
                seed=None,
            ),
            use_tqdm=False,
        )

        translated_results = [
            out.outputs[0].text.strip()
            for out in translate_outputs
        ]

        # 组织结果：只用 source 作为唯一标识
        lang_translations = []
        for ex, translation in zip(translate_meta, translated_results):
            lang_translations.append({
                "source": ex["source"],  # 唯一标识（如 mgsm-bn-47）
                "original_question": ex[question_field],
                "translation": translation,
                "answer": ex["answer"],
            })

        worker_results[lang] = lang_translations

        with progress.get_lock():
            progress.value += 1


    return_dict[rank] = worker_results
    
    # Cleanup
    del llm
    torch.cuda.empty_cache()


# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--data-dir", default="eval_data/mgsm")
    parser.add_argument("--model-dir", default="/root/autodl-tmp/local_model")
    parser.add_argument("--translate-temperature", type=float, default=0.3)
    
    args = parser.parse_args()
    
    # Check if user provided CUDA_VISIBLE_DEVICES and warn if num-gpus mismatches logic

    print("=" * 60)
    print("Operation: TRANSLATE ONLY")
    print(f"Model: {args.model}")
    print(f"Num samples per question: {args.num_samples}")
    print(f"Num GPUs: {args.num_gpus}")
    print("=" * 60)

    translation_dir = os.path.join("translation", args.model)
    os.makedirs(translation_dir, exist_ok=True)

    langs = []
    all_lang_data = {}

    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return

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

    # 多进程翻译
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    total_batches = sum(
        1 for lang in all_lang_data
        for r in range(args.num_gpus)
        if all_lang_data[lang][r]
    )

    progress = mp.Value('i', 0)

    for rank in range(args.num_gpus):
        rank_data = {lang: all_lang_data[lang][rank] for lang in all_lang_data}
        p = mp.Process(
            target=worker_translate,
            args=(rank, args, rank_data, return_dict, progress),
        )
        p.start()
        processes.append(p)

    if total_batches > 0:
        with tqdm(total=total_batches, desc="Translating") as pbar:
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

    # 保存翻译结果
    for lang in all_lang_data:
        all_translations = []
        for v in return_dict.values():
            if lang in v:
                all_translations.extend(v[lang])

        if all_translations:
            out_path = os.path.join(translation_dir, f"{lang}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for item in all_translations:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            num_questions = len(set(item["source"] for item in all_translations))
            print(f"✅ {lang}: {num_questions} questions × {args.num_samples} samples = {len(all_translations)} translations")
            print(f"   Saved to {out_path}")

    print(f"\n✅ All translations saved to {translation_dir}/")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()