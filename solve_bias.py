import os
import json
import csv
import math
import argparse
import time
import multiprocessing as mp
import contextlib
import itertools
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from utils import last_number_from_text


# 固定解题 prompt（与原流程保持一致）
SOLVE_PROMPT = (
    "Problem: {question}\n\n"
    "Translation: {translation}\n\n"
    "Solve the above problem in English and enclose the final number at the end of the response in $\\boxed{{}}$."
)


def build_chat_prompt(tokenizer, user_content: str) -> str:
    """构造 chat prompt，兼容 chat 模型模板。"""
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


# ---------------------- Bias Mask Controller ---------------------- #
def build_block_bias(translation_ranges: torch.Tensor, kv_len: int, device, dtype):
    """
    构造加性 bias，阻断：当 query 已在翻译之后且 key 在翻译区间内。
    translation_ranges: [B, 2]，已包含左侧 padding 偏移后的 (start, end)。
    返回形状: [B, 1, kv_len, kv_len]，满足 FlashAttention2 的加性 mask 约定。
    """
    # 预计算位置
    q_pos = torch.arange(kv_len, device=device).view(1, kv_len, 1)  # [1, kv, 1]
    k_pos = torch.arange(kv_len, device=device).view(1, 1, kv_len)  # [1, 1, kv]

    start = translation_ranges[:, 0].view(-1, 1, 1)  # [B,1,1]
    end = translation_ranges[:, 1].view(-1, 1, 1)    # [B,1,1]

    block = (q_pos >= end) & (k_pos >= start) & (k_pos < end)  # [B, kv, kv]

    bias = torch.zeros(
        (translation_ranges.size(0), 1, kv_len, kv_len),
        device=device,
        dtype=dtype,
    )
    bias = bias.masked_fill(block, torch.finfo(dtype).min)
    return bias


class LayerwiseMaskController:
    """
    在前 num_mask_layers 个自注意力层注册 forward_pre_hook，
    为 attention_mask 添加加性 bias，阻止翻译区间后的 token 关注翻译区间内的 token。
    """

    def __init__(self, num_mask_layers: int, translation_ranges, device, debug: bool = False):
        self.num_mask_layers = num_mask_layers
        self.translation_ranges = torch.as_tensor(translation_ranges, device=device)
        self.debug = debug
        self._debug_count = 0

    def _get_past_len(self, past_key_value) -> int:
        """从 past_key_value 中提取已缓存的序列长度"""
        if past_key_value is None:
            return 0
        # 新版 transformers 使用 DynamicCache 对象
        if hasattr(past_key_value, "get_seq_length"):
            return past_key_value.get_seq_length()
        # 旧版 tuple 格式
        if isinstance(past_key_value, (list, tuple)) and len(past_key_value) > 0:
            first = past_key_value[0]
            if first is not None and hasattr(first, "shape"):
                return first.shape[-2]
        return 0

    def _build_translation_bias(self, bsz: int, q_len: int, kv_len: int, past_len: int, device, dtype):
        """
        构造 translation mask bias: 阻止翻译区间后的 query 关注翻译区间内的 key。
        返回 shape: [B, 1, q_len, kv_len]
        """
        start = self.translation_ranges[:, 0].view(-1, 1, 1)  # [B, 1, 1]
        end = self.translation_ranges[:, 1].view(-1, 1, 1)    # [B, 1, 1]

        q_offset = past_len if past_len > 0 else (kv_len - q_len)
        q_pos = torch.arange(q_offset, q_offset + q_len, device=device).view(1, q_len, 1)
        k_pos = torch.arange(kv_len, device=device).view(1, 1, kv_len)

        # mask[b, q, k] = True 表示该位置需要被屏蔽
        mask = (q_pos >= end) & (k_pos >= start) & (k_pos < end)  # [B, q, kv]

        bias = torch.where(
            mask.unsqueeze(1),  # [B, 1, q, kv]
            torch.tensor(torch.finfo(dtype).min, device=device, dtype=dtype),
            torch.tensor(0.0, device=device, dtype=dtype),
        )
        return bias

    def hook(self, module, inputs, kwargs):
        """Forward pre-hook: 注入 translation bias 到 attention_mask"""
        hidden_states = inputs[0] if inputs else kwargs.get("hidden_states")
        if hidden_states is None:
            return None

        bsz, q_len, _ = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype

        attn_mask = kwargs.get("attention_mask")  # [B, 1, q, kv] bfloat16
        past_len = self._get_past_len(kwargs.get("past_key_value"))
        kv_len = (past_len + q_len) if past_len > 0 else (attn_mask.size(-1) if attn_mask is not None else q_len)

        # 构造 translation bias [B, 1, q, kv]
        bias = self._build_translation_bias(bsz, q_len, kv_len, past_len, device, dtype)

        # # 调试：打印 mask 信息
        # if self._debug_count >= 0:
        #     start_vals = self.translation_ranges[:, 0].tolist()
        #     end_vals = self.translation_ranges[:, 1].tolist()
        #     # 统计 mask 中有多少位置被屏蔽
        #     mask_count = (bias < -1e30).sum().item()
        #     total_count = bias.numel()
        #     print(f"[Mask Debug] past_len={past_len}, q_len={q_len}, kv_len={kv_len},dtype ={dtype}")
        #     print(f"  translation_ranges: start={start_vals}, end={end_vals}")
        #     print(f"  bias: shape={tuple(bias.shape)}, masked={mask_count}/{total_count}")
        #     if attn_mask is not None:
        #         print(f"  attn_mask: shape={tuple(attn_mask.shape)}, dtype={attn_mask.dtype}")
        #     else:
        #         print(f"  attn_mask: None")
        #     self._debug_count += 1

        # 合并 mask（始终保留原始 attn_mask 中的 padding 信息）
        if attn_mask is None:
            kwargs["attention_mask"] = bias.contiguous()
        else:
            # 如果 attn_mask 是 bool 类型，需要转换成加性 mask
            # bool mask: True = 可以 attend, False = 不能 attend
            # 加性 mask: 0 = 可以 attend, -inf = 不能 attend
            if attn_mask.dtype == torch.bool:
                min_val = torch.finfo(dtype).min
                attn_mask_additive = torch.where(attn_mask, 0.0, min_val).to(dtype)
            else:
                attn_mask_additive = attn_mask.to(dtype)
            kwargs["attention_mask"] = (attn_mask_additive + bias).contiguous()

        # 调试输出（仅在启用时打印前几次）
        if self.debug and self._debug_count < 6:
            self._log_debug(q_len, kv_len, hidden_states, attn_mask, bias, kwargs["attention_mask"])
            self._debug_count += 1

        return None

    def _log_debug(self, q_len, kv_len, hidden_states, attn_mask, bias, final_mask):
        """输出调试信息"""
        def info(t):
            return f"shape={tuple(t.shape)}, dtype={t.dtype}, contig={t.is_contiguous()}"
        print(f"[MaskDebug] q_len={q_len}, kv_len={kv_len}")
        print(f"  hidden_states: {info(hidden_states)}")
        if attn_mask is not None:
            print(f"  attn_mask_raw: {info(attn_mask)}")
        print(f"  bias: {info(bias)}")
        print(f"  final_mask: {info(final_mask)}")

    @contextlib.contextmanager
    def register_hooks(self, model):
        handles = []
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
        else:
            layers = []

        for idx, layer in enumerate(layers):
            if idx >= self.num_mask_layers:
                break
            if hasattr(layer, "self_attn"):
                h = layer.self_attn.register_forward_pre_hook(
                    self.hook, with_kwargs=True
                )
                handles.append(h)
        try:
            yield
        finally:
            for h in handles:
                h.remove()


# ---------------------- Solve Worker ---------------------- #
def worker_solve(rank, args, rank_lang_data, return_dict, progress):
    """解题操作：使用加性 bias mask + FlashAttention2"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)
    device = "cuda"

    model_path = os.path.join(args.model_dir, args.model)
    print(f"[Rank {rank}] Loading model from {model_path}...")

    # 统一使用 SDPA，mask_layers>0 时通过 sdp_kernel 强制 math 后端
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    print(f"[Rank {rank}] Using SDPA attention")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    worker_results = {}

    for lang, data_slice in rank_lang_data.items():
        if not data_slice:
            continue

        print(
            f"[Rank {rank}] Solving {lang}: {len(data_slice)} samples, mask_layers={args.num_mask_layers}"
        )

        # 聚合题目（source 唯一）
        unique_questions = {}
        for item in data_slice:
            src = item["source"]
            if src not in unique_questions:
                unique_questions[src] = {
                    "original_question": item["original_question"],
                    "answer": item["answer"],
                    "source": item["source"],
                }

        N = len(unique_questions)
        num_samples = len(data_slice) // N if N > 0 else 1

        per_source_qids = {src: [] for src in args.sources}
        for source_id, ex_info in unique_questions.items():
            for src in args.sources:
                if src in ex_info["source"]:
                    per_source_qids[src].append(source_id)

        per_example_correct = {source_id: 0 for source_id in unique_questions}
        all_records = []

        batch_size = getattr(args, "batch_size", 8)
        num_batches = (len(data_slice) + batch_size - 1) // batch_size
        pbar = tqdm(
            total=len(data_slice),
            desc=f"[Rank {rank}] {lang}",
            unit="sample",
            leave=False,
        )

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(data_slice))
            batch_items = data_slice[batch_start:batch_end]

            raw_input_ids = []
            raw_translation_ranges = []
            batch_metadata = []

            # 构造 prompts & 计算翻译区间
            for item in batch_items:
                source_id = item["source"]
                ex_info = unique_questions[source_id]

                solve_user_content = SOLVE_PROMPT.format(
                    question=item["original_question"],
                    translation=item["translation"],
                )
                solve_prompt = build_chat_prompt(tokenizer, solve_user_content)

                # 使用 offset_mapping 精确定位翻译区间，避免 tokenizer 在边界处合并 token 导致 off-by-one
                enc = tokenizer(
                    solve_prompt,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                # 使用 offset_mapping 精确定位翻译区间，避免 tokenizer 在边界处合并 token 导致 off-by-one
                enc = tokenizer(
                    solve_prompt,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                input_ids = enc["input_ids"]
                offsets = enc.get("offset_mapping", None)

                solve_part_start = solve_prompt.find(
                    "\n\nSolve the above problem"
                )
                if offsets:
                    # 起点包含 "Translation:" 标签本身
                    start_char = solve_prompt.index("Translation:")
                    end_char = (
                        solve_part_start
                        if solve_part_start > 0
                        else len(solve_prompt)
                    )
                    translation_start_idx = next(
                        i for i, (s, e) in enumerate(offsets) if e > start_char
                    )
                    translation_end_idx = (
                        max(i for i, (s, e) in enumerate(offsets) if s < end_char)
                        + 1
                    )
                else:
                    # 退回旧逻辑（不依赖 offset_mapping）
                    problem_part = solve_prompt.split("Translation:")[0]
                    problem_ids = tokenizer.encode(
                        problem_part, add_special_tokens=False
                    )
                    translation_start_idx = len(problem_ids)
                    if solve_part_start > 0:
                        pre_solve = solve_prompt[:solve_part_start]
                        pre_solve_ids = tokenizer.encode(
                            pre_solve, add_special_tokens=False
                        )
                        translation_end_idx = len(pre_solve_ids)
                    else:
                        translation_end_idx = len(input_ids)
                input_ids = enc["input_ids"]
                offsets = enc.get("offset_mapping", None)

                solve_part_start = solve_prompt.find(
                    "\n\nSolve the above problem"
                )
                if offsets:
                    # 起点包含 "Translation:" 标签本身
                    start_char = solve_prompt.index("Translation:")
                    end_char = (
                        solve_part_start
                        if solve_part_start > 0
                        else len(solve_prompt)
                    )
                    translation_start_idx = next(
                        i for i, (s, e) in enumerate(offsets) if e > start_char
                    )
                    translation_end_idx = (
                        max(i for i, (s, e) in enumerate(offsets) if s < end_char)
                        + 1
                    )
                else:
                    # 退回旧逻辑（不依赖 offset_mapping）
                    problem_part = solve_prompt.split("Translation:")[0]
                    problem_ids = tokenizer.encode(
                        problem_part, add_special_tokens=False
                    )
                    translation_start_idx = len(problem_ids)
                    if solve_part_start > 0:
                        pre_solve = solve_prompt[:solve_part_start]
                        pre_solve_ids = tokenizer.encode(
                            pre_solve, add_special_tokens=False
                        )
                        translation_end_idx = len(pre_solve_ids)
                    else:
                        translation_end_idx = len(input_ids)

                # 仅首条样例打印翻译区间及上下文，检查是否误遮题干
                if getattr(args, "debug_mask_span", False) and batch_idx == 0 and len(raw_input_ids) == 0:
                    # question_tail 去掉 "Translation:" 标签，方便观察题干末尾
                    if offsets:
                        label_char = solve_prompt.index("Translation:")
                        label_tok = next(i for i, (s, e) in enumerate(offsets) if e > label_char)
                        before_tail = tokenizer.decode(
                            input_ids[:label_tok]
                        )
                    else:
                        before_tail = tokenizer.decode(
                            input_ids[:translation_start_idx]
                        )

                    span = tokenizer.decode(
                        input_ids[translation_start_idx:translation_end_idx]
                    )
                    after_head = tokenizer.decode(
                        input_ids[translation_end_idx:translation_end_idx + 120]
                    )[:120]
                    print(f"[Debug][Rank {rank}] translation span: {span[:200]}...")
                    print(f"[Debug][Rank {rank}] before span tail: {before_tail}")
                    print(f"[Debug][Rank {rank}] after span head: {after_head}")

                # 仅首条样例打印翻译区间及上下文，检查是否误遮题干
                if getattr(args, "debug_mask_span", False) and batch_idx == 0 and len(raw_input_ids) == 0:
                    # question_tail 去掉 "Translation:" 标签，方便观察题干末尾
                    if offsets:
                        label_char = solve_prompt.index("Translation:")
                        label_tok = next(i for i, (s, e) in enumerate(offsets) if e > label_char)
                        before_tail = tokenizer.decode(
                            input_ids[:label_tok]
                        )
                    else:
                        before_tail = tokenizer.decode(
                            input_ids[:translation_start_idx]
                        )

                    span = tokenizer.decode(
                        input_ids[translation_start_idx:translation_end_idx]
                    )
                    after_head = tokenizer.decode(
                        input_ids[translation_end_idx:translation_end_idx + 120]
                    )[:120]
                    print(f"[Debug][Rank {rank}] translation span: {span[:200]}...")
                    print(f"[Debug][Rank {rank}] before span tail: {before_tail}")
                    print(f"[Debug][Rank {rank}] after span head: {after_head}")

                raw_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                raw_translation_ranges.append(
                    (translation_start_idx, translation_end_idx)
                )
                batch_metadata.append(
                    {
                        "source_id": source_id,
                        "ex_info": ex_info,
                        "item": item,
                        "input_len": len(input_ids),
                    }
                )

            # 左侧 padding 对齐并调整区间
            max_input_len = max(len(ids) for ids in raw_input_ids)
            padded_input_ids = []
            adjusted_ranges = []

            for ids, (start, end) in zip(raw_input_ids, raw_translation_ranges):
                pad_len = max_input_len - len(ids)
                if pad_len > 0:
                    padding = torch.full(
                        (pad_len,), tokenizer.pad_token_id, dtype=torch.long
                    )
                    padded_ids = torch.cat([padding, ids], dim=0)
                else:
                    padded_ids = ids
                padded_input_ids.append(padded_ids)
                adjusted_ranges.append((start + pad_len, end + pad_len))
                if getattr(args, "debug_mask_span", False) and batch_idx == 0:
                    print(
                        f"[Debug][Rank {rank}] adjusted range: {(start + pad_len, end + pad_len)}, "
                        f"padded_len: {len(padded_ids)}"
                    )
                if getattr(args, "debug_mask_span", False) and batch_idx == 0:
                    print(
                        f"[Debug][Rank {rank}] adjusted range: {(start + pad_len, end + pad_len)}, "
                        f"padded_len: {len(padded_ids)}"
                    )

            input_tensor = torch.stack(padded_input_ids).to(device)
            attention_mask = (input_tensor != tokenizer.pad_token_id).long()

            # 生成（统一使用 SDPA math 后端，保证 mask_layers=0 和 >0 行为一致）
            try:
                from torch.backends.cuda import sdp_kernel
                with torch.inference_mode(), sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    if args.num_mask_layers > 0:
                        mask_controller = LayerwiseMaskController(
                            num_mask_layers=args.num_mask_layers,
                            translation_ranges=adjusted_ranges,
                            device=device,
                        )
                        with mask_controller.register_hooks(model):
                            outputs = model.generate(
                                input_tensor,
                                attention_mask=attention_mask,
                                max_new_tokens=args.max_tokens,
                                temperature=args.solve_temperature,
                                do_sample=(args.solve_temperature > 0),
                                use_cache=True,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                    else:
                        outputs = model.generate(
                            input_tensor,
                            attention_mask=attention_mask,
                            max_new_tokens=args.max_tokens,
                            temperature=args.solve_temperature,
                            do_sample=(args.solve_temperature > 0),
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                for i, output_ids in enumerate(outputs):
                    meta = batch_metadata[i]
                    generated_part = output_ids[max_input_len:]
                    text = tokenizer.decode(
                        generated_part, skip_special_tokens=True
                    ).strip()

                    pred = last_number_from_text(text)
                    ans = meta["ex_info"]["answer"]
                    is_correct = int(pred == ans)
                    per_example_correct[meta["source_id"]] += is_correct

                    all_records.append(
                        {
                            "language": lang,
                            "source": meta["ex_info"]["source"],
                            "original_question": meta["ex_info"]["original_question"],
                            "translated_question": meta["item"]["translation"],
                            "full_response": text,
                            "pred_number": pred if pred is not None else "",
                            "answer": ans if ans is not None else "",
                            "is_correct": is_correct,
                        }
                    )

            except Exception as e:
                print(f"[Rank {rank}] Batch error: {e}")
                import traceback

                traceback.print_exc()
                for meta in batch_metadata:
                    all_records.append(
                        {
                            "language": lang,
                            "source": meta["ex_info"]["source"],
                            "original_question": meta["ex_info"]["original_question"],
                            "translated_question": meta["item"]["translation"],
                            "full_response": "",
                            "pred_number": "",
                            "answer": meta["ex_info"]["answer"]
                            if meta["ex_info"]["answer"] is not None
                            else "",
                            "is_correct": 0,
                        }
                    )

            pbar.update(len(batch_items))
            if all_records:
                cur_acc = sum(r["is_correct"] for r in all_records) / len(
                    all_records
                )
                pbar.set_postfix({"acc": f"{cur_acc:.3f}"})

        pbar.close()

        # 汇总统计
        k = num_samples
        total_correct = sum(per_example_correct.values())
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
            "all_records": all_records,
        }

    return_dict[rank] = worker_results
    del model
    torch.cuda.empty_cache()

    # 标记该 rank 完成
    try:
        with progress.get_lock():
            progress.value += 1
    except Exception:
        pass


# ---------------------- Main ---------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="模型名称或路径（相对 model_dir）",
    )
    parser.add_argument(
        "--num-mask-layers",
        type=int,
        default=0,
        help="前多少层看不到翻译",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=torch.cuda.device_count(),
    )
    parser.add_argument(
        "--model-dir",
        default="/root/autodl-tmp/local_model",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--solve-temperature",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="solve 阶段批大小（控制显存和吞吐）",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="使用 vLLM 直接解题（不做 mask，仅比对翻译质量/上限表现）",
    )
    parser.add_argument(
        "--samples-per-lang",
        type=int,
        default=-1,
        help="每个 lang 抽取多少条用于 solve (-1 表示不抽取全部测)",
    )
    parser.add_argument(
        "--debug-mask-span",
        action="store_true",
        help="打印首条样例的翻译区间及前后内容，辅助检查 mask 对齐",
    )
    parser.add_argument(
        "--debug-mask-span",
        action="store_true",
        help="打印首条样例的翻译区间及前后内容，辅助检查 mask 对齐",
    )

    args = parser.parse_args()
    args.sources = ["mgsm", "polymath-low"]

    model_name = os.path.basename(args.model.rstrip("/"))

    print("=" * 60)
    if args.use_vllm:
        print("Operation: SOLVE (vLLM, no mask)")
    else:
        print("Operation: SOLVE with bias mask")
    print(f"Model: {args.model}")
    print(f"Model name: {model_name}")
    print(f"Mask layers: {args.num_mask_layers}")
    print(f"Batch size: {args.batch_size}")
    print(f"samples_per_lang: {args.samples_per_lang}")
    print("=" * 60)

    # 翻译结果目录（沿用原流程 tmp/{model_name}）
    translation_dir = os.path.join("tmp", model_name)
    if not os.path.exists(translation_dir):
        print(f"❌ Translation directory not found: {translation_dir}")
        print("   请先运行 translate 流程生成翻译文件。")
        return

    all_lang_data = {}
    langs = []

    for fname in os.listdir(translation_dir):
        if not fname.endswith(".jsonl"):
            continue
        lang = fname.replace(".jsonl", "")
        langs.append(lang)
        with open(os.path.join(translation_dir, fname), encoding="utf-8") as f:
            data = [json.loads(l) for l in f]

        # 如果抽样，每个 lang 抽 samples_per_lang 条
        if args.samples_per_lang > 0 and len(data) > args.samples_per_lang:
            data = random.sample(data, args.samples_per_lang)

        if data:
            num_questions = len(set(item["source"] for item in data))
            num_samples = len(data) // num_questions if num_questions > 0 else 1
            print(
                f"   {lang}: {num_questions} questions × {num_samples} samples = {len(data)} translations"
            )

        chunk = math.ceil(len(data) / args.num_gpus)
        all_lang_data[lang] = [
            data[i * chunk : (i + 1) * chunk] for i in range(args.num_gpus)
        ]

    if not all_lang_data:
        print(f"❌ No translation files found in {translation_dir}")
        return

    # 输出目录
    output_dir = os.path.join(
        "output", model_name, "QxTenAen_mask", f"layer_{args.num_mask_layers}"
    )
    os.makedirs(output_dir, exist_ok=True)

    if args.use_vllm:
        solve_with_vllm(args, all_lang_data, langs, model_name, output_dir)
        return

    # -------- 默认：多进程 + bias mask 路径 --------
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    # 每个有数据的 rank 计 1 个 batch，用于进度
    total_batches = sum(
        1
        for r in range(args.num_gpus)
        if any(all_lang_data[lang][r] for lang in all_lang_data)
    )
    progress = mp.Value("i", 0)

    for rank in range(args.num_gpus):
        rank_data = {lang: all_lang_data[lang][rank] for lang in all_lang_data}
        p = mp.Process(
            target=worker_solve,
            args=(rank, args, rank_data, return_dict, progress),
        )
        p.start()
        processes.append(p)

    if total_batches > 0:
        with tqdm(total=total_batches, desc="Solving") as pbar:
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

    # 汇总并保存
    summary_rows = []

    for lang in langs:
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

        for src in args.sources:
            agg = source_agg[src]
            if agg["total"] == 0:
                continue
            acc = agg["correct"] / agg["total"]
            stderr = math.sqrt(acc * (1 - acc) / agg["total"])
            ci_radius = 1.96 * stderr
            summary_rows.append(
                {
                    "language": lang,
                    "source": src,
                    "total": agg["total"],
                    "correct": agg["correct"],
                    "accuracy": round(acc, 4),
                    "ci_radius": round(ci_radius, 4),
                }
            )

        acc_total = (
            total_correct / total_trials if total_trials else 0.0
        )
        stderr_total = (
            math.sqrt(acc_total * (1 - acc_total) / total_trials)
            if total_trials
            else 0.0
        )
        ci_radius_total = 1.96 * stderr_total
        summary_rows.append(
            {
                "language": lang,
                "source": "total",
                "total": total_trials,
                "correct": total_correct,
                "accuracy": round(acc_total, 4),
                "ci_radius": round(ci_radius_total, 4),
            }
        )

        if all_samples:
            out_path = os.path.join(output_dir, f"{lang}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "language",
                        "source",
                        "original_question",
                        "translated_question",
                        "full_response",
                        "pred_number",
                        "answer",
                        "is_correct",
                    ],
                )
                writer.writeheader()
                writer.writerows(all_samples)

        print(
            f"✅ {lang}: acc={acc_total:.4f}, ci_radius={ci_radius_total:.4f}"
        )

    out_csv = os.path.join(output_dir, "result.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["language", "source", "total", "correct", "accuracy", "ci_radius"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n✅ All results saved to {output_dir}")
    print(
        f"   Masked {args.num_mask_layers} layers from seeing the translation (bias mask)."
    )


# ---------------------- vLLM Worker (no mask) ---------------------- #
def worker_vllm(rank, args, rank_lang_data, return_dict, progress):
    import traceback
    from vllm import LLM, SamplingParams

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        torch.cuda.set_device(0)

        model_path = os.path.join(args.model_dir, args.model)
        print(f"[vLLM-rank{rank}] Loading {model_path}")

        llm = LLM(
            model=model_path,
            dtype="auto",
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        sampling_params = SamplingParams(
            temperature=args.solve_temperature,
            max_tokens=args.max_tokens,
        )

        worker_results = {}

        for lang, data_slice in rank_lang_data.items():
            if not data_slice:
                continue

            unique_questions = {}
            for item in data_slice:
                src = item["source"]
                if src not in unique_questions:
                    unique_questions[src] = {
                        "original_question": item["original_question"],
                        "answer": item["answer"],
                        "source": item["source"],
                    }

            N = len(unique_questions)
            num_samples = len(data_slice) // N if N > 0 else 1

            per_source_qids = {src: [] for src in args.sources}
            for source_id, ex_info in unique_questions.items():
                for src in args.sources:
                    if src in ex_info["source"]:
                        per_source_qids[src].append(source_id)

            per_example_correct = {source_id: 0 for source_id in unique_questions}
            all_records = []

            prompts = []
            metas = []
            for item in data_slice:
                solve_user_content = SOLVE_PROMPT.format(
                    question=item["original_question"],
                    translation=item["translation"],
                )
                solve_prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": solve_user_content},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompts.append(solve_prompt)
                metas.append(item)

            outputs = llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            for out, meta in zip(outputs, metas):
                text = out.outputs[0].text.strip()
                pred = last_number_from_text(text)
                ans = meta["answer"]
                is_correct = int(pred == ans)
                per_example_correct[meta["source"]] += is_correct
                all_records.append(
                    {
                        "language": lang,
                        "source": meta["source"],
                        "original_question": meta["original_question"],
                        "translated_question": meta["translation"],
                        "full_response": text,
                        "pred_number": pred if pred is not None else "",
                        "answer": ans if ans is not None else "",
                        "is_correct": is_correct,
                    }
                )

            k = num_samples
            total_correct = sum(per_example_correct.values())
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
                "all_records": all_records,
            }

        return_dict[rank] = worker_results
    except Exception:
        print(f"[vLLM-rank{rank}] ERROR:")
        traceback.print_exc()
        return_dict[rank] = {}
    finally:
        if "llm" in locals():
            del llm
            torch.cuda.empty_cache()
        try:
            with progress.get_lock():
                progress.value += 1
        except Exception:
            pass


# ---------------------- vLLM Solve (no mask) ---------------------- #
def solve_with_vllm(args, all_lang_data, langs, model_name, output_dir):
    """
    基于相同 prompt/sampling，用 vLLM 直接解题，不做 mask。
    多进程分卡：每个 rank 绑定一张 GPU（和 eval_QxTenAen2.py 类似）。
    """
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    total_batches = sum(
        1
        for r in range(args.num_gpus)
        if any(all_lang_data[lang][r] for lang in all_lang_data)
    )
    progress = mp.Value("i", 0)

    for rank in range(args.num_gpus):
        rank_data = {lang: all_lang_data[lang][rank] for lang in all_lang_data}
        p = mp.Process(
            target=worker_vllm,
            args=(rank, args, rank_data, return_dict, progress),
        )
        p.start()
        processes.append(p)

    if total_batches > 0:
        with tqdm(total=total_batches, desc="[vLLM] solving") as pbar:
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

    # 汇总并保存
    summary_rows = []

    for lang in langs:
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

        for src in args.sources:
            agg = source_agg[src]
            if agg["total"] == 0:
                continue
            acc = agg["correct"] / agg["total"]
            stderr = math.sqrt(acc * (1 - acc) / agg["total"])
            ci_radius = 1.96 * stderr
            summary_rows.append(
                {
                    "language": lang,
                    "source": src,
                    "total": agg["total"],
                    "correct": agg["correct"],
                    "accuracy": round(acc, 4),
                    "ci_radius": round(ci_radius, 4),
                }
            )

        acc_total = total_correct / total_trials if total_trials else 0.0
        stderr_total = (
            math.sqrt(acc_total * (1 - acc_total) / total_trials)
            if total_trials
            else 0.0
        )
        ci_radius_total = 1.96 * stderr_total
        summary_rows.append(
            {
                "language": lang,
                "source": "total",
                "total": total_trials,
                "correct": total_correct,
                "accuracy": round(acc_total, 4),
                "ci_radius": round(ci_radius_total, 4),
            }
        )

        if all_samples:
            out_path = os.path.join(output_dir, f"{lang}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "language",
                        "source",
                        "original_question",
                        "translated_question",
                        "full_response",
                        "pred_number",
                        "answer",
                        "is_correct",
                    ],
                )
                writer.writeheader()
                writer.writerows(all_samples)

        print(f"[vLLM] {lang}: acc={acc_total:.4f}, ci_radius={ci_radius_total:.4f}")

    out_csv = os.path.join(output_dir, "result.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["language", "source", "total", "correct", "accuracy", "ci_radius"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n✅ [vLLM] All results saved to {output_dir}")



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

