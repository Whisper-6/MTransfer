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
    "Solve the above problem and enclose the final number at the end of the response in $\\boxed{{}}$."
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
    为它们的 attention_mask 添加加性 bias，实现浅层不看翻译。
    """

    def __init__(self, num_mask_layers: int, translation_ranges, device):
        self.num_mask_layers = num_mask_layers
        self.translation_ranges = torch.as_tensor(translation_ranges, device=device)
        self.cache = {}  # kv_len -> bias tensor
        self._debug_once = True

    def _bias_for(self, kv_len: int, device, dtype):
        bias = self.cache.get(kv_len)
        if bias is None:
            bias = build_block_bias(self.translation_ranges, kv_len, device, dtype)
            self.cache[kv_len] = bias
        return bias

    def hook(self, module, inputs, kwargs):
        # 兼容不同调用方式：hidden_states 可能在 args 也可能在 kwargs
        if len(inputs) >= 1:
            hidden_states = inputs[0]
        else:
            hidden_states = kwargs.get("hidden_states", None)
        if hidden_states is None:
            return None  # 无法获取输入，跳过

        bsz, q_len, _ = hidden_states.size()
        attn_mask = kwargs.get("attention_mask", None)
        kv_len = attn_mask.size(-1) if attn_mask is not None else q_len

        # 目标 head 数：优先取模块属性，否则取 config
        head_dim = (
            getattr(module, "num_heads", None)
            or getattr(getattr(module, "config", None), "num_attention_heads", None)
            or 1
        )

        # 直接按当前 q_len/kv_len 生成 bias_q，避免缓存大矩阵
        start = self.translation_ranges[:, 0].view(-1, 1, 1)  # [B,1,1]
        end = self.translation_ranges[:, 1].view(-1, 1, 1)    # [B,1,1]
        q_pos = torch.arange(kv_len - q_len, kv_len, device=hidden_states.device, dtype=start.dtype).view(1, q_len, 1)
        k_pos = torch.arange(kv_len, device=hidden_states.device, dtype=start.dtype).view(1, 1, kv_len)
        mask = (q_pos >= end) & (k_pos >= start) & (k_pos < end)  # [B, q, kv]
        
        # 统一使用 float32 以保证 mask 的数值稳定性，避免 bfloat16/float16 下的精度问题
        dtype = torch.float32
        # float32 下可以使用更激进的负数，确保 softmax 后为 0
        min_val = -1e9

        # 构造 bias [B, 1, q, kv]
        bias_q = torch.zeros(
            (bsz, 1, q_len, kv_len),
            device=hidden_states.device,
            dtype=dtype,
        ).masked_fill(mask.unsqueeze(1), min_val)

        # 确保 bias_q 最后一维连续，且 shape 匹配 head_dim
        # 为了兼容性，我们直接 expand 到完整 shape [B, head_dim, q, kv] 再 contiguous
        bias_q = bias_q.expand(-1, head_dim, -1, -1).contiguous()

        # [Debug] 每次都打印，查看 dtype/shape
        # try:
        #     # 仅打印 layer 0 的信息避免刷屏太快，或者你可以选择全部打印
        #     # 这里简单判断一下 hidden_states 大小是否变了，或者直接打印
        #     layer_idx = getattr(self, "debug_layer_count", 0)
            
        #     # 增加 decode 阶段的显式捕获
        #     if q_len == 1 and layer_idx < 1: # 只打印 decode 阶段第一层的一次
        #          print(f"[MaskDebug][Decode] layer_{layer_idx} q_len=1 bias_q: {bias_q.shape} {bias_q.dtype}")
        #          if attn_mask is not None:
        #              print(f"[MaskDebug][Decode] layer_{layer_idx} attn_mask: {attn_mask.shape} {attn_mask.dtype}")
        #          else:
        #              print(f"[MaskDebug][Decode] layer_{layer_idx} attn_mask is None")
            
        #     # 为了不刷屏太狠，我们只打印前 3 层的信息 (Prefill)
        #     elif layer_idx < 3: 
        #          print(f"[MaskDebug][Prefill] layer_{layer_idx} bias_q: {bias_q.shape} {bias_q.dtype}")
        #          if attn_mask is not None:
        #              print(f"[MaskDebug] layer_{layer_idx} attn_mask: {attn_mask.shape} {attn_mask.dtype}")
        #          else:
        #              print(f"[MaskDebug] layer_{layer_idx} attn_mask is None")
            
        #     # 每个 batch 重置计数器需要在外面控制，这里只是简单自增
        #     # 实际上 hook 是每层调用的，所以 layer_idx 会一直增到 num_layers-1
        #     # 当新的 forward (new token) 来时，我们需要某种方式重置 layer_idx
        #     # 简单做法：利用 q_len 变化或者 kv_len 变化来感知，或者就让它这么流下去，只看最前面的 log
        #     if layer_idx >= self.num_mask_layers - 1:
        #         self.debug_layer_count = 0 # 简易重置，假设下一轮从 layer 0 开始
        #     else:
        #         self.debug_layer_count = layer_idx + 1
        # except Exception:
        #     pass

        if attn_mask is None:
            # 如果没有传入 attention_mask，直接使用我们的 bias
            kwargs["attention_mask"] = bias_q
        else:
            # 如果有传入 attention_mask，需要将其广播到 [B, head_dim, q, kv] 并转为相同 dtype
            # 注意：某些模型传入的 attn_mask 可能是 4D [B, 1, q, kv] 或者是 2D [B, kv]
            
            # 1. 维度扩展
            if attn_mask.dim() == 2: # [B, kv]
                attn_mask_exp = attn_mask[:, None, None, :]
            elif attn_mask.dim() == 3: # [B, q, kv]
                attn_mask_exp = attn_mask[:, None, :, :]
            elif attn_mask.dim() == 4: # [B, 1/H, q, kv]
                attn_mask_exp = attn_mask
            else:
                return None # 未知格式，放弃注入

            try:
                if layer_idx < 3:
                    print(f"[MaskDebug] layer_{layer_idx} attn_mask_exp (pre-cast): {attn_mask_exp.dtype}")
            except:
                pass

            # 2. 类型转换 (确保是加性 mask 而不是 bool)
            if attn_mask_exp.dtype == torch.bool:
                # bool 转 float: False -> 0, True -> min_val (注意 pytorch 定义通常 True 是 mask 掉)
                # 但 transformers 通常: 0保留, 1 mask (对 bool) 或者 0保留, -inf mask (对 float)
                # 安全起见，假设输入已经是 float (causal mask 通常是 float)
                # 如果是 bool: ~mask * min_val ? 
                # 大多数 causal LM 的 attention_mask 传入的是 float (0.0 或 -inf)
                # 这里我们假设它已经是 float，或者强转
                attn_mask_exp = attn_mask_exp.to(dtype)
                # 如果转完全是 0/1，可能需要处理，但一般 causal mask 已经是 -inf/0
            else:
                attn_mask_exp = attn_mask_exp.to(dtype)

            # 3. 广播到 [B, head_dim, q, kv]
            if attn_mask_exp.size(1) != head_dim:
                attn_mask_exp = attn_mask_exp.expand(-1, head_dim, -1, -1)
            
            # 4. 合并 & 确保连续
            # 这一步非常关键：(A + B) 产生的新 tensor 不一定满足 SDPA 的 contiguous 要求
            # 必须显式调用 .contiguous()
            combined_mask = (attn_mask_exp + bias_q).contiguous()
            kwargs["attention_mask"] = combined_mask


        # try:
        #     print(
        #         "[MaskDebug] device",
        #         torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
        #         "hs",
        #         tuple(hidden_states.shape),
        #         "attn_mask",
        #         attn_mask.shape if attn_mask is not None else None,
        #         "attn_mask_exp",
        #         attn_mask_exp.shape if attn_mask is not None else None,
        #         "bias_q",
        #         tuple(bias_q.shape),
        #         "head_dim",
        #         head_dim,
        #     )
        # except Exception:
        #     pass
        return None

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

    # 优先 FlashAttention2
    try:
        # 尝试使用 FlashAttention2，但如果 mask 报错太难搞，可以考虑设为 "eager" 或 "sdpa" (默认)
        # 这里为了避开 strict contiguous 报错，先尝试去掉显式指定，让它自动选择或回退
        # 如果依然报错，可以将 attn_implementation 设为 "eager"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            # attn_implementation="eager", 
        )
        print(f"[Rank {rank}] Using Auto Attention Implementation")
    except Exception as e:
        print(f"[Rank {rank}] Auto loading failed ({e}), fallback to auto")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

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

            input_tensor = torch.stack(padded_input_ids).to(device)
            # Fix for mask_layers=0 drop:
            # When using left padding with model.generate(), we must provide a correct attention_mask.
            # Otherwise, the model might attend to pad tokens or misalign position embeddings.
            attention_mask = (input_tensor != tokenizer.pad_token_id).long()

            # 生成
            try:
                # 当 num_mask_layers=0 时，register_hooks 不会注册任何 hook，
                # 此时只有上面定义的 attention_mask (0/1 mask) 生效。
                # 这就是标准的 generate 行为，不应该导致大幅掉点。
                # 如果掉点，可能是 attention_mask 没传对位置或者模型对 left padding 的处理问题。
                
                mask_controller = LayerwiseMaskController(
                    num_mask_layers=args.num_mask_layers,
                    translation_ranges=adjusted_ranges,
                    device=device,
                )
                with mask_controller.register_hooks(model):
                    with torch.inference_mode():
                        outputs = model.generate(
                            input_tensor,
                            attention_mask=attention_mask, # 显式传入 0/1 mask
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
        default=torch.cuda.EFD(),
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


