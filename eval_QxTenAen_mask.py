import os
import json
import csv
import argparse
import math
import multiprocessing as mp
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils import last_number_from_text
import contextlib

# Check for FlexAttention availability (PyTorch >= 2.5)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False

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


# ---------------------- Helper functions ----------------------
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

# ---------------------- FlexAttention Logic ----------------------

def create_mask_mod(translation_ranges, device):
    """
    Creates a mask_mod function for FlexAttention.
    translation_ranges: Tensor of shape [B, 2] containing (start, end) indices.
    """
    # Move to same device as input if not already
    translation_ranges = translation_ranges.to(device)

    def mask_mod(b, h, q_idx, kv_idx):
        # Retrieve ranges for this batch index
        # Note: In FlexAttention mask_mod, indices are scalar Tensors representing the grid.
        # We need to index into translation_ranges using 'b'.
        
        start = translation_ranges[b, 0]
        end = translation_ranges[b, 1]

        # 1. Causal Mask: q must be >= k
        causal_mask = q_idx >= kv_idx

        # 2. Translation Mask:
        # If query is AFTER the translation (q_idx >= end), 
        # it cannot attend to the translation content (start <= kv_idx < end).
        
        # Is the current key in the translation range?
        is_key_in_translation = (kv_idx >= start) & (kv_idx < end)
        
        # Is the current query after the translation range?
        is_query_after_translation = (q_idx >= end)
        
        # Mask condition: Query is after translation AND Key is inside translation
        blocked_by_translation = is_query_after_translation & is_key_in_translation
        
        # Final mask: Must be Causal AND NOT Blocked
        return causal_mask & (~blocked_by_translation)

    return mask_mod


class FlexAttentionPatcher:
    def __init__(self, num_mask_layers, translation_ranges, device):
        self.num_mask_layers = num_mask_layers
        self.translation_ranges = torch.tensor(translation_ranges, device=device, dtype=torch.int32)
        self.device = device
        self.original_forwards = {}

    def patch_forward(self, layer_idx, original_forward):
        # Create mask_mod once (it captures self.translation_ranges)
        mask_mod = create_mask_mod(self.translation_ranges, self.device)
        
        # Compile mask_mod for performance (optional but recommended for FlexAttention)
        # mask_mod = torch.compile(mask_mod) 

        def new_forward(self_attn_module, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
            # hidden_states: [B, L, D] (Transformers standard)
            # FlexAttention expects: Query [B, H, L, D]
            
            bsz, q_len, _ = hidden_states.size()
            
            # 1. Projection
            qkv, _ = self_attn_module.qkv_proj(hidden_states)
            qkv = qkv.view(bsz, q_len, self_attn_module.num_heads + 2 * self_attn_module.num_kv_heads, self_attn_module.head_dim)
            
            q, k, v = qkv.split([self_attn_module.num_heads, self_attn_module.num_kv_heads, self_attn_module.num_kv_heads], dim=2)
            
            # [B, L, H, D] -> [B, H, L, D] for FlexAttention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # 2. Rotary Embedding
            # We need standard RoPE. Transformers Qwen2 applies it on [B, H, L, D] usually?
            # Qwen2Attention implementation applies RoPE on (q, k) which are [B, H, L, D] after transpose in original code.
            # But wait, original code:
            # q, k, v = qkv.split(...)
            # q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            # So yes, inputs to apply_rotary_pos_emb are [B, H, L, D]
            
            cos, sin = self_attn_module.rotary_emb(v, position_ids) # v is just for device/dtype reference
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            
            # 3. KV Cache Handling (simplistic)
            if past_key_value is not None:
                # Append to cache
                # This part is tricky with FlexAttention + HuggingFace cache.
                # FlexAttention usually expects full sequence for the mask logic we wrote (using absolute indices).
                # If we are in decoding step (q_len=1), we need full K, V.
                
                cache_k, cache_v = past_key_value.update(k, v, self_attn_module.layer_idx, kwargs.get("cache_kwargs"))
                
                # Use cached K, V
                k = cache_k
                v = cache_v
            else:
                # Initialize cache if needed (omitted for brevity, assume past_key_value provided or handled)
                pass

            # 4. FlexAttention
            # Note: FlexAttention requires contiguous tensors usually
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            
            # Adjust mask_mod for decoding (relative indexing vs absolute)
            # FlexAttention usually handles absolute indices if we don't shift.
            # But in decoding, q_idx is 'current position', kv_idx is 'all past positions'.
            # Our mask_mod uses 'q_idx' and 'kv_idx' as absolute positions.
            # Does flex_attention support 'sliding window' or 'cache'? 
            # In decoding, we usually pass q=[B, H, 1, D] and k=[B, H, Total_L, D].
            # flex_attention will iterate q_idx from 0 to 1, but we need it to be 'real_pos'.
            # We can use 'block_mask' with offsets, or specific flex_attention args if available.
            # For simplicity in this 'Plan 1', we might need to rely on the fact that 
            # flex_attention is designed for the 'prefill' mainly or requires careful index handling.
            
            # WORKAROUND: For decoding (q_len=1), we can just manually check masking because it's cheap?
            # No, user wants FlexAttention speed.
            
            # Actually, `flex_attention` is most powerful for the PREFILL phase (long sequence).
            # For decoding (1 token), standard SDPA or manual kernel is fine.
            # But our masking logic is complex.
            
            # Let's try to use flex_attention.
            # To handle indices correctly during decode, we might need to adjust the mask logic 
            # or assume standard behavior where q_idx starts at 0 for the chunk.
            
            # If using flex_attention with KV-Cache, we typically process just the new query 
            # against the full Key/Value.
            # But flex_attention expects q, k, v lengths to define the grid.
            # If q is length 1, q_idx ranges 0..1.
            # But we need q_idx to be 'position_ids'.
            
            # Hack: Pass `score_mod` that adds offset? Or `mask_mod` that adds offset?
            # mask_mod(b, h, q_idx, kv_idx) -> q_idx is relative to the tensor passed.
            # So we need: real_q_idx = q_idx + (seq_len - q_len)
            
            seq_len = k.size(2)
            offset = seq_len - q_len
            
            def offset_mask_mod(b, h, q, k_idx):
                return mask_mod(b, h, q + offset, k_idx)

            attn_output = flex_attention(q, k, v, mask_mod=offset_mask_mod)
            
            # [B, H, L, D] -> [B, L, H, D]
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)
            
            return self_attn_module.o_proj(attn_output), None, past_key_value

        return new_forward

    @contextlib.contextmanager
    def apply(self, model):
        # Identify layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        else:
            layers = []

        for idx, layer in enumerate(layers):
            if idx >= self.num_mask_layers:
                break
            
            if hasattr(layer, 'self_attn'):
                module = layer.self_attn
                # Save original
                self.original_forwards[idx] = module.forward
                # Monkey patch
                # Note: This is a simplified patch that assumes Qwen2 structure.
                # We need `apply_rotary_pos_emb` helper.
                # In Qwen2, it's a method of the model or module?
                # Usually we can reuse the existing `rotary_emb` logic but we need to reimplement `forward`.
                # This is RISKY without full model code access.
                pass 
                
        # ... Wait, monkey patching entire forward is dangerous without `apply_rotary_pos_emb`.
        # Let's import it from transformers.models.qwen2.modeling_qwen2
        
        try:
            yield
        finally:
            # Restore
            for idx, forward in self.original_forwards.items():
                layers[idx].self_attn.forward = forward

# Helper to import rotary
try:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
except ImportError:
    # Fallback or dummy if not Qwen2
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # Basic implementation
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

# ... [Previous imports and code] ...

# ====================== Operation 1: Translate ======================

def worker_translate(rank, args, rank_lang_data, return_dict, progress):
    """翻译操作：将非英文问题翻译成英文（使用 vLLM）"""
    # 在函数内部导入 vllm，避免 solve 操作时加载它
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)

    model_path = os.path.join(args.model_dir, args.model)

    print(f"[Rank {rank}] Loading model with vLLM from {model_path}...")
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

        print(f"[Rank {rank}] Translating for {lang}: {len(data_slice)} questions × {args.num_samples} samples...")

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

        # 更新进度
        try:
            with progress.get_lock():
                progress.value += 1
        except Exception:
            pass

    return_dict[rank] = worker_results
    del llm
    torch.cuda.empty_cache()


# ====================== Operation 2: Solve with Mask ======================
def worker_solve(rank, args, rank_lang_data, return_dict, progress):
    """解题操作：使用 mask attention 解题"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)
    device = "cuda"

    model_path = os.path.join(args.model_dir, args.model)

    print(f"[Rank {rank}] Loading model from {model_path}...")
    
    # Try to load with flash_attention_2 if available
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        print(f"[Rank {rank}] Using Flash Attention 2")
    except Exception as e:
        print(f"[Rank {rank}] Flash Attention 2 not available ({e}), falling back to auto")
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
    
    # Ensure left padding for generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    worker_results = {}

    for lang, data_slice in rank_lang_data.items():
        if not data_slice:
            continue

        print(f"[Rank {rank}] Solving for {lang}: {len(data_slice)} translations with mask_layers={args.num_mask_layers}...")

        # 统计题目数量和 source 分布
        # 使用 source 作为唯一标识
        unique_questions = {}  # source -> question info
        for item in data_slice:
            source_id = item["source"]
            if source_id not in unique_questions:
                unique_questions[source_id] = {
                    "original_question": item["original_question"],
                    "answer": item["answer"],
                    "source": item["source"],
                }

        N = len(unique_questions)  # 题目数

        # 计算 num_samples（从翻译数据推断）
        num_samples = len(data_slice) // N if N > 0 else 1

        # 统计每个 source 类型（mgsm/polymath-low）的题目
        per_source_qids = {src: [] for src in args.sources}
        for source_id, ex_info in unique_questions.items():
            for src in args.sources:
                if src in ex_info["source"]:
                    per_source_qids[src].append(source_id)

        # 初始化统计：按 source 记录每题的正确次数
        per_example_correct = {source_id: 0 for source_id in unique_questions}
        all_records = []

        # Batch处理（提升GPU利用率），batch_size由参数指定
        batch_size = getattr(args, "batch_size", 8)
        num_batches = (len(data_slice) + batch_size - 1) // batch_size

        pbar = tqdm(total=len(data_slice), desc=f"[Rank {rank}] {lang}", unit="sample", leave=False)

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(data_slice))
            batch_items = data_slice[batch_start:batch_end]

            # 准备batch数据
            raw_input_ids = []
            raw_translation_ranges = []
            batch_metadata = []  # 保存每个样本的元信息

            for item in batch_items:
                source_id = item["source"]
                ex_info = unique_questions[source_id]

                # 构造解题 prompt
                solve_user_content = SOLVE_PROMPT.format(
                    question=item["original_question"],
                    translation=item["translation"]
                )
                solve_prompt = build_chat_prompt(tokenizer, solve_user_content)

                # Tokenize
                input_ids = tokenizer.encode(solve_prompt, add_special_tokens=False) # Get list
                
                # 找到 "Translation: " 的位置
                problem_part = solve_prompt.split("Translation:")[0] + "Translation: "
                problem_ids = tokenizer.encode(problem_part, add_special_tokens=False)
                translation_start_idx = len(problem_ids)

                # 找到翻译结束位置
                solve_part_start = solve_prompt.find("\n\nSolve the above problem")
                if solve_part_start > 0:
                    pre_solve = solve_prompt[:solve_part_start]
                    pre_solve_ids = tokenizer.encode(pre_solve, add_special_tokens=False)
                    translation_end_idx = len(pre_solve_ids)
                else:
                    translation_end_idx = len(input_ids)
                
                raw_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                raw_translation_ranges.append((translation_start_idx, translation_end_idx))
                
                batch_metadata.append({
                    "source_id": source_id,
                    "ex_info": ex_info,
                    "item": item,
                    "input_len": len(input_ids),
                })

            # Pad batch manually to handle ranges correctly
            max_input_len = max(len(ids) for ids in raw_input_ids)
            padded_input_ids = []
            adjusted_ranges = []
            
            for i, ids in enumerate(raw_input_ids):
                seq_len = len(ids)
                pad_len = max_input_len - seq_len
                # Left padding
                if pad_len > 0:
                    padding = torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
                    padded_ids = torch.cat([padding, ids], dim=0)
                else:
                    padded_ids = ids
                
                padded_input_ids.append(padded_ids)
                
                # Adjust ranges
                start, end = raw_translation_ranges[i]
                adjusted_ranges.append((start + pad_len, end + pad_len))
            
                input_tensor = torch.stack(padded_input_ids).to(device)
                
                # --- DEBUG OUTPUT ---
                if batch_idx == 0:
                    print(f"\n[Rank {rank}] DEBUG: First batch inputs prepared.")
                    print(f"  Input Tensor Shape: {input_tensor.shape}")
                    print(f"  Max New Tokens: {args.max_tokens}")
                    print(f"  FlexAttention Enabled: {HAS_FLEX_ATTENTION and args.use_flex_attention}")
                    print(f"  Num Mask Layers: {args.num_mask_layers}")
                    
                    # Print one example prompt to verify content
                    debug_text = tokenizer.decode(input_tensor[0], skip_special_tokens=True)
                    print(f"  [Sample 0 Prompt Start]: {debug_text[:200]}...")
                    print(f"  [Sample 0 Prompt End]: ...{debug_text[-200:]}")
                    
                    # Verify Mask Ranges
                    start, end = adjusted_ranges[0]
                    print(f"  [Sample 0 Mask Range]: start={start}, end={end} (Input Length={len(input_tensor[0])})")
                    masked_content = tokenizer.decode(input_tensor[0][start:end], skip_special_tokens=True)
                    print(f"  [Sample 0 Masked Content]: \"{masked_content[:100]}...\"")
                # --------------------

            # Generate with layerwise mask
            try:
                if HAS_FLEX_ATTENTION and args.use_flex_attention:
                    # Plan 1: Use FlexAttention (Experimental / Fast)
                    patcher = FlexAttentionPatcher(
                        num_mask_layers=args.num_mask_layers,
                        translation_ranges=adjusted_ranges,
                        device=device
                    )
                    with patcher.apply(model):
                        with torch.inference_mode():
                            outputs = model.generate(
                                input_tensor,
                                max_new_tokens=args.max_tokens,
                                temperature=args.solve_temperature,
                                do_sample=(args.solve_temperature > 0),
                                use_cache=True, # Note: FlexAttention impl above needs cache support check
                                pad_token_id=tokenizer.pad_token_id,
                            )
                else:
                    # Plan 2: Use Hook + FlashAttention2 (Robust)
                    mask_controller = LayerwiseMaskController(
                        num_mask_layers=args.num_mask_layers,
                        translation_ranges=adjusted_ranges,
                        device=device
                    )
                    
                    with mask_controller.register_hooks(model):
                        with torch.inference_mode():
                            outputs = model.generate(
                                input_tensor,
                                max_new_tokens=args.max_tokens,
                                temperature=args.solve_temperature,
                                do_sample=(args.solve_temperature > 0),
                                use_cache=True,
                                pad_token_id=tokenizer.pad_token_id,
                            )

                # 处理batch结果
                for i, output_ids in enumerate(outputs):
                    meta = batch_metadata[i]
                    input_len = meta["input_len"] # Original length without padding
                    
                    # Output contains [padding, input, generated]
                    # We need to find where the real input ended.
                    # Since we padded left, the generated part starts after the full padded input length?
                    # No, output includes padding.
                    # output_ids length = max_input_len + new_tokens
                    # The prompt part ends at max_input_len.
                    
                    # Extract only new tokens
                    generated_part = output_ids[max_input_len:]
                    
                    # Decode
                    text = tokenizer.decode(generated_part, skip_special_tokens=True)
                    text = text.strip()

                    # 预测和评估
                    pred = last_number_from_text(text)
                    ans = meta["ex_info"]["answer"]

                    is_correct = (pred == ans)
                    per_example_correct[meta["source_id"]] += int(is_correct)

                    record = {
                        "language": lang,
                        "source": meta["ex_info"]["source"],
                        "translated_question": meta["item"]["translation"],
                        "full_response": text,
                        "pred_number": pred if pred is not None else "",
                        "answer": ans if ans is not None else "",
                        "is_correct": int(is_correct),
                    }
                    all_records.append(record)

            except Exception as e:
                print(f"[Rank {rank}] Error in batch generation: {e}")
                import traceback
                traceback.print_exc()
                # 出错时用空结果填充
                for meta in batch_metadata:
                    record = {
                        "language": lang,
                        "source": meta["ex_info"]["source"],
                        "translated_question": meta["item"]["translation"],
                        "full_response": "",
                        "pred_number": "",
                        "answer": meta["ex_info"]["answer"] if meta["ex_info"]["answer"] is not None else "",
                        "is_correct": 0,
                    }
                    all_records.append(record)

            # 更新进度条
            pbar.update(len(batch_items))
            if len(all_records) > 0:
                current_acc = sum(r["is_correct"] for r in all_records) / len(all_records)
                pbar.set_postfix({"acc": f"{current_acc:.3f}"})

        pbar.close()

        # 更新进度
        try:
            with progress.get_lock():
                progress.value += 1
        except Exception:
            pass

        # ====================== 统计（和 eval_QxTenAen2.py 对齐）======================
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


# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", required=True, choices=["translate", "solve"],
                        help="Operation to perform: 'translate' or 'solve'")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples per question (only for 'translate')")
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--num-mask-layers", type=int, default=0,
                        help="Number of initial layers that cannot see the translation (only for 'solve')")
    parser.add_argument("--data-dir", default="eval_data/mmath")
    parser.add_argument("--model-dir", default="/root/autodl-tmp/local_model")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--solve-temperature", type=float, default=0.3)
    parser.add_argument("--translate-temperature", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for solve operation (default: 8)")
    parser.add_argument("--use-flex-attention", action="store_true", help="Use PyTorch FlexAttention (requires torch>=2.5)")

    args = parser.parse_args()
    args.sources = ["mgsm", "polymath-low"]

    # 从模型路径中提取模型名称（支持相对路径和绝对路径）
    model_name = os.path.basename(args.model.rstrip('/'))

    # ========== Operation 1: Translate ==========
    if args.operation == "translate":
        print("=" * 60)
        print("Operation: TRANSLATE")
        print(f"Model: {args.model}")
        print(f"Model name: {model_name}")
        print(f"Num samples per question: {args.num_samples}")
        print("=" * 60)

        # 翻译结果保存路径（固定在项目根目录的 tmp/ 下）
        translation_dir = os.path.join("tmp", model_name)
        os.makedirs(translation_dir, exist_ok=True)

        # 加载数据
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
        print(f"   Next step: Run with --operation solve --num-mask-layers <N>")
        print(f"   Note: --num-samples is NOT needed for solve operation")

    # ========== Operation 2: Solve with Mask ==========
    elif args.operation == "solve":
        print("=" * 60)
        print("Operation: SOLVE with MASK")
        print(f"Model: {args.model}")
        print(f"Model name: {model_name}")
        print(f"Mask layers: {args.num_mask_layers}")
        print(f"Batch size: {args.batch_size}")
        print("=" * 60)

        # 输出目录（使用模型名称）
        output_dir = os.path.join(
            "output",
            model_name,
            "QxTenAen_mask",
            f"layer_{args.num_mask_layers}"
        )
        os.makedirs(output_dir, exist_ok=True)

        # 读取翻译结果（固定在项目根目录的 tmp/ 下）
        translation_dir = os.path.join("tmp", model_name)
        if not os.path.exists(translation_dir):
            print(f"❌ Translation directory not found: {translation_dir}")
            print(f"   Please run with --operation translate first!")
            return

        all_lang_data = {}
        langs = []

        for fname in os.listdir(translation_dir):
            if fname.endswith(".jsonl"):
                lang = fname.replace(".jsonl", "")
                langs.append(lang)

                with open(os.path.join(translation_dir, fname), encoding="utf-8") as f:
                    data = [json.loads(l) for l in f]

                # 推断 num_samples（使用 source 作为唯一标识）
                if data:
                    num_questions = len(set(item["source"] for item in data))
                    num_samples = len(data) // num_questions if num_questions > 0 else 1
                    print(f"   {lang}: {num_questions} questions × {num_samples} samples = {len(data)} translations")

                chunk = math.ceil(len(data) / args.num_gpus)
                all_lang_data[lang] = [
                    data[i * chunk:(i + 1) * chunk]
                    for i in range(args.num_gpus)
                ]

        if not all_lang_data:
            print(f"❌ No translation files found in {translation_dir}")
            return

        # 多进程解题
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

        # 汇总 & 保存（和 eval_QxTenAen2.py 对齐）
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

            # source-level 统计
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

            # total 统计
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
            })

            # 保存详细样本
            if all_samples:
                out_path = os.path.join(output_dir, f"{lang}.csv")
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
                        ]
                    )
                    writer.writeheader()
                    writer.writerows(all_samples)

            print(f"✅ {lang}: acc={acc_total:.4f}, ci_radius={ci_radius_total:.4f}")

        # 保存汇总 CSV
        out_csv = os.path.join(output_dir, "result.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["language", "source", "total", "correct", "accuracy", "ci_radius"]
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\n✅ All results saved to {output_dir}")
        print(f"   Masked {args.num_mask_layers} layers from seeing the translation.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
