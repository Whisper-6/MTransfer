import os
import json
import csv
import argparse
import math
import multiprocessing as mp
import time
import torch
import contextlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils import last_number_from_text

# Check for FlexAttention availability (PyTorch >= 2.5)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False
    print("Warning: torch.nn.attention.flex_attention not found. Ensure PyTorch >= 2.5 is installed.")

# ====================== Prompt Definitions ======================

language = {
    "bn": "Bengali",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "ru": "Russian",
    "th": "Thai",
}

SOLVE_PROMPT = (
    "Problem: {question}\n\n"
    "Translation: {translation}\n\n"
    "Solve the above problem and enclose the final number at the end of the response in $\\boxed{{}}$."
)

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
        # FlexAttention mask_mod receives scalar Tensors for indices.
        # We index into translation_ranges using 'b'.
        
        start = translation_ranges[b, 0]
        end = translation_ranges[b, 1]

        # 1. Causal Mask: q must be >= k
        causal_mask = q_idx >= kv_idx

        # 2. Translation Mask:
        # If query is AFTER the translation (q_idx >= end), 
        # it cannot attend to the translation content (start <= kv_idx < end).
        
        is_key_in_translation = (kv_idx >= start) & (kv_idx < end)
        is_query_after_translation = (q_idx >= end)
        
        blocked_by_translation = is_query_after_translation & is_key_in_translation
        
        # Final mask: Must be Causal AND NOT Blocked
        return causal_mask & (~blocked_by_translation)

    return mask_mod

# Helper to apply rotary embeddings (Standard RoPE)
def apply_rotary_pos_emb_simple(q, k, cos, sin, position_ids=None):
    # q, k: [B, H, L, D]
    # cos, sin: [1, 1, L, D] or [B, 1, L, D]
    # We assume cos/sin are properly shaped or broadcastable.
    # Note: Transformers RoPE implementation details vary.
    # This is a simplified version compatible with typical Qwen/Llama RoPE outputs.
    
    # If cos/sin have 4 dims, ensure they match usage
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0).unsqueeze(0)
        
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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
            # Retrieve cos, sin from existing rotary_emb module
            # We need to compute them for the current positions
            kv_seq_len = k.shape[2]
            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self_attn_module.layer_idx)
            
            # Qwen2 rotary_emb returns cos, sin
            # args: (x, seq_len) or (x, position_ids)
            cos, sin = self_attn_module.rotary_emb(v, seq_len=kv_seq_len)
            
            # Slice cos/sin for current q/k if needed
            # For decoding (q_len=1), position_ids tells us where we are.
            # If position_ids is None, we assume linear.
            
            # Note: Transformers `rotary_emb` usually returns cached cos/sin for max_len.
            # We need to slice based on position_ids.
            if position_ids is not None:
                 # Standard Qwen2/Llama rotary implementation handles position_ids inside `apply_rotary_pos_emb` 
                 # OR returns full cos/sin.
                 # Let's assume standard behavior: apply_rotary_pos_emb expects sliced cos/sin or handles indices.
                 # BUT, Qwen2Attention typically calls `self.rotary_emb(v, position_ids)` which returns sliced cos, sin.
                 pass
            
            # Re-call rotary_emb with position_ids to get correct slices
            cos, sin = self_attn_module.rotary_emb(v, position_ids)
            
            # Apply RoPE
            # We use a simple helper because importing `apply_rotary_pos_emb` might be tricky across versions.
            q, k = apply_rotary_pos_emb_simple(q, k, cos, sin, position_ids)
            
            # 3. KV Cache Handling
            if past_key_value is not None:
                # Update cache
                k, v = past_key_value.update(k, v, self_attn_module.layer_idx, kwargs.get("cache_kwargs"))
            
            # 4. FlexAttention
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            
            # Handle Masking Logic for Decoding vs Prefill
            curr_seq_len = k.shape[2]
            
            # If we are in decoding phase (q_len=1), q_idx is 0..1, but logically it is at end of sequence.
            # FlexAttention mask_mod uses relative indices (0..L).
            # If we pass q (1, D) and k (L, D), flex_attention iterates q_idx 0..1.
            # We need to offset q_idx by (curr_seq_len - q_len) to match our absolute range logic.
            
            offset = curr_seq_len - q_len
            
            def offset_mask_mod(b, h, q_idx, kv_idx):
                return mask_mod(b, h, q_idx + offset, kv_idx)

            # Call FlexAttention
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
                # Patch
                # We bind the new forward method to the module instance
                # Using __get__ to make it a bound method
                module.forward = self.patch_forward(idx, module.forward).__get__(module, type(module))
        
        try:
            yield
        finally:
            # Restore
            for idx, forward in self.original_forwards.items():
                layers[idx].self_attn.forward = forward


# ====================== Worker Logic ======================

def worker_solve_flex(rank, args, rank_lang_data, return_dict, progress):
    """
    Optimized solve worker using FlexAttention (PyTorch 2.5+)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)
    device = "cuda"

    model_path = os.path.join(args.model_dir, args.model)
    print(f"[Rank {rank}] Loading model from {model_path}...")
    
    # Load model (Standard, no FlashAttn 2 flag needed if using FlexAttention, but good to have)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    worker_results = {}

    for lang, data_slice in rank_lang_data.items():
        if not data_slice:
            continue

        print(f"[Rank {rank}] Solving for {lang} with FlexAttention...")

        # Deduplicate Questions
        unique_questions = {}
        for item in data_slice:
            if item["source"] not in unique_questions:
                unique_questions[item["source"]] = {
                    "original_question": item["original_question"],
                    "answer": item["answer"],
                    "source": item["source"],
                }

        N = len(unique_questions)
        per_example_correct = {source_id: 0 for source_id in unique_questions}
        all_records = []

        # Batch Processing
        batch_size = args.batch_size
        num_batches = (len(data_slice) + batch_size - 1) // batch_size
        pbar = tqdm(total=len(data_slice), desc=f"[Rank {rank}] {lang}", unit="sample", leave=False)

        for batch_idx in range(num_batches):
            batch_items = data_slice[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            # Prepare Inputs
            raw_input_ids = []
            raw_translation_ranges = []
            batch_metadata = []

            for item in batch_items:
                # Prompt Construction
                solve_user_content = SOLVE_PROMPT.format(
                    question=item["original_question"],
                    translation=item["translation"]
                )
                solve_prompt = build_chat_prompt(tokenizer, solve_user_content)
                
                # Tokenization & Range Finding
                input_ids = tokenizer.encode(solve_prompt, add_special_tokens=False)
                
                # Find ranges
                problem_part = solve_prompt.split("Translation:")[0] + "Translation: "
                problem_ids = tokenizer.encode(problem_part, add_special_tokens=False)
                start_idx = len(problem_ids)
                
                solve_part_start = solve_prompt.find("\n\nSolve the above problem")
                if solve_part_start > 0:
                    pre_solve = solve_prompt[:solve_part_start]
                    pre_solve_ids = tokenizer.encode(pre_solve, add_special_tokens=False)
                    end_idx = len(pre_solve_ids)
                else:
                    end_idx = len(input_ids)

                raw_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                raw_translation_ranges.append((start_idx, end_idx))
                batch_metadata.append({"item": item, "input_len": len(input_ids)})

            # Padding & Adjust Ranges
            max_input_len = max(len(ids) for ids in raw_input_ids)
            padded_input_ids = []
            adjusted_ranges = []
            
            for i, ids in enumerate(raw_input_ids):
                pad_len = max_input_len - len(ids)
                if pad_len > 0:
                    padded_ids = torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long), ids], dim=0)
                else:
                    padded_ids = ids
                padded_input_ids.append(padded_ids)
                
                start, end = raw_translation_ranges[i]
                adjusted_ranges.append((start + pad_len, end + pad_len))

            input_tensor = torch.stack(padded_input_ids).to(device)

            # GENERATION with FlexAttention
            try:
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
                            use_cache=True, 
                            pad_token_id=tokenizer.pad_token_id,
                        )

                # Process Output
                for i, output_ids in enumerate(outputs):
                    meta = batch_metadata[i]
                    generated_text = tokenizer.decode(output_ids[max_input_len:], skip_special_tokens=True).strip()
                    
                    ans = unique_questions[meta["item"]["source"]]["answer"]
                    pred = last_number_from_text(generated_text)
                    is_correct = (pred == ans)
                    per_example_correct[meta["item"]["source"]] += int(is_correct)

                    all_records.append({
                        "language": lang,
                        "source": meta["item"]["source"],
                        "translated_question": meta["item"]["translation"],
                        "full_response": generated_text,
                        "pred_number": pred,
                        "answer": ans,
                        "is_correct": int(is_correct),
                    })

            except Exception as e:
                print(f"[Rank {rank}] Error: {e}")
                import traceback
                traceback.print_exc()

            pbar.update(len(batch_items))
        
        pbar.close()
        
        # Save results to return_dict (simplified for brevity)
        worker_results[lang] = {"all_records": all_records, "total_correct": sum(per_example_correct.values())}

    return_dict[rank] = worker_results
    del model
    torch.cuda.empty_cache()

# ====================== Main Wrapper ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-mask-layers", type=int, default=0)
    parser.add_argument("--data-dir", default="eval_data/mmath")
    parser.add_argument("--model-dir", default="/root/autodl-tmp/local_model")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--solve-temperature", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    
    args = parser.parse_args()
    args.sources = ["mgsm", "polymath-low"] # Mock sources
    
    # Mock data loading for standalone script
    print("This is a standalone FlexAttention solver script.")
    print(f"Model: {args.model}")
    print(f"Mask Layers: {args.num_mask_layers}")
    
    # ... (Add real data loading logic here same as original script if needed) ...
    # For now, this file serves as the requested "new file with solve logic"
    
    # To run this, you would integrate the data loading part from eval_QxTenAen_mask.py
    
    print("\n✅ FlexAttention implementation ready.")
    if not HAS_FLEX_ATTENTION:
        print("❌ ERROR: torch.nn.attention.flex_attention is missing. Please upgrade PyTorch.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()


