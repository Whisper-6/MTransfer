import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

model_path = "/root/autodl-tmp/local_model/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.to("cuda")
model.eval()

# -------------------- 初始化 tokenizer --------------------
tokenizer.padding_side = "left"

# -------------------- 准备不同的前缀 --------------------
prefixes = ["You are a helpful assistant. ", "You are a helpful assistant, too. "]
inputs_prefixes = tokenizer(prefixes, return_tensors="pt", padding=True).to("cuda")

# -------------------- 初始化 StaticCache --------------------
batch_size = len(prefixes)
prompt_cache = StaticCache(config=model.config, max_batch_size=batch_size, max_cache_len=1024, device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    prompt_cache = model(**inputs_prefixes, past_key_values=prompt_cache).past_key_values

# -------------------- 追加 prompt --------------------
prompts = ["Help me write a blogpost about travelling.", "What is the capital of France?"]
batch_prompts = [p1 + p2 for p1, p2 in zip(prefixes, prompts)]
inputs_batch = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")

# deepcopy cache 给 batch
past_key_values = copy.deepcopy(prompt_cache)

# 生成
outputs = model.generate(**inputs_batch, past_key_values=past_key_values, max_new_tokens=20)

# 解码
responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(responses)
