import os
import json
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

langs = ["bn", "de", "es", "fr", "ja", "ru", "th"]
# langs = ["bn"]

model_name = "Qwen2.5-1.5B-Instruct"  # 修改为你的模型名称
model_path = f"/root/autodl-tmp/local_model/{model_name}"  # 修改为你的模型目录
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

data_dir = os.path.join("output", model_name, "QxTenAen-2step")  # 修改为你的数据目录

# ---------- 加载数据 ----------
data = []
for lang in langs:
    path = os.path.join(data_dir, f"{lang}.jsonl")
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8") as f:
        lang_data = [json.loads(l) for l in f]
    for ex in lang_data:
        ex["lang"] = lang
        # 删除不需要的字段
        for k in ["pred", "answer", "is_correct"]:
            ex.pop(k, None)
        data.append(ex)

print(f"Total loaded {len(data)} samples from {data_dir}")

# ---------- 计算 token 长度 ----------
points = []

gen_cost_ratio = 3

for ex in data:
    prompt = ex["prompt"]
    response = ex["response"]

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=True)["input_ids"]

    x, y = len(prompt_ids), gen_cost_ratio * len(response_ids)
    if y > 4000:  # 过滤过长 response
        continue
    points.append((x, y))

print(f"Total samples after filtering: {len(points)}")

B = 128  # batch size

print(f"Cost Lower Bound= {sum(x+y for x,y in points)//B}")

# ---------------- 分组算法 1：按总长度排序，每 B 个一组 ----------------
points_sorted = sorted(points, key=lambda p: p[0]+p[1], reverse=True)

batches_1 = [points_sorted[i:i+B] for i in range(0, len(points_sorted), B)]

max_points_1 = []
total_cost_1 = 0
for batch in batches_1:
    max_x = max(p[0] for p in batch)
    max_y = max(p[1] for p in batch)
    max_points_1.append((max_x, max_y))
    total_cost_1 += max_x + max_y

print(f"Algorithm 1: total cost = {total_cost_1}")

# ---------------- 分组算法 2：贪心最小点代价 ----------------
remaining = points.copy()
batches_2 = []

while remaining:
    # 取长度最小的点
    x0, y0 = min(remaining, key=lambda p: p[0]+p[1])
    remaining.remove((x0, y0))

    # 计算代价并排序
    def cost(p):
        return max(p[0]-x0,0) + max(p[1]-y0,0)
    sorted_remain = sorted(remaining, key=cost)

    batch = [(x0,y0)] + sorted_remain[:B-1]
    for p in batch[1:]:
        remaining.remove(p)
    batches_2.append(batch)

max_points_2 = []
total_cost_2 = 0
for batch in batches_2:
    max_x = max(p[0] for p in batch)
    max_y = max(p[1] for p in batch)
    max_points_2.append((max_x, max_y))
    total_cost_2 += max_x + max_y

print(f"Algorithm 2: total cost = {total_cost_2}")

# ---------------- 绘制散点图 ----------------
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.scatter([p[0] for p in points], [p[1] for p in points], alpha=0.5, s=4)
plt.xlabel("Prompt length")
plt.ylabel("Response length")
plt.title("Original points")
plt.grid(True)

plt.subplot(1,3,2)
plt.scatter([p[0] for p in max_points_1], [p[1] for p in max_points_1], alpha=0.5, s=10, color='orange')
plt.xlabel("Max prompt length")
plt.ylabel("Max response length")
plt.title(f"Algorithm 1 batches\nTotal cost={total_cost_1}")
plt.grid(True)

plt.subplot(1,3,3)
plt.scatter([p[0] for p in max_points_2], [p[1] for p in max_points_2], alpha=0.5, s=10, color='green')
plt.xlabel("Max prompt length")
plt.ylabel("Max response length")
plt.title(f"Algorithm 2 batches\nTotal cost={total_cost_2}")
plt.grid(True)

plt.tight_layout()
plt.savefig("batch_scatter.png")
print("Batch scatter plot saved to batch_scatter.png")
