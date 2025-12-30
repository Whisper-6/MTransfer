import re
import os
import json
import math
import csv

BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"
def convert_to_arabic_digits(s):
    return ''.join(str(BENGALI_DIGITS.index(c)) if c in BENGALI_DIGITS else c for c in s)

def last_number_from_text(text):
    text = convert_to_arabic_digits(text)
    # Prefer the last number inside a \boxed{...} if present (allow decimals)
    boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    number_pattern = r"[-+]?\d+(?:\.\d+)?"
    if boxed_matches:
        last_box = boxed_matches[-1]
        nums_in_box = re.findall(number_pattern, last_box)
        if nums_in_box:
            return int(float(nums_in_box[-1]))
    # Fallback to last number in the whole text (allow decimals)
    nums = re.findall(number_pattern, text)
    try:
        int_num = int(float(nums[-1]))
    except Exception:
        int_num = None
    return int_num

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

def save_results(args, langs, lang_results):
    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []
    def get_acc(correct, total):
        acc = correct / total
        stderr = math.sqrt(acc * (1 - acc) / total)
        ci_radius = 1.96 * stderr
        return acc, ci_radius

    correct_all = 0
    total_all = 0

    for lang in langs:
        results = lang_results[lang]

        correct = 0
        total = len(results)

        output_path = os.path.join(args.output_dir, f"{lang}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                correct += record["is_correct"]

        acc, ci_radius = get_acc(correct, total)

        summary_rows.append({
            "language": lang,
            "total": total,
            "correct": correct,
            "accuracy": round(acc, 6),
            "ci_radius": round(ci_radius, 6),
        })

        correct_all += correct
        total_all += total

    acc_all, ci_radius_all = get_acc(correct_all, total_all)
    summary_rows.append({
        "language": "all",
        "total": total_all,
        "correct": correct_all,
        "accuracy": round(acc_all, 6),
        "ci_radius": round(ci_radius_all, 6),
    })

    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "total", "correct", "accuracy", "ci_radius"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

def build_batches(gen_cost_ratio, data, B, max_mass):
    points = [(ex["gen_start_idx"], gen_cost_ratio * ex["gen_len"]) for ex in data]
    remaining = list(enumerate(points))  # 保存 index
    data_batches = []
    
    while remaining:
        idx0, (x0, y0) = min(remaining, key=lambda t: t[1][0]+t[1][1])
        remaining.remove((idx0, (x0, y0)))
    
        def cost(t):
            _, (x, y) = t
            return max(x-x0,0) + max(y-y0,0)
        
        sorted_remain = sorted(remaining, key=cost)

        batch = [(idx0, (x0, y0))]
        max_x, max_y = x0, y0
        for t in sorted_remain[:B-1]:
            max_x, max_y = max(max_x, t[1][0]), max(max_y, t[1][1])
            if (max_x + max_y) * len(batch) > B * max_mass:
                break
            batch.append(t)

        for t in batch[1:]:
            remaining.remove(t)

        batch_data = [data[idx] for idx, _ in batch]
        data_batches.append(batch_data)

    return data_batches