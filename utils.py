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
    summary_rows = []

    for lang in langs:
        results = lang_results[lang]

        correct = 0
        total = len(results)

        output_path = os.path.join(args.output_dir, f"{lang}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                correct += record["is_correct"]

        acc = correct / total
        stderr = math.sqrt(acc * (1 - acc) / total)
        ci_radius = 1.96 * stderr

        summary_rows.append({
            "language": lang,
            "total": total,
            "correct": correct,
            "accuracy": round(acc, 6),
            "ci_radius": round(ci_radius, 6),
        })

    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "total", "correct", "accuracy", "ci_radius"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)