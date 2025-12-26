import re

# ----------------------
# 数字转换函数
# ----------------------
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