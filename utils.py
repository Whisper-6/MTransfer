import re

# ----------------------
# 数字转换函数
# ----------------------
BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"

def convert_to_arabic_digits(s):
    return ''.join(str(BENGALI_DIGITS.index(c)) if c in BENGALI_DIGITS else c for c in s)

def last_number_from_text(text):
    text = convert_to_arabic_digits(text)
    nums = re.findall(r"[-+]?\d+", text)
    return int(nums[-1]) if nums else None