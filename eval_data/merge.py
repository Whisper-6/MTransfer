import os
import json

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "mmath")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGS = ["bn", "de", "es", "fr", "ja", "ru", "th"]
EN_LANG = "en"

DATASETS = {
    "mgsm": {"prefix": ""},
    "polymath_low": {"prefix": "polymath-"},
}


def load_jsonl(file_path):
    """读取 jsonl 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def merge_language(lang):
    """合并单个语言"""
    merged = []

    for ds_name, ds_info in DATASETS.items():
        ds_dir = os.path.join(BASE_DIR, ds_name)
        lang_file = os.path.join(ds_dir, f"{lang}.jsonl")
        en_file = os.path.join(ds_dir, "en.jsonl")

        if not os.path.exists(lang_file) or not os.path.exists(en_file):
            continue

        lang_data = load_jsonl(lang_file)
        en_data = load_jsonl(en_file)

        for i in range(min(len(lang_data), len(en_data))):
            ex = lang_data[i]
            merged.append({
                "source": ds_info["prefix"] + ex["id"] if ds_info["prefix"] else ex["id"],
                "query": en_data[i]["question"],
                "m_query": ex["question"],
                "answer": ex["answer"]
            })

    return merged


def main():
    # 小语种
    for lang in LANGS:
        merged = merge_language(lang)
        out_file = os.path.join(OUTPUT_DIR, f"{lang}.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for ex in merged:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"✅ {lang}: merged {len(merged)} samples -> {out_file}")

    # 英文
    en_merged = merge_language(EN_LANG)
    for ex in en_merged:
        ex["query"] = ex["m_query"]
    out_file = os.path.join(OUTPUT_DIR, "en.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in en_merged:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"✅ en: merged {len(en_merged)} samples -> {out_file}")


if __name__ == "__main__":
    main()
