import os
import json
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from tqdm import tqdm

DATASET_NAME = "Qwen/PolyMath"
TARGET_LANGS = ["bn", "de", "es", "fr", "ja", "ru", "th", "en"]
TARGET_DIFFICULTY = "low"
OUTPUT_DIR = "polymath_low"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def main():
    print(f"🔍 Inspecting dataset: {DATASET_NAME}")

    # 1. 判断 language 是否是 subset
    try:
        subsets = get_dataset_config_names(DATASET_NAME)
        print("Available subsets:", subsets)
    except Exception:
        subsets = []

    for lang in TARGET_LANGS:
        print(f"\n=== Processing language: {lang} ===")

        # ---------- Case 1: language 是 subset ----------
        if lang in subsets:
            splits = get_dataset_split_names(DATASET_NAME, lang)
            print("Available splits:", splits)

            if TARGET_DIFFICULTY not in splits:
                print(f"⚠️ split `{TARGET_DIFFICULTY}` not found, skip.")
                continue

            ds = load_dataset(
                DATASET_NAME,
                lang,
                split=TARGET_DIFFICULTY,
            )

        # ---------- Case 2: language 是字段 ----------
        else:
            print("Language is not a subset, loading full dataset...")
            ds = load_dataset(DATASET_NAME, split="train")

            # 自动识别字段名
            cols = ds.column_names
            print("Columns:", cols)

            lang_field = "language" if "language" in cols else "lang"
            diff_field = "difficulty" if "difficulty" in cols else "level"

            ds = ds.filter(
                lambda x: x.get(lang_field) == lang
                and x.get(diff_field) == TARGET_DIFFICULTY
            )

        print(f"✓ {len(ds)} samples selected")

        out_path = os.path.join(
            OUTPUT_DIR, f"{lang}.jsonl"
        )
        save_jsonl(ds, out_path)
        print(f"💾 Saved to {out_path}")


if __name__ == "__main__":
    main()