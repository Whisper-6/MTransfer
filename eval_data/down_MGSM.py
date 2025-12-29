import os
import json
from huggingface_hub import snapshot_download
from tqdm import tqdm

REPO_ID = "juletxara/mgsm"
LANGS = ["bn", "de", "es", "fr", "ja", "ru", "th", "en"]
OUTPUT_DIR = "mgsm_raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("📥 Downloading MGSM repository snapshot...")
    repo_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"✓ Repository downloaded to: {repo_dir}")

    for lang in LANGS:
        print(f"\n=== Processing language: {lang} ===")

        tsv_path = os.path.join(repo_dir, f"mgsm_{lang}.tsv")
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"File not found: {tsv_path}")

        out_path = os.path.join(
            OUTPUT_DIR, f"{lang}.jsonl"
        )

        count = 0
        with open(tsv_path, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:

            for idx, line in enumerate(tqdm(fin)):
                line = line.rstrip("\n")
                if not line:
                    continue

                # ⚠️ 用 rsplit，防止题面里出现 tab
                try:
                    question, answer = line.rsplit("\t", 1)
                except ValueError:
                    raise ValueError(
                        f"Line {idx} in {tsv_path} does not contain a tab."
                    )

                record = {
                    "id": f"mgsm-{lang}-{count}",
                    "question": question.strip(),
                    "answer": answer.strip(),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"✓ {count} samples")
        print(f"💾 Saved to {out_path}")


if __name__ == "__main__":
    main()