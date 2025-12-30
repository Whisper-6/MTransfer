import os
import csv
import math

ROOT_DIR = "output/Qwen2.5-1.5B-Instruct/QxTenAen-2step-hf-mask"


def get_acc(correct, total):
    acc = correct / total
    stderr = math.sqrt(acc * (1 - acc) / total)
    ci_radius = 1.96 * stderr
    return acc, ci_radius


def process_summary_csv(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 检查是否已有 all
    languages = {row["language"] for row in rows}
    if "all" in languages:
        return False  # 不需要处理

    total_sum = 0
    correct_sum = 0

    for row in rows:
        total_sum += int(row["total"])
        correct_sum += int(row["correct"])

    acc, ci_radius = get_acc(correct_sum, total_sum)

    summary_rows = rows.copy()
    summary_rows.append({
        "language": "all",
        "total": total_sum,
        "correct": correct_sum,
        "accuracy": round(acc, 6),
        "ci_radius": round(ci_radius, 6),
    })

    # 写回
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["language", "total", "correct", "accuracy", "ci_radius"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return True


def main():
    updated = 0
    for root, _, files in os.walk(ROOT_DIR):
        if "summary.csv" in files:
            csv_path = os.path.join(root, "summary.csv")
            if process_summary_csv(csv_path):
                print(f"[UPDATED] {csv_path}")
                updated += 1

    print(f"\nDone. Updated {updated} summary.csv files.")


if __name__ == "__main__":
    main()
