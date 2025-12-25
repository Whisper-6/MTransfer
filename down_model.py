from modelscope import snapshot_download
from pathlib import Path

# ====== 改成你的数据盘路径 ======
BASE_DIR = Path("/root/autodl-tmp/local_model")
BASE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

for model_id in MODELS:
    model_name = model_id.split("/")[-1]
    local_dir = BASE_DIR / model_name

    print(f"\n===== Downloading {model_id} =====")
    print(f"→ Local dir: {local_dir}")

    snapshot_download(
        model_id=model_id,
        local_dir=str(local_dir)
    )

print("\n✅ All models downloaded.")
