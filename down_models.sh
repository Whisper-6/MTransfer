cd ~/autodl-tmp/
mkdir -p local_model
cd ./local_model
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download \
    --resume-download \
    --local-dir ./Qwen2.5-0.5B-Instruct \
    --local-dir-use-symlinks False \
    Qwen/Qwen2.5-0.5B-Instruct
huggingface-cli download \
    --resume-download \
    --local-dir ./Qwen2.5-1.5B-Instruct \
    --local-dir-use-symlinks False \
    Qwen/Qwen2.5-1.5B-Instruct
huggingface-cli download \
    --resume-download \
    --local-dir ./Qwen2.5-3B-Instruct \
    --local-dir-use-symlinks False \
    Qwen/Qwen2.5-3B-Instruct
huggingface-cli download \
    --resume-download \
    --local-dir ./Qwen3-8B \
    --local-dir-use-symlinks False \
    Qwen/Qwen3-8B
huggingface-cli download \
    --resume-download \
    --local-dir ./Qwen3-4B \
    --local-dir-use-symlinks False \
    Qwen/Qwen3-4B
huggingface-cli download \
    --resume-download \
    --local-dir ./Qwen3-4B \
    --local-dir-use-symlinks False \
    Qwen/Qwen3-4B