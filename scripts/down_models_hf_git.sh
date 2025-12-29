cd ~/autodl-tmp/
mkdir -p local_model
cd ./local_model

git lfs clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct --depth 1
rm -rf ./Qwen2.5-0.5B-Instruct/.git
git lfs clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct --depth 1
rm -rf ./Qwen2.5-1.5B-Instruct/.git
git lfs clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct --depth 1
rm -rf ./Qwen2.5-3B-Instruct/.git
git lfs clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct --depth 1
rm -rf ./Qwen2.5-7B-Instruct/.git