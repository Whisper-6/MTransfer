cd ~/autodl-tmp/
mkdir -p local_model
cd ./local_model

git lfs clone https://www.modelscope.cn/Qwen/Qwen2.5-0.5B-Instruct.git --depth 1
rm -rf ./Qwen2.5-0.5B-Instruct/.git
git lfs clone https://www.modelscope.cn/Qwen/Qwen2.5-1.5B-Instruct.git --depth 1
rm -rf ./Qwen2.5-1.5B-Instruct/.git
git lfs clone https://www.modelscope.cn/Qwen/Qwen2.5-3B-Instruct.git --depth 1
rm -rf ./Qwen2.5-3B-Instruct/.git
git lfs clone https://www.modelscope.cn/Qwen/Qwen2.5-7B-Instruct.git --depth 1
rm -rf ./Qwen2.5-7B-Instruct/.git