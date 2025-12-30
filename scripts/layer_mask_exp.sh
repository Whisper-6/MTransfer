# python translate.py --model Qwen2.5-1.5B-Instruct --num-samples 16 --batch-size 16
echo "Evaluating Baseline"
# python eval_QxTenAen_hf_layer.py --model Qwen2.5-1.5B-Instruct --batch-size 128
echo "Evaluating mask L[0, 28)"
# python eval_QxTenAen_hf_layer.py --model Qwen2.5-1.5B-Instruct --batch-size 128 --mask-layers 0 28

# 前缀遮蔽实验, 间距为 2
for layer_end in $(seq 2 2 26); do
    echo "Evaluating mask L[0, $layer_end)"
    # python eval_QxTenAen_hf_layer.py --model Qwen2.5-1.5B-Instruct --batch-size 128 --mask-layers 0 $layer_end
done

# 后缀遮蔽实验, 间距为 2
for layer_start in $(seq 26 -2 2); do
    echo "Evaluating mask L[$layer_start, 28)"
    # python eval_QxTenAen_hf_layer.py --model Qwen2.5-1.5B-Instruct --batch-size 128 --mask-layers $layer_start 28
done