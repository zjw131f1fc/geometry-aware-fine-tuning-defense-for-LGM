#!/bin/bash
# 在 GPU 5,6,7 上并行跑 6 种 trap combo 的完整 pipeline
# 每张卡跑 2 个组合（串行），3 张卡并行
#
# 用法: bash script/sweep_combos.sh

set -e

GPUS=(5 6 7)
COMBOS=(
    "position,scale"
    "position,opacity"
    "position,rotation"
    "scale,opacity"
    "scale,rotation"
    "opacity,rotation"
)

CONFIG="configs/config.yaml"
NUM_LAYERS=${NUM_LAYERS:-}  # 留空则使用 config.yaml 中的值
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="output/sweep_combos_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
echo "=========================================="
echo "Sweep: 6 trap combos × 3 GPUs"
echo "Output: ${OUTPUT_ROOT}"
echo "lambda_distill: 3.0, num_target_layers: ${NUM_LAYERS}"
echo "=========================================="

# 分配: GPU0→combo0,3  GPU1→combo1,4  GPU2→combo2,5
run_gpu() {
    local gpu_idx=$1
    local gpu=${GPUS[$gpu_idx]}
    local combo1_idx=$((gpu_idx))
    local combo2_idx=$((gpu_idx + 3))
    local combo1=${COMBOS[$combo1_idx]}
    local combo2=${COMBOS[$combo2_idx]}

    local tag1=$(echo "$combo1" | tr ',' '+')
    local tag2=$(echo "$combo2" | tr ',' '+')
    local log="${OUTPUT_ROOT}/gpu${gpu}.log"

    echo "[GPU ${gpu}] 任务: ${tag1}, ${tag2}"

    local nl_arg=""
    if [ -n "${NUM_LAYERS}" ]; then
        nl_arg="--num_target_layers ${NUM_LAYERS}"
    fi

    {
        echo "=== GPU ${gpu}: ${tag1} ==="
        python script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            --trap_losses "${combo1}" \
            ${nl_arg} \
            --tag "sweep_${tag1}" \
            --output_dir "${OUTPUT_ROOT}/${tag1}"

        echo ""
        echo "=== GPU ${gpu}: ${tag2} ==="
        python script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            --trap_losses "${combo2}" \
            ${nl_arg} \
            --tag "sweep_${tag2}" \
            --output_dir "${OUTPUT_ROOT}/${tag2}"
    } > "${log}" 2>&1 &

    echo "[GPU ${gpu}] PID: $!, log: ${log}"
}

# 启动 3 个后台进程
for i in 0 1 2; do
    run_gpu $i
done

echo ""
echo "3 个后台进程已启动，等待完成..."
echo "查看进度: tail -f ${OUTPUT_ROOT}/gpu*.log"
wait
echo ""
echo "全部完成！"

# 汇总结果
echo ""
echo "=========================================="
echo "汇总"
echo "=========================================="
for combo in "${COMBOS[@]}"; do
    tag=$(echo "$combo" | tr ',' '+')
    metrics="${OUTPUT_ROOT}/${tag}/metrics.json"
    if [ -f "$metrics" ]; then
        echo "--- ${tag} ---"
        python -c "
import json
with open('${metrics}') as f:
    m = json.load(f)
b = m['baseline_attack'][-1] if m['baseline_attack'] else {}
p = m['postdefense_attack'][-1] if m['postdefense_attack'] else {}
bl = b.get('loss_lpips', 0)
pl = p.get('loss_lpips', 0)
bs = b.get('source_psnr', 0)
ps = p.get('source_psnr', 0)
print(f'  Target LPIPS: {bl:.4f} → {pl:.4f} (Δ={pl-bl:+.4f})')
print(f'  Source PSNR:  {bs:.2f} → {ps:.2f} (Δ={ps-bs:+.2f})')
"
    else
        echo "--- ${tag} --- (未完成)"
    fi
done
