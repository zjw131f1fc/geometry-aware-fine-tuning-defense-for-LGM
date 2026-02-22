#!/bin/bash
# 在 GPU 7,6,5,4,3,2 上并行跑 6 种 trap combo 的完整 pipeline
# 每张卡跑 1 个组合，6 张卡并行
#
# 用法: bash script/sweep_combos.sh

set -e

GPUS=(7 6 5 4 3 2)
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
echo "Sweep: 6 trap combos × 6 GPUs"
echo "Output: ${OUTPUT_ROOT}"
echo "lambda_distill: 3.0, num_target_layers: ${NUM_LAYERS}"
echo "=========================================="

# 分配: GPU0→combo0, GPU1→combo1, ..., GPU5→combo5
run_gpu() {
    local gpu_idx=$1
    local gpu=${GPUS[$gpu_idx]}
    local combo=${COMBOS[$gpu_idx]}

    local tag=$(echo "$combo" | tr ',' '+')
    local log="${OUTPUT_ROOT}/gpu${gpu}.log"

    echo "[GPU ${gpu}] 任务: ${tag}"

    local nl_arg=""
    if [ -n "${NUM_LAYERS}" ]; then
        nl_arg="--num_target_layers ${NUM_LAYERS}"
    fi

    {
        echo "=== GPU ${gpu}: ${tag} ==="
        python script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            --trap_losses "${combo}" \
            ${nl_arg} \
            --tag "sweep_${tag}" \
            --output_dir "${OUTPUT_ROOT}/${tag}"
    } > "${log}" 2>&1 &

    echo "[GPU ${gpu}] PID: $!, log: ${log}"
}

# 启动 6 个后台进程
for i in 0 1 2 3 4 5; do
    run_gpu $i
done

echo ""
echo "6 个后台进程已启动，等待完成..."
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
bt = m.get('baseline_target', {})
pt = m.get('postdefense_target', {})
bs = m.get('baseline_source', {})
ps = m.get('postdefense_source', {})
bl = bt.get('lpips', 0)
pl = pt.get('lpips', 0)
bsp = bs.get('psnr', 0)
psp = ps.get('psnr', 0)
print(f'  Target LPIPS: {bl:.4f} → {pl:.4f} (Δ={pl-bl:+.4f})')
print(f'  Source PSNR:  {bsp:.2f} → {psp:.2f} (Δ={psp-bsp:+.2f})')
"
    else
        echo "--- ${tag} --- (未完成)"
    fi
done
