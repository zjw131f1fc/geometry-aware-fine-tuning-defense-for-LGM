#!/bin/bash
# 防御训练时长消融：不同 defense_epochs 对攻击抵抗力的影响
# 测试类别=coconut, 攻击配置固定 (全量微调, AdamW lr=5e-5, 2 epochs)
#
# 用法: bash experiments/run_ablation_defense_epochs.sh GPU_LIST
# 示例: bash experiments/run_ablation_defense_epochs.sh 0,1,2,3

set -e

if [ $# -eq 0 ]; then
    echo "用法: bash experiments/run_ablation_defense_epochs.sh GPU_LIST"
    echo "示例: bash experiments/run_ablation_defense_epochs.sh 0,1,2,3"
    exit 1
fi

# 解析GPU列表
IFS=',' read -ra GPUS <<< "$1"
NUM_GPUS=${#GPUS[@]}

echo "使用 ${NUM_GPUS} 张GPU: ${GPUS[@]}"

CONFIG="configs/config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="experiments_output/ablation_defense_epochs_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
echo "=========================================="
echo "防御训练时长消融"
echo "测试类别: coconut"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

TEST_CAT="coconut"
DEFENSE_EPOCHS=(4 8 12 16 20 24 28)

# ============================================================================
# 任务列表
# ============================================================================

TASKS=()
for ep in "${DEFENSE_EPOCHS[@]}"; do
    TASKS+=("def_ep${ep}:--categories ${TEST_CAT} --defense_epochs ${ep}")
done

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "总任务数: ${TOTAL_TASKS}"
for ep in "${DEFENSE_EPOCHS[@]}"; do
    echo "  - defense_epochs=${ep}"
done
echo ""

# ============================================================================
# 任务执行函数
# ============================================================================

run_task() {
    local task_idx=$1
    local gpu_idx=$((task_idx % NUM_GPUS))
    local gpu=${GPUS[$gpu_idx]}
    local task=${TASKS[$task_idx]}

    IFS=':' read -r tag params <<< "$task"

    local log="${OUTPUT_ROOT}/${tag}.log"
    local output_dir="${OUTPUT_ROOT}/${tag}"

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

    {
        echo "=== GPU ${gpu}: ${tag} ==="
        echo "Params: ${params}"
        echo ""

        XFORMERS_DISABLED=1 python script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            ${params} \
            --tag "${tag}" \
            --output_dir "${output_dir}"
    } > "${log}" 2>&1 &

    echo "[GPU ${gpu}] PID: $!, log: ${log}"
}

# ============================================================================
# 启动所有任务
# ============================================================================

for i in $(seq 0 $((TOTAL_TASKS-1))); do
    run_task $i

    if [ $(((i+1) % NUM_GPUS)) -eq 0 ] && [ $((i+1)) -lt ${TOTAL_TASKS} ]; then
        echo ""
        echo "已启动 $((i+1)) 个任务，等待当前批次完成..."
        wait
        echo "当前批次完成，继续启动..."
        echo ""
    fi
done

echo ""
echo "所有任务已启动，等待完成..."
echo "查看进度: tail -f ${OUTPUT_ROOT}/*.log"
wait
echo ""
echo "全部完成！"

# ============================================================================
# 汇总结果
# ============================================================================

SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"

{
echo ""
echo "=========================================="
echo "防御训练时长消融结果汇总"
echo "=========================================="
echo ""

printf "%-15s %-15s %-15s %-15s %-15s\n" \
    "defense_epochs" "Target LPIPS↑" "Target PSNR↓" "Source PSNR↑" "Source LPIPS↓"
echo "------------------------------------------------------------------------"

for task in "${TASKS[@]}"; do
    IFS=':' read -r tag params <<< "$task"

    metrics="${OUTPUT_ROOT}/${tag}/metrics.json"

    if [ -f "$metrics" ]; then
        python -c "
import json
with open('${metrics}') as f:
    m = json.load(f)

bt = m.get('postdefense_target') or m.get('baseline_target') or {}
bs = m.get('postdefense_source') or m.get('baseline_source') or {}

print(f'${tag:<15s} {bt.get(\"lpips\", 0):>13.4f}   {bt.get(\"psnr\", 0):>13.2f}   {bs.get(\"psnr\", 0):>13.2f}   {bs.get(\"lpips\", 0):>13.4f}')
"
    else
        printf "%-15s (未完成或失败)\n" "${tag}"
    fi
done

echo ""
echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

echo ""
echo "全部完成！"
