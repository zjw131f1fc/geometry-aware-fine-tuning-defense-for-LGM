#!/bin/bash
# 主实验：6个类别 × 2种防御方法（naive_unlearning + geotrap）
# 每个pipeline自动产生Undefended（Phase 1）和对应防御方法（Phase 3）的结果
#
# 用法: bash experiments/run_main.sh 0,1,2,3

set -e

if [ $# -eq 0 ]; then
    echo "用法: bash experiments/run_main.sh GPU_LIST"
    echo "示例: bash experiments/run_main.sh 0,1,2,3"
    exit 1
fi

# 解析GPU列表
IFS=',' read -ra GPUS <<< "$1"
NUM_GPUS=${#GPUS[@]}

echo "使用 ${NUM_GPUS} 张GPU: ${GPUS[@]}"

CATEGORIES=(knife broccoli conch garlic durian coconut)
METHODS=(naive_unlearning geotrap)

CONFIG="configs/config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="experiments_output/main_experiment_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
echo "=========================================="
echo "主实验: 6类别 × 2方法 = 12个pipeline"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

# 生成任务列表
TASKS=()
for method in "${METHODS[@]}"; do
    for category in "${CATEGORIES[@]}"; do
        TASKS+=("${category}:${method}")
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo "总任务数: ${TOTAL_TASKS}"
echo ""

# 任务分配函数
run_task() {
    local task_idx=$1
    local gpu_idx=$((task_idx % NUM_GPUS))
    local gpu=${GPUS[$gpu_idx]}
    local task=${TASKS[$task_idx]}

    IFS=':' read -r category method <<< "$task"
    local tag="${category}_${method}"
    local log="${OUTPUT_ROOT}/${tag}.log"

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

    {
        echo "=== GPU ${gpu}: ${tag} ==="
        python script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            --categories "${category}" \
            --defense_method "${method}" \
            --tag "${tag}" \
            --output_dir "${OUTPUT_ROOT}/${tag}"
    } > "${log}" 2>&1 &

    echo "[GPU ${gpu}] PID: $!, log: ${log}"
}

# 启动所有任务（自动分配到GPU）
for i in $(seq 0 $((TOTAL_TASKS-1))); do
    run_task $i

    # 每启动NUM_GPUS个任务后等待，避免同时启动太多
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

# 汇总结果（同时输出到终端和文件）
SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"

{
echo ""
echo "=========================================="
echo "汇总结果"
echo "=========================================="

for category in "${CATEGORIES[@]}"; do
    echo ""
    echo "=== ${category} ==="
    echo ""

    for method in "${METHODS[@]}"; do
        tag="${category}_${method}"
        metrics="${OUTPUT_ROOT}/${tag}/metrics.json"

        if [ -f "$metrics" ]; then
            echo "--- ${method} ---"
            python -c "
import json
with open('${metrics}') as f:
    m = json.load(f)

# Undefended = baseline
bt_base = m.get('baseline_target') or {}
bs_base = m.get('baseline_source') or {}

# 防御方法 = postdefense
bt_def = m.get('postdefense_target') or {}
bs_def = m.get('postdefense_source') or {}

print(f'  Target LPIPS: {bt_base.get(\"lpips\", 0):.4f} → {bt_def.get(\"lpips\", 0):.4f} (Δ={bt_def.get(\"lpips\", 0)-bt_base.get(\"lpips\", 0):+.4f})')
print(f'  Target PSNR:  {bt_base.get(\"psnr\", 0):.2f} → {bt_def.get(\"psnr\", 0):.2f} (Δ={bt_def.get(\"psnr\", 0)-bt_base.get(\"psnr\", 0):+.2f})')
print(f'  Source PSNR:  {bs_base.get(\"psnr\", 0):.2f} → {bs_def.get(\"psnr\", 0):.2f} (Δ={bs_def.get(\"psnr\", 0)-bs_base.get(\"psnr\", 0):+.2f})')
print(f'  Source LPIPS: {bs_base.get(\"lpips\", 0):.4f} → {bs_def.get(\"lpips\", 0):.4f} (Δ={bs_def.get(\"lpips\", 0)-bs_base.get(\"lpips\", 0):+.4f})')
"
        else
            echo "--- ${method} --- (未完成)"
        fi
    done

    # 打印Undefended（使用第一个方法的baseline）
    first_method="${METHODS[0]}"
    tag="${category}_${first_method}"
    metrics="${OUTPUT_ROOT}/${tag}/metrics.json"

    if [ -f "$metrics" ]; then
        echo ""
        echo "--- Undefended (baseline) ---"
        python -c "
import json
with open('${metrics}') as f:
    m = json.load(f)
bt = m.get('baseline_target') or {}
bs = m.get('baseline_source') or {}
print(f'  Target LPIPS: {bt.get(\"lpips\", 0):.4f}')
print(f'  Target PSNR:  {bt.get(\"psnr\", 0):.2f}')
print(f'  Source PSNR:  {bs.get(\"psnr\", 0):.2f}')
print(f'  Source LPIPS: {bs.get(\"lpips\", 0):.4f}')
"
    fi
done

echo ""
echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"
