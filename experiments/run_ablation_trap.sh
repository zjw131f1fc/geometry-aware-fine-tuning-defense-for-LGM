#!/bin/bash
# Trap组合消融实验（Section 5.1）：coconut + durian
# 5.1.1 单属性Trap (4种) × 2类别 = 8
# 5.1.2 两两组合Trap (6种) × 2类别 = 12
# 合计: 20个实验
#
# 用法: bash experiments/run_ablation_trap.sh GPU_LIST
# 示例: bash experiments/run_ablation_trap.sh 0,1,2,3

set -e

if [ $# -eq 0 ]; then
    echo "用法: bash experiments/run_ablation_trap.sh GPU_LIST"
    echo "示例: bash experiments/run_ablation_trap.sh 0,1,2,3"
    exit 1
fi

# 解析GPU列表
IFS=',' read -ra GPUS <<< "$1"
NUM_GPUS=${#GPUS[@]}

echo "使用 ${NUM_GPUS} 张GPU: ${GPUS[@]}"

CONFIG="configs/config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="experiments_output/ablation_trap_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
echo "=========================================="
echo "Trap组合消融实验 (Section 5.1)"
echo "测试类别: coconut, durian"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

CATEGORIES=(coconut durian)
ATTACK_EPOCHS=5

# ============================================================================
# 任务列表
# ============================================================================

TASKS=()

# 5.1.1 单属性Trap
for cat in "${CATEGORIES[@]}"; do
    TASKS+=("5.1.1:single_position_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position --attack_epochs ${ATTACK_EPOCHS}")
    TASKS+=("5.1.1:single_scale_${cat}:--categories ${cat} --defense_method geotrap --trap_losses scale --attack_epochs ${ATTACK_EPOCHS}")
    TASKS+=("5.1.1:single_rotation_${cat}:--categories ${cat} --defense_method geotrap --trap_losses rotation --attack_epochs ${ATTACK_EPOCHS}")
    TASKS+=("5.1.1:single_opacity_${cat}:--categories ${cat} --defense_method geotrap --trap_losses opacity --attack_epochs ${ATTACK_EPOCHS}")
done

# 5.1.2 两两组合Trap
for cat in "${CATEGORIES[@]}"; do
    TASKS+=("5.1.2:combo_position_scale_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,scale --attack_epochs ${ATTACK_EPOCHS}")
    TASKS+=("5.1.2:combo_position_rotation_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,rotation --attack_epochs ${ATTACK_EPOCHS}")
    TASKS+=("5.1.2:combo_position_opacity_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,opacity --attack_epochs ${ATTACK_EPOCHS}")
    TASKS+=("5.1.2:combo_scale_rotation_${cat}:--categories ${cat} --defense_method geotrap --trap_losses scale,rotation --attack_epochs ${ATTACK_EPOCHS}")
    TASKS+=("5.1.2:combo_scale_opacity_${cat}:--categories ${cat} --defense_method geotrap --trap_losses scale,opacity --attack_epochs ${ATTACK_EPOCHS}")
    TASKS+=("5.1.2:combo_rotation_opacity_${cat}:--categories ${cat} --defense_method geotrap --trap_losses rotation,opacity --attack_epochs ${ATTACK_EPOCHS}")
done

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "总任务数: ${TOTAL_TASKS}"
echo "  - Section 5.1.1 (单属性): 4 × 2类别 = 8 个任务"
echo "  - Section 5.1.2 (两两组合): 6 × 2类别 = 12 个任务"
echo ""

# ============================================================================
# 任务执行函数
# ============================================================================

run_task() {
    local task_idx=$1
    local gpu_idx=$((task_idx % NUM_GPUS))
    local gpu=${GPUS[$gpu_idx]}
    local task=${TASKS[$task_idx]}

    IFS=':' read -r section tag params <<< "$task"

    local log="${OUTPUT_ROOT}/${section}_${tag}.log"
    local output_dir="${OUTPUT_ROOT}/${section}_${tag}"

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${section} - ${tag}"

    {
        echo "=== GPU ${gpu}: ${section} - ${tag} ==="
        echo "Params: ${params}"
        echo ""

        python script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            ${params} \
            --tag "${section}_${tag}" \
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
echo "Trap组合消融结果汇总 (Section 5.1)"
echo "=========================================="

for section_label in "5.1.1:单属性Trap" "5.1.2:两两组合Trap"; do
    IFS=':' read -r section section_name <<< "$section_label"
    echo ""
    echo "=== Section ${section}: ${section_name} ==="
    echo ""

    for task in "${TASKS[@]}"; do
        IFS=':' read -r tsection tag params <<< "$task"
        [ "$tsection" != "$section" ] && continue

        metrics="${OUTPUT_ROOT}/${tsection}_${tag}/metrics.json"
        echo "--- ${tag} ---"

        if [ -f "$metrics" ]; then
            python -c "
import json
with open('${metrics}') as f:
    m = json.load(f)

bt_base = m.get('baseline_target') or {}
bs_base = m.get('baseline_source') or {}
bt_def = m.get('postdefense_target') or {}
bs_def = m.get('postdefense_source') or {}

print(f'  Target LPIPS: {bt_base.get(\"lpips\", 0):.4f} → {bt_def.get(\"lpips\", 0):.4f} (Δ={bt_def.get(\"lpips\", 0)-bt_base.get(\"lpips\", 0):+.4f})')
print(f'  Target PSNR:  {bt_base.get(\"psnr\", 0):.2f} → {bt_def.get(\"psnr\", 0):.2f} (Δ={bt_def.get(\"psnr\", 0)-bt_base.get(\"psnr\", 0):+.2f})')
print(f'  Source PSNR:  {bs_base.get(\"psnr\", 0):.2f} → {bs_def.get(\"psnr\", 0):.2f} (Δ={bs_def.get(\"psnr\", 0)-bs_base.get(\"psnr\", 0):+.2f})')
print(f'  Source LPIPS: {bs_base.get(\"lpips\", 0):.4f} → {bs_def.get(\"lpips\", 0):.4f} (Δ={bs_def.get(\"lpips\", 0)-bs_base.get(\"lpips\", 0):+.4f})')
"
        else
            echo "  (未完成或失败)"
        fi
        echo ""
    done
done

echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

echo ""
echo "全部完成！"
