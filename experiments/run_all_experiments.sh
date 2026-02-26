#!/bin/bash
# 聚合实验脚本：防御时长消融 + 互锁机制消融
#
# 实验 1: 防御时长消融 - defense_epochs = 4,8,12,16,20, attack_epochs = 3
# 实验 2: 互锁机制消融 - 每个 defense_epoch 下做 baseline + 3 个 w/o
#
# 用法: bash experiments/run_all_experiments.sh GPU_LIST
# 示例: bash experiments/run_all_experiments.sh 0,1,2,3

set -e

if [ $# -eq 0 ]; then
    echo "用法: bash experiments/run_all_experiments.sh GPU_LIST"
    echo "示例: bash experiments/run_all_experiments.sh 0,1,2,3"
    exit 1
fi

# 解析GPU列表
IFS=',' read -ra GPUS <<< "$1"
NUM_GPUS=${#GPUS[@]}

echo "=========================================="
echo "聚合实验脚本"
echo "使用 ${NUM_GPUS} 张GPU: ${GPUS[@]}"
echo "=========================================="

CONFIG="configs/config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="experiments_output/all_experiments_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"

TEST_CAT="coconut"
ATTACK_EPOCHS=10
DEFENSE_EPOCHS=(10 20 30 40)
TRAP_LOSSES="position,scale"

TASKS=()

# ============================================================================
# 实验 1: 防御时长消融
# ============================================================================

echo ""
echo "=========================================="
echo "实验 1: 防御时长消融"
echo "=========================================="

for def_ep in "${DEFENSE_EPOCHS[@]}"; do
    TASKS+=("exp1:def_ep${def_ep}:--categories ${TEST_CAT} --defense_method geotrap --defense_epochs ${def_ep} --attack_epochs ${ATTACK_EPOCHS}")
done

# ============================================================================
# 实验 2: 互锁机制消融（每个 defense_epoch 下做 4 种配置）
# ============================================================================

echo ""
echo "=========================================="
echo "实验 2: 互锁机制消融"
echo "=========================================="

for def_ep in "${DEFENSE_EPOCHS[@]}"; do
    # Baseline：全部启用
    TASKS+=("exp2:coupling_baseline_def${def_ep}:--categories ${TEST_CAT} --defense_method geotrap --trap_losses ${TRAP_LOSSES} --multiplicative true --gradient_conflict true --robustness true --defense_epochs ${def_ep} --attack_epochs ${ATTACK_EPOCHS}")

    # w/o 乘法耦合
    TASKS+=("exp2:coupling_wo_mult_def${def_ep}:--categories ${TEST_CAT} --defense_method geotrap --trap_losses ${TRAP_LOSSES} --multiplicative false --gradient_conflict true --robustness true --defense_epochs ${def_ep} --attack_epochs ${ATTACK_EPOCHS}")

    # w/o 梯度冲突
    TASKS+=("exp2:coupling_wo_gc_def${def_ep}:--categories ${TEST_CAT} --defense_method geotrap --trap_losses ${TRAP_LOSSES} --multiplicative true --gradient_conflict false --robustness true --defense_epochs ${def_ep} --attack_epochs ${ATTACK_EPOCHS}")

    # w/o 参数加噪
    TASKS+=("exp2:coupling_wo_robust_def${def_ep}:--categories ${TEST_CAT} --defense_method geotrap --trap_losses ${TRAP_LOSSES} --multiplicative true --gradient_conflict true --robustness false --defense_epochs ${def_ep} --attack_epochs ${ATTACK_EPOCHS}")
done

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "=========================================="
echo "总任务统计"
echo "=========================================="
echo "实验 1 (防御时长): ${#DEFENSE_EPOCHS[@]} 个任务"
echo "实验 2 (互锁消融): $((${#DEFENSE_EPOCHS[@]} * 4)) 个任务 (${#DEFENSE_EPOCHS[@]} def_ep × 4 配置)"
echo "总计: ${TOTAL_TASKS} 个任务"
echo ""

# ============================================================================
# 任务执行函数
# ============================================================================

run_task() {
    local task_idx=$1
    local gpu_idx=$((task_idx % NUM_GPUS))
    local gpu=${GPUS[$gpu_idx]}
    local task=${TASKS[$task_idx]}

    IFS=':' read -r exp_id tag params <<< "$task"

    local log="${OUTPUT_ROOT}/${exp_id}_${tag}.log"
    local output_dir="${OUTPUT_ROOT}/${exp_id}_${tag}"

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${exp_id} - ${tag}"

    {
        echo "=== GPU ${gpu}: ${exp_id} - ${tag} ==="
        echo "Params: ${params}"
        echo ""

        XFORMERS_DISABLED=1 python script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            ${params} \
            --tag "${exp_id}_${tag}" \
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
echo "聚合实验结果汇总"
echo "=========================================="
echo ""

# 实验 1: 防御时长消融
echo "=== 实验 1: 防御时长消融 (attack_epochs=${ATTACK_EPOCHS}) ==="
echo ""
printf "%-15s %-15s %-15s %-15s %-15s\n" \
    "defense_epochs" "Target LPIPS↑" "Target PSNR↓" "Source PSNR↑" "Source LPIPS↓"
echo "------------------------------------------------------------------------"

for def_ep in "${DEFENSE_EPOCHS[@]}"; do
    tag="def_ep${def_ep}"
    metrics="${OUTPUT_ROOT}/exp1_${tag}/metrics.json"

    if [ -f "$metrics" ]; then
        python -c "
import json
with open('${metrics}') as f:
    m = json.load(f)
bt = m.get('postdefense_target') or m.get('baseline_target') or {}
bs = m.get('postdefense_source') or m.get('baseline_source') or {}
print(f'${def_ep:<15d} {bt.get(\"lpips\", 0):>13.4f}   {bt.get(\"psnr\", 0):>13.2f}   {bs.get(\"psnr\", 0):>13.2f}   {bs.get(\"lpips\", 0):>13.4f}')
"
    else
        printf "%-15s (未完成或失败)\n" "${def_ep}"
    fi
done

# 实验 2: 互锁机制消融
echo ""
echo "=== 实验 2: 互锁机制消融 (attack_epochs=${ATTACK_EPOCHS}) ==="
echo ""
printf "%-35s %-10s %-15s %-15s %-15s %-15s\n" \
    "配置" "Def_Ep" "Target LPIPS↑" "Target PSNR↓" "Source PSNR↑" "Source LPIPS↓"
echo "------------------------------------------------------------------------------------------------------------"

for def_ep in "${DEFENSE_EPOCHS[@]}"; do
    for config in "baseline" "wo_mult" "wo_gc" "wo_robust"; do
        tag="coupling_${config}_def${def_ep}"
        metrics="${OUTPUT_ROOT}/exp2_${tag}/metrics.json"

        if [ -f "$metrics" ]; then
            python -c "
import json
with open('${metrics}') as f:
    m = json.load(f)
bt = m.get('postdefense_target') or m.get('baseline_target') or {}
bs = m.get('postdefense_source') or m.get('baseline_source') or {}
print(f'${config:<35s} ${def_ep:<10d} {bt.get(\"lpips\", 0):>13.4f}   {bt.get(\"psnr\", 0):>13.2f}   {bs.get(\"psnr\", 0):>13.2f}   {bs.get(\"lpips\", 0):>13.4f}')
"
        else
            printf "%-35s %-10s (未完成或失败)\n" "${config}" "${def_ep}"
        fi
    done
done

echo ""
echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

echo ""
echo "全部完成！"
