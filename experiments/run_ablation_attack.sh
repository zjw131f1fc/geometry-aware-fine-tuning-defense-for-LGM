#!/bin/bash
# 攻击实验消融（Section 5.3）：测试类别=coconut
# 包含所有消融实验，优先分配语义偏转攻击
#
# 用法: bash experiments/run_ablation_attack.sh GPU_LIST
# 示例: bash experiments/run_ablation_attack.sh 0,1,2,3

set -e

if [ $# -eq 0 ]; then
    echo "用法: bash experiments/run_ablation_attack.sh GPU_LIST"
    echo "示例: bash experiments/run_ablation_attack.sh 0,1,2,3"
    exit 1
fi

# 解析GPU列表
IFS=',' read -ra GPUS <<< "$1"
NUM_GPUS=${#GPUS[@]}

echo "使用 ${NUM_GPUS} 张GPU: ${GPUS[@]}"

CONFIG="configs/config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="experiments_output/ablation_attack_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
echo "=========================================="
echo "攻击实验消融 (Section 5.3)"
echo "测试类别: coconut"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

# 实验配置
TEST_CAT="coconut"
ATTACK_EPOCHS=5
DEFENSE_EPOCHS=60

# ============================================================================
# 任务列表（优先级排序：语义偏转 > 其他）
# ============================================================================

TASKS=()

# ========== 优先级1: Section 5.3.1 微调方式 ==========
# 全量微调 (baseline): 已有结果 (LPIPS=0.4655, PSNR=13.5424)，跳过
# LoRA r=8, alpha=16
TASKS+=("5.3.1:finetune_lora_r8:--categories ${TEST_CAT} --defense_method geotrap --training_mode lora --lora_r 8 --lora_alpha 16")
# LoRA r=16, alpha=32
TASKS+=("5.3.1:finetune_lora_r16:--categories ${TEST_CAT} --defense_method geotrap --training_mode lora --lora_r 16 --lora_alpha 32")

# ========== 优先级3: Section 5.3.2 优化器与学习率 (成对: AdamW vs SGD, 小/中/大) ==========
# AdamW lr=1e-5 (小)
TASKS+=("5.3.2:optimizer_adamw_1e5:--categories ${TEST_CAT} --defense_method geotrap --lr 1e-5")
# SGD+momentum(0.9) lr=1e-4 (小)
TASKS+=("5.3.2:optimizer_sgd_1e4:--categories ${TEST_CAT} --defense_method geotrap --optimizer sgd --lr 1e-4")
# AdamW lr=5e-5 (中, baseline): 已有结果 (LPIPS=0.4655, PSNR=13.5424)，跳过
# SGD+momentum(0.9) lr=1e-3 (中)
TASKS+=("5.3.2:optimizer_sgd_1e3:--categories ${TEST_CAT} --defense_method geotrap --optimizer sgd --lr 1e-3")
# AdamW lr=2e-4 (大)
TASKS+=("5.3.2:optimizer_adamw_2e4:--categories ${TEST_CAT} --defense_method geotrap --lr 2e-4")
# SGD+momentum(0.9) lr=1e-2 (大)
TASKS+=("5.3.2:optimizer_sgd_1e2:--categories ${TEST_CAT} --defense_method geotrap --optimizer sgd --lr 1e-2")

# ========== 优先级4: Section 5.3.3 攻击时长 ==========
# 2 epochs
TASKS+=("5.3.3:duration_2ep:--categories ${TEST_CAT} --defense_method geotrap --attack_epochs 2")
# 5 epochs (baseline): 已有结果 (LPIPS=0.4655, PSNR=13.5424)，跳过
# 10 epochs
TASKS+=("5.3.3:duration_10ep:--categories ${TEST_CAT} --defense_method geotrap --attack_epochs 10")
# 20 epochs
TASKS+=("5.3.3:duration_20ep:--categories ${TEST_CAT} --defense_method geotrap --attack_epochs 20")

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "总任务数: ${TOTAL_TASKS}"
echo "  - Section 5.3.1 (微调方式): 2 个任务 (baseline已有结果)"
echo "  - Section 5.3.2 (优化器): 5 个任务"
echo "  - Section 5.3.3 (攻击时长): 3 个任务"
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

        # 执行 pipeline
        python script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            ${params} \
            --defense_epochs "${DEFENSE_EPOCHS}" \
            --skip_baseline \
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

    # 每启动NUM_GPUS个任务后等待
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
echo "攻击实验消融结果汇总 (Section 5.3)"
echo "=========================================="
echo ""

# ========== Section 5.3.1: 微调方式 ==========
echo "=== Section 5.3.1: 微调方式 ==="
echo ""

for task in "${TASKS[@]}"; do
    IFS=':' read -r section tag params <<< "$task"

    if [ "$section" != "5.3.1" ]; then
        continue
    fi

    metrics="${OUTPUT_ROOT}/${section}_${tag}/metrics.json"

    echo "--- ${tag} ---"

    if [ -f "$metrics" ]; then
        python -c "
import json
with open('${metrics}') as f:
    m = json.load(f)

target = m.get('postdefense_target', {})
print(f'  Target LPIPS: {target.get(\"lpips\", 0):.4f}')
print(f'  Target PSNR:  {target.get(\"psnr\", 0):.2f}')
"
    else
        echo "  (未完成或失败)"
    fi
    echo ""
done

# ========== Section 5.3.2: 优化器与学习率 ==========
echo "=== Section 5.3.2: 优化器与学习率 ==="
echo ""

for task in "${TASKS[@]}"; do
    IFS=':' read -r section tag params <<< "$task"

    if [ "$section" != "5.3.2" ]; then
        continue
    fi

    metrics="${OUTPUT_ROOT}/${section}_${tag}/metrics.json"

    echo "--- ${tag} ---"

    if [ -f "$metrics" ]; then
        python -c "
import json
with open('${metrics}') as f:
    m = json.load(f)

target = m.get('postdefense_target', {})
print(f'  Target LPIPS: {target.get(\"lpips\", 0):.4f}')
print(f'  Target PSNR:  {target.get(\"psnr\", 0):.2f}')
"
    else
        echo "  (未完成或失败)"
    fi
    echo ""
done

# ========== Section 5.3.3: 攻击时长 ==========
echo "=== Section 5.3.3: 攻击时长 ==="
echo ""

for task in "${TASKS[@]}"; do
    IFS=':' read -r section tag params <<< "$task"

    if [ "$section" != "5.3.3" ]; then
        continue
    fi

    metrics="${OUTPUT_ROOT}/${section}_${tag}/metrics.json"

    echo "--- ${tag} ---"

    if [ -f "$metrics" ]; then
        python -c "
import json
with open('${metrics}') as f:
    m = json.load(f)

target = m.get('postdefense_target', {})
print(f'  Target LPIPS: {target.get(\"lpips\", 0):.4f}')
print(f'  Target PSNR:  {target.get(\"psnr\", 0):.2f}')
"
    else
        echo "  (未完成或失败)"
    fi
    echo ""
done

echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

echo ""
echo "全部完成！"
