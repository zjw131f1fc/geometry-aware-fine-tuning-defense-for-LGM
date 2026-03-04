#!/bin/bash
# 攻击实验消融（Section 5.3）：测试类别=bowl
#
# 用法:
#   bash experiments/run_ablation_attack.sh            # 默认 GPU=0 (单卡顺序执行)
#   bash experiments/run_ablation_attack.sh 0          # 指定 GPU=0 (单卡顺序执行)
#   bash experiments/run_ablation_attack.sh 0,1        # 多卡并行: 2张卡动态调度任务

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 加载GPU调度器
source experiments/lib/gpu_scheduler.sh

# Prefer project venv python if available; allow override via $PYTHON
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${ROOT_DIR}/../venvs/3d-defense/bin/python" ]]; then
        PYTHON="${ROOT_DIR}/../venvs/3d-defense/bin/python"
    else
        PYTHON="python"
    fi
fi

# Avoid OpenMP env issues + keep caches/tmp off system disk when possible
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
if [[ -d "/root/autodl-tmp" ]]; then
    export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/.cache}"
    export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/.cache/torch}"
    export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
    export WANDB_DIR="${WANDB_DIR:-/root/autodl-tmp/.cache/wandb}"
    mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${HF_HOME}" "${WANDB_DIR}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl}"
mkdir -p "${MPLCONFIGDIR}"

# 解析GPU列表
GPU_LIST="${1:-0}"
echo "GPU列表: ${GPU_LIST}"

CONFIG="configs/config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 默认把实验输出放到 repo 的 output/ 下（本地目录）
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/ablation_attack_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
# 复制配置文件到输出目录，避免后续修改影响实验参数
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_ROOT}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"

echo "=========================================="
echo "攻击实验消融"
echo "测试类别: bowl"
echo "Config: ${CONFIG} (已复制)"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

# 实验配置
TEST_CAT="bowl"

# ============================================================================
# 任务列表（稳健性口径：跑完整 pipeline，使用默认 steps）
# ============================================================================

TASKS=()

# 0. 默认配置（用于稳健性最终指标，attack_steps/defense_steps 使用 config 默认值）
TASKS+=("robust:default:--categories ${TEST_CAT} --defense_method geotrap")

# 1. LoRA rank=8, alpha=8 (已跑过，保留注释不再启用)
# TASKS+=("lora:r8a8:--categories ${TEST_CAT} --defense_method geotrap --training_mode lora --lora_r 8 --lora_alpha 8 --skip_baseline")

# 2. LoRA rank=32, alpha=32 (已跑过，保留注释不再启用)
# TASKS+=("lora:r32a32:--categories ${TEST_CAT} --defense_method geotrap --training_mode lora --lora_r 32 --lora_alpha 32 --skip_baseline")

# 3. AdamW lr=3e-6 (已跑过，保留注释不再启用)
# TASKS+=("optimizer:adamw_3e6:--categories ${TEST_CAT} --defense_method geotrap --lr 3e-6 --skip_baseline")

# 4. AdamW lr=3e-4 (已跑过，保留注释不再启用)
# TASKS+=("optimizer:adamw_3e4:--categories ${TEST_CAT} --defense_method geotrap --lr 3e-4 --skip_baseline")

# 5. SGD lr=3e-5 (已跑过，保留注释不再启用)
# TASKS+=("optimizer:sgd_3e5:--categories ${TEST_CAT} --defense_method geotrap --optimizer sgd --lr 3e-5 --skip_baseline")

# 6. SGD lr=3e-4 (已跑过，保留注释不再启用)
# TASKS+=("optimizer:sgd_3e4:--categories ${TEST_CAT} --defense_method geotrap --optimizer sgd --lr 3e-4 --skip_baseline")

# 7. SGD lr=3e-3 (已跑过，保留注释不再启用)
# TASKS+=("optimizer:sgd_3e3:--categories ${TEST_CAT} --defense_method geotrap --optimizer sgd --lr 3e-3 --skip_baseline")

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "总任务数: ${TOTAL_TASKS}"
echo "  - 稳健性: ${TOTAL_TASKS} 个任务（完整 pipeline；默认 steps）"
echo ""

# ============================================================================
# 任务执行函数
# ============================================================================

run_task() {
    local gpu=$1
    local task_idx=$2
    local task=${TASKS[$task_idx]}

    IFS=':' read -r section tag params <<< "$task"

    local log="${OUTPUT_ROOT}/${section}_${tag}.log"
    local output_dir="${OUTPUT_ROOT}/${section}_${tag}"

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${section} - ${tag}"

    if {
        echo "=== GPU ${gpu}: ${section} - ${tag} ==="
        echo "Params: ${params}"
        echo ""

        # 执行 pipeline
        "${PYTHON}" script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            ${params} \
            --tag "${section}_${tag}" \
            --output_dir "${output_dir}"
    } > "${log}" 2>&1; then
        echo "[GPU ${gpu}] 完成: ${section} - ${tag}"
        return 0
    fi

    exit_code=$?
    echo "[GPU ${gpu}] 失败: ${section} - ${tag} (exit=${exit_code}), log: ${log}"
    return "${exit_code}"
}

# ============================================================================
# 启动所有任务
# ============================================================================

# 初始化GPU池并提交所有任务
init_gpu_pool "${GPU_LIST}"

echo ""
if [[ "${SCHEDULER_ENABLED}" == "true" ]]; then
    echo "多卡并行已启动：动态调度 ${TOTAL_TASKS} 个任务到 GPU [${GPU_LIST}]"
else
    echo "单卡顺序执行已启动：逐个任务运行"
fi
echo "查看进度: tail -f ${OUTPUT_ROOT}/*.log"
echo ""

for i in $(seq 0 $((TOTAL_TASKS-1))); do
    submit_task run_task "${i}"
done

wait_all_tasks

FAILED=${#FAILED_TASKS[@]}

echo ""
echo "全部完成！成功: $((TOTAL_TASKS - FAILED)), 失败: ${FAILED}"

# ============================================================================
# 汇总结果
# ============================================================================

SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"

{
echo ""
echo "=========================================="
echo "攻击实验消融结果汇总"
echo "=========================================="
echo ""

printf "%-30s %-15s %-15s %-15s %-15s\n" \
    "实验配置" "Target LPIPS↑" "Target PSNR↓" "Source PSNR↑" "Source LPIPS↓"
echo "----------------------------------------------------------------------------------------------------"

for task in "${TASKS[@]}"; do
    IFS=':' read -r section tag params <<< "$task"

    metrics="${OUTPUT_ROOT}/${section}_${tag}/metrics.json"

    if [ -f "$metrics" ]; then
        "${PYTHON}" -c "
import json
with open('${metrics}') as f:
    m = json.load(f)

pt = m.get('postdefense_target') or {}
ps = m.get('postdefense_source') or {}

name = '${section}_${tag}'
print(f'{name:<30s} {pt.get(\"lpips\", 0):>13.4f}   {pt.get(\"psnr\", 0):>13.2f}   {ps.get(\"psnr\", 0):>13.2f}   {ps.get(\"lpips\", 0):>13.4f}')
"
    else
        printf "%-30s (未完成或失败)\n" "${section}_${tag}"
    fi
done

echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

echo ""
echo "全部完成！"
