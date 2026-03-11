#!/bin/bash
# 攻击实验消融（Section 5.3）：测试类别=bowl
#
# 当前口径：
# - 类别固定为 bowl
# - defense_method 固定为 geotrap
# - defense_steps 固定为 100
# - 其余未显式覆盖的参数保持 config 默认
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
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-registry}"
DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-}"
DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:--1}"
DEFENSE_STEPS="${DEFENSE_STEPS:-100}"
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
echo "Defense steps: ${DEFENSE_STEPS}"
echo "Config: ${CONFIG} (已复制)"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

# 实验配置
TEST_CAT="bowl"

# ============================================================================
# 任务列表（稳健性口径：跑完整 pipeline，使用默认 steps）
# ============================================================================

TASKS=()

# 0. 默认配置（保持 config 默认 attack 设置）
TASKS+=("robust:default:--categories ${TEST_CAT} --defense_method geotrap --defense_steps ${DEFENSE_STEPS}")

# 1. 攻击训练模式
TASKS+=("mode:lora8:--categories ${TEST_CAT} --defense_method geotrap --training_mode lora --lora_r 8 --lora_alpha 8 --defense_steps ${DEFENSE_STEPS}")
TASKS+=("mode:lora32:--categories ${TEST_CAT} --defense_method geotrap --training_mode lora --lora_r 32 --lora_alpha 32 --defense_steps ${DEFENSE_STEPS}")
TASKS+=("mode:full:--categories ${TEST_CAT} --defense_method geotrap --training_mode full --defense_steps ${DEFENSE_STEPS}")

# 2. Optimizer / LR
TASKS+=("optimizer:adamw_3e6:--categories ${TEST_CAT} --defense_method geotrap --optimizer adamw --lr 3e-6 --defense_steps ${DEFENSE_STEPS}")
TASKS+=("optimizer:adamw_3e4:--categories ${TEST_CAT} --defense_method geotrap --optimizer adamw --lr 3e-4 --defense_steps ${DEFENSE_STEPS}")
TASKS+=("optimizer:sgd_3e5:--categories ${TEST_CAT} --defense_method geotrap --optimizer sgd --lr 3e-5 --defense_steps ${DEFENSE_STEPS}")
TASKS+=("optimizer:sgd_3e4:--categories ${TEST_CAT} --defense_method geotrap --optimizer sgd --lr 3e-4 --defense_steps ${DEFENSE_STEPS}")
TASKS+=("optimizer:sgd_3e3:--categories ${TEST_CAT} --defense_method geotrap --optimizer sgd --lr 3e-3 --defense_steps ${DEFENSE_STEPS}")

# 3. Attack steps
TASKS+=("steps:attack_800:--categories ${TEST_CAT} --defense_method geotrap --attack_steps 800 --defense_steps ${DEFENSE_STEPS}")
TASKS+=("steps:attack_1600:--categories ${TEST_CAT} --defense_method geotrap --attack_steps 1600 --defense_steps ${DEFENSE_STEPS}")

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "总任务数: ${TOTAL_TASKS}"
echo "  - 默认: 1 个任务"
echo "  - 训练模式: 3 个任务（lora8 / lora32 / full）"
echo "  - Optimizer/LR: 5 个任务"
echo "  - Attack steps: 2 个任务（800 / 1600）"
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

        extra_args=()
        if [[ -n "${DEFENSE_BATCH_SIZE}" ]]; then
            extra_args+=(--defense_batch_size "${DEFENSE_BATCH_SIZE}")
        fi
        if [[ -n "${DEFENSE_GRAD_ACCUM}" ]]; then
            extra_args+=(--defense_grad_accumulation_steps "${DEFENSE_GRAD_ACCUM}")
        fi

        "${PYTHON}" script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            ${params} \
            --defense_cache_mode "${DEFENSE_CACHE_MODE}" \
            --eval_every_steps "${EVAL_EVERY_STEPS}" \
            --tag "${section}_${tag}" \
            --output_dir "${output_dir}" \
            "${extra_args[@]}"
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

scheduler_exit_code=0
wait_all_tasks || scheduler_exit_code=$?

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

for task in "${TASKS[@]}"; do
    IFS=':' read -r section tag params <<< "$task"

    metrics="${OUTPUT_ROOT}/${section}_${tag}/metrics.json"

    if [ -f "$metrics" ]; then
        echo "--- ${section}_${tag} ---"
        "${PYTHON}" script/print_attack_step_report.py --metrics "$metrics"
        echo ""
    else
        echo "--- ${section}_${tag} ---"
        echo "(未完成或失败)"
        echo ""
    fi
done

echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

echo ""
echo "全部完成！"

exit "${scheduler_exit_code}"
