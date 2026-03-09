#!/bin/bash
# 防御机制消融实验：grad surgery, energy梯度裁剪, 陷阱聚合
# Baseline 全启用，然后 w/o 形式去掉每个机制
#
# 用法:
#   bash experiments/run_ablation_mechanisms.sh            # 默认 GPU=0 (单卡顺序执行)
#   bash experiments/run_ablation_mechanisms.sh 0          # 指定 GPU=0 (单卡顺序执行)
#   bash experiments/run_ablation_mechanisms.sh 0,1,2,3    # 多卡并行: 4张卡动态调度6个任务
#
# 环境变量:
#   SKIP_BASELINE=1                  # 跳过 baseline attack，直接从 defense 开始
#   DEFENSE_CACHE_MODE=none          # 防御缓存模式（none/readonly/registry）
#   EVAL_EVERY_STEPS=10              # 评估间隔步数

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

# Avoid OpenMP env issues + make matplotlib cache writable (important for multiprocessing)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "${MPLCONFIGDIR}"

# 解析GPU列表
GPU_LIST="${1:-0}"
echo "GPU列表: ${GPU_LIST}"

CONFIG="configs/config.yaml"
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-none}"  # 消融实验不使用缓存，确保每个配置独立训练
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-10}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"  # 默认不跳过 baseline
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 默认把实验输出放到 repo 的 output/ 下（本地目录）
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/ablation_mechanisms_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
# 复制配置文件到输出目录，避免后续修改影响实验参数
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_ROOT}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"

echo "=========================================="
echo "防御机制消融实验"
echo "测试配置: ${CONFIG} (已复制)"
echo "Output: ${OUTPUT_ROOT}"
if [[ "${SKIP_BASELINE}" == "1" ]]; then
    echo "模式: 跳过 Baseline Attack"
fi
echo "=========================================="

# ============================================================================
# 任务列表（Baseline 全启用，然后单项消融：每次只去掉一个机制）
# ============================================================================

TASKS=()

# Baseline：Energy梯度裁剪 + bottleneck_logsumexp 陷阱聚合 + freeze_head + 上游防御（不使用 PCGrad）
TASKS+=("baseline_all:--grad_surgery_enabled false --grad_clip_mode energy --grad_energy_mult 5.0 --trap_aggregation_method bottleneck_logsumexp --trap_bottleneck_tau 0.25 --freeze_head true --freeze_lora_targets true")

# w/o Energy梯度裁剪 (改用普通 norm 裁剪)
TASKS+=("wo_energy_clip:--grad_surgery_enabled false --grad_clip_mode norm --trap_aggregation_method bottleneck_logsumexp --trap_bottleneck_tau 0.25 --freeze_head true --freeze_lora_targets true")

# w/o bottleneck聚合 (改用简单平均)
TASKS+=("wo_bottleneck_agg:--grad_surgery_enabled false --grad_clip_mode energy --grad_energy_mult 5.0 --trap_aggregation_method mean --freeze_head true --freeze_lora_targets true")

# w/o freeze_head (不冻结头部卷积层)
TASKS+=("wo_freeze_head:--grad_surgery_enabled false --grad_clip_mode energy --grad_energy_mult 5.0 --trap_aggregation_method bottleneck_logsumexp --trap_bottleneck_tau 0.25 --freeze_head false --freeze_lora_targets true")

# w/o upstream defense (不启用上游防御)
TASKS+=("wo_upstream_defense:--grad_surgery_enabled false --grad_clip_mode energy --grad_energy_mult 5.0 --trap_aggregation_method bottleneck_logsumexp --trap_bottleneck_tau 0.25 --freeze_head true --freeze_lora_targets false")

# w/o 所有机制
TASKS+=("wo_all_mechanisms:--grad_surgery_enabled false --grad_clip_mode norm --trap_aggregation_method mean --freeze_head false --freeze_lora_targets false")

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "总任务数: ${TOTAL_TASKS}"
echo "  1. Baseline (Energy裁剪 + Bottleneck聚合 + Freeze Head + 上游防御)"
echo "  2. w/o Energy梯度裁剪"
echo "  3. w/o Bottleneck聚合"
echo "  4. w/o Freeze Head"
echo "  5. w/o Upstream Defense (不冻结 LoRA 目标层)"
echo "  6. w/o 所有机制"
echo ""

# ============================================================================
# 任务执行函数
# 参数: $1=GPU_ID, $2=task_idx
# ============================================================================

run_task() {
    local gpu=$1
    local task_idx=$2
    local task=${TASKS[$task_idx]}

    IFS=':' read -r tag params <<< "$task"

    local log="${OUTPUT_ROOT}/${tag}.log"
    local output_dir="${OUTPUT_ROOT}/${tag}"

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

    if {
        echo "=== GPU ${gpu}: ${tag} ==="
        echo "Params: ${params}"
        if [[ "${SKIP_BASELINE}" == "1" ]]; then
            echo "Skip baseline: true"
        fi
        echo ""

        # 构建命令
        if [[ "${SKIP_BASELINE}" == "1" ]]; then
            XFORMERS_DISABLED=1 "${PYTHON}" script/run_pipeline.py \
                --gpu "${gpu}" \
                --config "${CONFIG}" \
                ${params} \
                --defense_cache_mode "${DEFENSE_CACHE_MODE}" \
                --eval_every_steps "${EVAL_EVERY_STEPS}" \
                --tag "${tag}" \
                --output_dir "${output_dir}" \
                --skip_baseline
        else
            XFORMERS_DISABLED=1 "${PYTHON}" script/run_pipeline.py \
                --gpu "${gpu}" \
                --config "${CONFIG}" \
                ${params} \
                --defense_cache_mode "${DEFENSE_CACHE_MODE}" \
                --eval_every_steps "${EVAL_EVERY_STEPS}" \
                --tag "${tag}" \
                --output_dir "${output_dir}"
        fi
    } > "${log}" 2>&1; then
        echo "[GPU ${gpu}] 完成: ${tag}"
        return 0
    fi

    exit_code=$?
    echo "[GPU ${gpu}] 失败: ${tag} (exit=${exit_code}), log: ${log}"
    return "${exit_code}"
}

# ============================================================================
# 启动所有任务
# ============================================================================

# 初始化GPU池
init_gpu_pool "${GPU_LIST}"

# 提交所有任务
echo ""
echo "开始提交任务..."
for i in $(seq 0 $((TOTAL_TASKS-1))); do
    submit_task run_task "$i"
done

# 等待所有任务完成
echo ""
wait_all_tasks
scheduler_exit_code=$?

echo ""
echo "=========================================="
echo "汇总结果"
echo "=========================================="

# ============================================================================
# 汇总结果
# ============================================================================

SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"

{
echo ""
echo "=========================================="
echo "防御机制消融结果汇总"
echo "=========================================="
echo ""

for task in "${TASKS[@]}"; do
    IFS=':' read -r tag params <<< "$task"

    metrics="${OUTPUT_ROOT}/${tag}/metrics.json"

    if [ -f "$metrics" ]; then
        echo "--- ${tag} ---"
        "${PYTHON}" script/print_attack_step_report.py --metrics "$metrics"
        echo ""
    else
        echo "--- ${tag} ---"
        echo "(未完成或失败)"
        echo ""
    fi
done

echo ""
echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

exit "${scheduler_exit_code}"
