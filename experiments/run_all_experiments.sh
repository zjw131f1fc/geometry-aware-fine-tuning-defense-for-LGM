#!/bin/bash
# 聚合实验脚本：防御时长消融 + 互锁机制消融
#
# 实验 1: 防御时长消融 - defense_epochs = 10,20,30,40, attack_epochs = 10
# 实验 2: 互锁机制消融 - 每个 defense_epoch 下做 baseline + w/o robustness
#
# 单卡顺序执行（不做 GPU 空闲检查）
# 用法:
#   bash experiments/run_all_experiments.sh            # 默认 GPU=0
#   bash experiments/run_all_experiments.sh 0          # 指定 GPU=0
#   bash experiments/run_all_experiments.sh 0,1,2,3    # 兼容旧 GPU_LIST，仅使用第一个 GPU

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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

# 单卡选择：优先使用 CLI 参数，其次 GPU_ID/GPU 环境变量，最后默认 0
GPU_ID="${GPU_ID:-${GPU:-0}}"
if [ $# -ge 1 ]; then
    GPU_ARG="$1"
    if [[ "${GPU_ARG}" == *","* ]]; then
        GPU_ID="${GPU_ARG%%,*}"
        echo "检测到 GPU_LIST='${GPU_ARG}'，单卡模式仅使用第一个 GPU: ${GPU_ID}"
    else
        GPU_ID="${GPU_ARG}"
    fi
fi

if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID 必须是非负整数，当前: '${GPU_ID}'"
    exit 1
fi

GPU="${GPU_ID}"

echo "=========================================="
echo "聚合实验脚本"
echo "单卡顺序执行: GPU=${GPU}"
echo "=========================================="

CONFIG="configs/config.yaml"
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-registry}"
DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-}"
DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-10}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 默认把实验输出放到 repo 的 output/ 下（本地目录）
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/all_experiments_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
# 复制配置文件到输出目录，避免后续修改影响实验参数
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_ROOT}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"
echo "Config copied to: ${CONFIG}"

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
# 实验 2: 互锁机制消融（每个 defense_epoch 下做 2 种配置）
# ============================================================================

echo ""
echo "=========================================="
echo "实验 2: 互锁机制消融"
echo "=========================================="

for def_ep in "${DEFENSE_EPOCHS[@]}"; do
    # Baseline：全部启用
    TASKS+=("exp2:coupling_baseline_def${def_ep}:--categories ${TEST_CAT} --defense_method geotrap --trap_losses ${TRAP_LOSSES} --robustness true --defense_epochs ${def_ep} --attack_epochs ${ATTACK_EPOCHS}")

    # w/o 参数加噪
    TASKS+=("exp2:coupling_wo_robust_def${def_ep}:--categories ${TEST_CAT} --defense_method geotrap --trap_losses ${TRAP_LOSSES} --robustness false --defense_epochs ${def_ep} --attack_epochs ${ATTACK_EPOCHS}")
done

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "=========================================="
echo "总任务统计"
echo "=========================================="
echo "实验 1 (防御时长): ${#DEFENSE_EPOCHS[@]} 个任务"
echo "实验 2 (互锁消融): $((${#DEFENSE_EPOCHS[@]} * 2)) 个任务 (${#DEFENSE_EPOCHS[@]} def_ep × 2 配置)"
echo "总计: ${TOTAL_TASKS} 个任务"
echo ""

# ============================================================================
# 任务执行函数
# ============================================================================

run_task() {
    local task_idx=$1
    local task=${TASKS[$task_idx]}

    IFS=':' read -r exp_id tag params <<< "$task"

    local log="${OUTPUT_ROOT}/${exp_id}_${tag}.log"
    local output_dir="${OUTPUT_ROOT}/${exp_id}_${tag}"

    echo "[GPU ${GPU}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${exp_id} - ${tag}"

    if {
        echo "=== GPU ${GPU}: ${exp_id} - ${tag} ==="
        echo "Params: ${params}"
        echo ""

        extra_args=()
        if [[ -n "${DEFENSE_BATCH_SIZE}" ]]; then
            extra_args+=(--defense_batch_size "${DEFENSE_BATCH_SIZE}")
        fi
        if [[ -n "${DEFENSE_GRAD_ACCUM}" ]]; then
            extra_args+=(--defense_grad_accumulation_steps "${DEFENSE_GRAD_ACCUM}")
        fi
        XFORMERS_DISABLED=1 "${PYTHON}" script/run_pipeline.py \
            --gpu "${GPU}" \
            --config "${CONFIG}" \
            ${params} \
            --defense_cache_mode "${DEFENSE_CACHE_MODE}" \
            --eval_every_steps "${EVAL_EVERY_STEPS}" \
            --tag "${exp_id}_${tag}" \
            --output_dir "${output_dir}" \
            "${extra_args[@]}"
    } > "${log}" 2>&1; then
        echo "[GPU ${GPU}] 完成: ${exp_id} - ${tag}"
        return 0
    fi

    exit_code=$?
    echo "[GPU ${GPU}] 失败: ${exp_id} - ${tag} (exit=${exit_code}), log: ${log}"
    return "${exit_code}"
}

# ============================================================================
# 启动所有任务（单卡、顺序）
# ============================================================================

echo ""
echo "顺序执行已启动：单卡逐个任务运行（不检查 GPU 空闲）"
echo "查看进度: tail -f ${OUTPUT_ROOT}/*.log"
echo ""

COMPLETED=0
FAILED=0
for i in $(seq 0 $((TOTAL_TASKS-1))); do
    if run_task "${i}"; then
        COMPLETED=$((COMPLETED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "全部完成！成功: $((TOTAL_TASKS - FAILED)), 失败: ${FAILED}"

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

for def_ep in "${DEFENSE_EPOCHS[@]}"; do
    tag="def_ep${def_ep}"
    metrics="${OUTPUT_ROOT}/exp1_${tag}/metrics.json"

    if [ -f "$metrics" ]; then
        echo "--- defense_epochs=${def_ep} ---"
        "${PYTHON}" script/print_attack_step_report.py --metrics "$metrics"
        echo ""
    else
        echo "--- defense_epochs=${def_ep} ---"
        echo "(未完成或失败)"
        echo ""
    fi
done

# 实验 2: 互锁机制消融
echo ""
echo "=== 实验 2: 互锁机制消融 (attack_epochs=${ATTACK_EPOCHS}) ==="
echo ""

for def_ep in "${DEFENSE_EPOCHS[@]}"; do
    for config in "baseline" "wo_robust"; do
        tag="coupling_${config}_def${def_ep}"
        metrics="${OUTPUT_ROOT}/exp2_${tag}/metrics.json"

        if [ -f "$metrics" ]; then
            echo "--- config=${config}, defense_epochs=${def_ep} ---"
            "${PYTHON}" script/print_attack_step_report.py --metrics "$metrics"
            echo ""
        else
            echo "--- config=${config}, defense_epochs=${def_ep} ---"
            echo "(未完成或失败)"
            echo ""
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
