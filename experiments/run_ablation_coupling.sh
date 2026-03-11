#!/bin/bash
# 互锁机制消融实验（Section 5.2）
# Baseline 全启用，然后 w/o 形式去掉每个机制
#
# 单卡顺序执行（不做 GPU 空闲检查）
# 用法:
#   bash experiments/run_ablation_coupling.sh            # 默认 GPU=0
#   bash experiments/run_ablation_coupling.sh 0          # 指定 GPU=0
#   bash experiments/run_ablation_coupling.sh 0,1,2,3    # 兼容旧 GPU_LIST，仅使用第一个 GPU

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
echo "单卡顺序执行: GPU=${GPU}"

CONFIG="configs/config.yaml"
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-registry}"
DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-}"
DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:--1}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 默认把实验输出放到 repo 的 output/ 下（本地目录）
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/ablation_coupling_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
# 复制配置文件到输出目录，避免后续修改影响实验参数
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_ROOT}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"

echo "=========================================="
echo "互锁机制消融实验 (Section 5.2)"
echo "测试类别: bowl"
echo "Config: ${CONFIG} (已复制)"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

TEST_CAT="bowl"
# 精细指标口径：Defense 仅训练 50 step（不看最终 LPIPS/PSNR）
DEFENSE_STEPS=50

# 固定最优trap组合（从5.1实验结果确定，这里假设是scale+opacity）
TRAP_LOSSES="scale,opacity"

# ============================================================================
# 任务列表（Baseline 全启用，然后单项消融：每次只去掉一个机制）
# ============================================================================

TASKS=()

# Baseline：输入加噪 + 冻结head + logsumexp聚合
TASKS+=("5.2:baseline_all:--categories ${TEST_CAT} --defense_method geotrap --trap_losses ${TRAP_LOSSES}")

# w/o 输入加噪
TASKS+=("5.2:wo_input_noise:--categories ${TEST_CAT} --defense_method geotrap --trap_losses ${TRAP_LOSSES} --input_noise_enabled false")

# w/o 冻结head
TASKS+=("5.2:wo_freeze_head:--categories ${TEST_CAT} --defense_method geotrap --trap_losses ${TRAP_LOSSES} --freeze_head false")

# logsumexp 换成 mean（平均，保持数值范围）
TASKS+=("5.2:mean_aggregation:--categories ${TEST_CAT} --defense_method geotrap --trap_losses ${TRAP_LOSSES} --trap_aggregation_method mean")

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "总任务数: ${TOTAL_TASKS}"
echo "  1. Baseline (输入加噪 + 冻结head + logsumexp聚合)"
echo "  2. w/o 输入加噪"
echo "  3. w/o 冻结head"
echo "  4. mean聚合（替代logsumexp，保持数值范围）"
echo ""

# ============================================================================
# 任务执行函数
# ============================================================================

run_task() {
    local task_idx=$1
    local task=${TASKS[$task_idx]}

    IFS=':' read -r section tag params <<< "$task"

    local log="${OUTPUT_ROOT}/${section}_${tag}.log"
    local output_dir="${OUTPUT_ROOT}/${section}_${tag}"

    echo "[GPU ${GPU}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${section} - ${tag}"

    if {
        echo "=== GPU ${GPU}: ${section} - ${tag} ==="
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
            --defense_steps "${DEFENSE_STEPS}" \
            --defense_cache_mode "${DEFENSE_CACHE_MODE}" \
            --eval_every_steps "${EVAL_EVERY_STEPS}" \
            --tag "${section}_${tag}" \
            --output_dir "${output_dir}" \
            "${extra_args[@]}"
    } > "${log}" 2>&1; then
        echo "[GPU ${GPU}] 完成: ${section} - ${tag}"
        return 0
    else
        exit_code=$?
        echo "[GPU ${GPU}] 失败: ${section} - ${tag} (exit=${exit_code}), log: ${log}"
        return "${exit_code}"
    fi
}

# ============================================================================
# 启动所有任务
# ============================================================================

echo ""
echo "顺序执行已启动：单卡逐个任务运行（不检查 GPU 空闲）"
echo "查看进度: tail -f ${OUTPUT_ROOT}/*.log"
echo ""

FAILED=0
for i in $(seq 0 $((TOTAL_TASKS-1))); do
    if ! run_task "${i}"; then
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
echo "互锁机制消融结果汇总 (Section 5.2)"
echo "=========================================="
echo ""

for task in "${TASKS[@]}"; do
    IFS=':' read -r section tag params <<< "$task"

    metrics="${OUTPUT_ROOT}/${section}_${tag}/metrics.json"

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

echo ""
echo "全部完成！"
