#!/bin/bash
# 防御训练时长消融：不同 defense_epochs 对攻击抵抗力的影响
# 测试类别=coconut, 攻击配置固定 (全量微调, AdamW lr=5e-5, 2 epochs)
#
# 单卡顺序执行（不做 GPU 空闲检查）
# 用法:
#   bash experiments/run_ablation_defense_epochs.sh            # 默认 GPU=0
#   bash experiments/run_ablation_defense_epochs.sh 0          # 指定 GPU=0
#   bash experiments/run_ablation_defense_epochs.sh 0,1,2,3    # 兼容旧 GPU_LIST，仅使用第一个 GPU

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
echo "单卡顺序执行: GPU=${GPU}"

CONFIG="configs/config.yaml"
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-registry}"
DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-}"
DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-10}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 默认把实验输出放到 repo 的 output/ 下（本地目录）
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/ablation_defense_epochs_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
# 复制配置文件到输出目录，避免后续修改影响实验参数
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_ROOT}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"

echo "=========================================="
echo "防御训练时长消融"
echo "测试类别: coconut"
echo "Config: ${CONFIG} (已复制)"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

TEST_CAT="coconut"
DEFENSE_EPOCHS=(10 20 30 40 50 60 70 80 90 100)

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
    local task=${TASKS[$task_idx]}

    IFS=':' read -r tag params <<< "$task"

    local log="${OUTPUT_ROOT}/${tag}.log"
    local output_dir="${OUTPUT_ROOT}/${tag}"

    echo "[GPU ${GPU}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

    if {
        echo "=== GPU ${GPU}: ${tag} ==="
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
            --tag "${tag}" \
            --output_dir "${output_dir}" \
            "${extra_args[@]}"
    } > "${log}" 2>&1; then
        echo "[GPU ${GPU}] 完成: ${tag}"
        return 0
    fi

    exit_code=$?
    echo "[GPU ${GPU}] 失败: ${tag} (exit=${exit_code}), log: ${log}"
    return "${exit_code}"
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
        "${PYTHON}" -c "
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
