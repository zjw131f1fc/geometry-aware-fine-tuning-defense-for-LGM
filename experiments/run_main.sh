#!/bin/bash
# 主实验：5个类别 × 2种防御方法（naive_unlearning + geotrap）
# 每个pipeline自动产生Undefended（Phase 1）和对应防御方法（Phase 3）的结果
#
# 用法: bash experiments/run_main.sh 0,1,2,3

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

if [ $# -eq 0 ]; then
    echo "用法: bash experiments/run_main.sh GPU_LIST"
    echo "示例: bash experiments/run_main.sh 0,1,2,3"
    exit 1
fi

# 解析GPU列表
IFS=',' read -ra GPUS <<< "$1"
NUM_GPUS=${#GPUS[@]}

echo "使用 ${NUM_GPUS} 张GPU: ${GPUS[@]}"

CATEGORIES=(shoe plant dish bowl box)
METHODS=(geotrap naive_unlearning)

CONFIG="configs/config.yaml"
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-registry}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 默认把实验输出放到 repo 的 output/ 下（本环境通常会把 output/ 链接到系统盘，避免写满数据盘）
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/main_experiment_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
echo "=========================================="
echo "主实验: 5类别 × 2方法 = 10个pipeline"
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

# 任务启动函数（指定GPU）
run_task() {
    local task_idx=$1
    local gpu=$2
    local task=${TASKS[$task_idx]}

    IFS=':' read -r category method <<< "$task"
    local tag="${category}_${method}"
    local log="${OUTPUT_ROOT}/${tag}.log"

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

    {
        echo "=== GPU ${gpu}: ${tag} ==="
        "${PYTHON}" script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            --categories "${category}" \
            --defense_method "${method}" \
            --defense_cache_mode "${DEFENSE_CACHE_MODE}" \
            --tag "${tag}" \
            --output_dir "${OUTPUT_ROOT}/${tag}"
    } > "${log}" 2>&1 &

    local pid=$!
    PID_TO_GPU["${pid}"]="${gpu}"
    PID_TO_TAG["${pid}"]="${tag}"
    RUNNING=$((RUNNING + 1))
    echo "[GPU ${gpu}] PID: ${pid}, log: ${log}"
}

# 动态调度：谁先空闲就立刻补下一个任务
declare -A PID_TO_GPU
declare -A PID_TO_TAG
RUNNING=0
NEXT_TASK=0
COMPLETED=0
FAILED=0

# 先给每张GPU各发一个任务
for gpu in "${GPUS[@]}"; do
    if [ ${NEXT_TASK} -lt ${TOTAL_TASKS} ]; then
        run_task "${NEXT_TASK}" "${gpu}"
        NEXT_TASK=$((NEXT_TASK + 1))
    fi
done

echo ""
echo "动态调度已启动：GPU 空闲后立即补任务"
echo ""

while [ ${RUNNING} -gt 0 ]; do
    finished_pid=""
    if wait -n -p finished_pid; then
        exit_code=0
    else
        exit_code=$?
    fi

    finished_gpu="${PID_TO_GPU[${finished_pid}]}"
    finished_tag="${PID_TO_TAG[${finished_pid}]}"
    unset PID_TO_GPU["${finished_pid}"]
    unset PID_TO_TAG["${finished_pid}"]

    RUNNING=$((RUNNING - 1))
    COMPLETED=$((COMPLETED + 1))

    if [ ${exit_code} -eq 0 ]; then
        echo "[GPU ${finished_gpu}] 完成: ${finished_tag} (${COMPLETED}/${TOTAL_TASKS})"
    else
        FAILED=$((FAILED + 1))
        echo "[GPU ${finished_gpu}] 失败: ${finished_tag} (exit=${exit_code})"
    fi

    if [ ${NEXT_TASK} -lt ${TOTAL_TASKS} ]; then
        run_task "${NEXT_TASK}" "${finished_gpu}"
        NEXT_TASK=$((NEXT_TASK + 1))
    fi
done

echo ""
echo "全部完成！成功: $((TOTAL_TASKS - FAILED)), 失败: ${FAILED}"

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
            "${PYTHON}" -c "
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
        "${PYTHON}" -c "
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
