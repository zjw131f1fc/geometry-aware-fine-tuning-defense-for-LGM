#!/bin/bash
# 对比实验：随机初始化 vs 预训练（5 个类别）
#
# 用法:
#   bash experiments/run_compare_random_vs_pretrained_5cats.sh 0,1
#
# 可选环境变量：
#   EXPERIMENTS_BASE=output/experiments_output
#   ATTACK_STEPS=200
#   ATTACK_EPOCHS=5
#   EVAL_EVERY_STEPS=10
#   NUM_RENDER=1
#
# 输出：
#   ${EXPERIMENTS_BASE}/compare_random_vs_pretrained_5cats_<timestamp>/

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
    echo "用法: bash experiments/run_compare_random_vs_pretrained_5cats.sh GPU_LIST"
    echo "示例: bash experiments/run_compare_random_vs_pretrained_5cats.sh 0,1"
    exit 1
fi

IFS=',' read -ra GPUS <<< "$1"
NUM_GPUS=${#GPUS[@]}
echo "使用 ${NUM_GPUS} 张GPU: ${GPUS[@]}"

CONFIG="${CONFIG:-configs/config.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/compare_random_vs_pretrained_5cats_${TIMESTAMP}"
mkdir -p "${OUTPUT_ROOT}"

ATTACK_STEPS="${ATTACK_STEPS:-}"
ATTACK_EPOCHS="${ATTACK_EPOCHS:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-10}"
NUM_RENDER="${NUM_RENDER:-1}"

CATEGORIES=(shoe plant dish bowl box)

echo "=========================================="
echo "Random Init vs Pretrained（5 类别）"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_ROOT}"
if [[ -n "${ATTACK_STEPS}" ]]; then
    echo "Attack steps: ${ATTACK_STEPS}"
elif [[ -n "${ATTACK_EPOCHS}" ]]; then
    echo "Attack epochs: ${ATTACK_EPOCHS}"
else
    echo "Attack steps/epochs: 从 config 读取"
fi
echo "eval_every_steps: ${EVAL_EVERY_STEPS}, num_render: ${NUM_RENDER}"
echo "=========================================="

# 动态调度：谁先空闲就立刻补下一个任务
declare -A PID_TO_GPU
declare -A PID_TO_CAT
RUNNING=0
NEXT_TASK=0
TOTAL_TASKS=${#CATEGORIES[@]}

launch_on_gpu() {
    local gpu=$1
    local idx=$2
    local category=${CATEGORIES[$idx]}
    local tag="compare_${category}"
    local log="${OUTPUT_ROOT}/${tag}.log"
    local out_dir="${OUTPUT_ROOT}/${tag}"

    extra_args=()
    if [[ -n "${ATTACK_STEPS}" ]]; then
        extra_args+=(--attack_steps "${ATTACK_STEPS}")
    fi
    if [[ -n "${ATTACK_EPOCHS}" ]]; then
        extra_args+=(--attack_epochs "${ATTACK_EPOCHS}")
    fi

    echo "[GPU ${gpu}] 启动: ${category}"
    echo "=== GPU ${gpu}: ${category} ===" > "${log}"
    "${PYTHON}" script/compare_random_vs_pretrained.py \
        --config "${CONFIG}" \
        --gpu "${gpu}" \
        --categories "${category}" \
        --output_dir "${out_dir}" \
        --eval_every_steps "${EVAL_EVERY_STEPS}" \
        --num_render "${NUM_RENDER}" \
        "${extra_args[@]}" >> "${log}" 2>&1 &

    local pid=$!
    PID_TO_GPU["${pid}"]="${gpu}"
    PID_TO_CAT["${pid}"]="${category}"
    RUNNING=$((RUNNING + 1))
    echo "[GPU ${gpu}] PID: ${pid}, log: ${log}"
}

# 先给每张 GPU 各发一个任务
for gpu in "${GPUS[@]}"; do
    if [ ${NEXT_TASK} -lt ${TOTAL_TASKS} ]; then
        launch_on_gpu "${gpu}" "${NEXT_TASK}"
        NEXT_TASK=$((NEXT_TASK + 1))
    fi
done

echo ""
echo "动态调度已启动：GPU 空闲后立即补任务"
echo "查看进度: tail -f ${OUTPUT_ROOT}/*.log"
echo ""

while [ ${RUNNING} -gt 0 ]; do
    finished_pid=""
    if wait -n -p finished_pid; then
        exit_code=0
    else
        exit_code=$?
    fi

    finished_gpu="${PID_TO_GPU[${finished_pid}]}"
    finished_cat="${PID_TO_CAT[${finished_pid}]}"
    unset PID_TO_GPU["${finished_pid}"]
    unset PID_TO_CAT["${finished_pid}"]
    RUNNING=$((RUNNING - 1))

    if [ ${exit_code} -eq 0 ]; then
        echo "[GPU ${finished_gpu}] 完成: ${finished_cat}"
    else
        echo "[GPU ${finished_gpu}] 失败: ${finished_cat} (exit=${exit_code})"
    fi

    if [ ${NEXT_TASK} -lt ${TOTAL_TASKS} ]; then
        launch_on_gpu "${finished_gpu}" "${NEXT_TASK}"
        NEXT_TASK=$((NEXT_TASK + 1))
    fi
done

echo ""
echo "全部完成！结果保存在: ${OUTPUT_ROOT}"
