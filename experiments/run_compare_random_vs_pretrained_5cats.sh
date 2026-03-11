#!/bin/bash
# 对比实验：随机初始化 vs 预训练（5 个类别）
#
# 用法:
#   bash experiments/run_compare_random_vs_pretrained_5cats.sh            # 默认 GPU=0
#   bash experiments/run_compare_random_vs_pretrained_5cats.sh 0          # 指定 GPU=0
#   bash experiments/run_compare_random_vs_pretrained_5cats.sh 0,1        # 兼容旧 GPU_LIST，仅使用第一个 GPU
#
# 可选环境变量：
#   EXPERIMENTS_BASE=output/experiments_output
#   ATTACK_STEPS=400
#   ATTACK_EPOCHS=5
#   EVAL_EVERY_STEPS=-1
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

CONFIG="${CONFIG:-configs/config.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/compare_random_vs_pretrained_5cats_${TIMESTAMP}"
mkdir -p "${OUTPUT_ROOT}"
# 复制配置文件到输出目录，避免后续修改影响实验参数
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_ROOT}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"

ATTACK_TARGET_DATASET="${ATTACK_TARGET_DATASET:-omni}"
ATTACK_STEPS="${ATTACK_STEPS:-400}"
ATTACK_EPOCHS="${ATTACK_EPOCHS:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:--1}"
NUM_RENDER="${NUM_RENDER:-1}"

CATEGORIES=(shoe plant dish bowl box)

"${PYTHON}" - "${CONFIG}" "${ATTACK_TARGET_DATASET}" <<'PY'
import sys
import yaml

config_path, attack_dataset = sys.argv[1:3]

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("data", {})
cfg["data"].setdefault("target", {})
cfg["data"]["target"]["dataset"] = attack_dataset

with open(config_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

echo "=========================================="
echo "Random Init vs Pretrained（5 类别，attack-only）"
echo "Attack target dataset: ${ATTACK_TARGET_DATASET}"
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

TOTAL_TASKS=${#CATEGORIES[@]}

run_task() {
    local idx=$1
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

    echo "[GPU ${GPU}] 任务 $((idx+1))/${TOTAL_TASKS}: ${category}"

    if {
        echo "=== GPU ${GPU}: ${category} ==="
        "${PYTHON}" script/compare_random_vs_pretrained.py \
            --config "${CONFIG}" \
            --gpu "${GPU}" \
            --categories "${category}" \
            --output_dir "${out_dir}" \
            --eval_every_steps "${EVAL_EVERY_STEPS}" \
            --num_render "${NUM_RENDER}" \
            "${extra_args[@]}"
    } > "${log}" 2>&1; then
        echo "[GPU ${GPU}] 完成: ${category}"
        return 0
    fi

    exit_code=$?
    echo "[GPU ${GPU}] 失败: ${category} (exit=${exit_code}), log: ${log}"
    return "${exit_code}"
}

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
echo "结果保存在: ${OUTPUT_ROOT}"
