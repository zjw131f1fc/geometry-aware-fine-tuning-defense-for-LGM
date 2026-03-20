#!/bin/bash
# 一步跑完：
# 1. 运行单个 pipeline（自动复用 baseline / defense cache）
# 2. 导出 target baseline / post-defense 对比预览
# 3. 交互输入两个编号，或直接传入两个编号，输出最终 PNG/PDF
#
# 用法:
#   bash experiments/run_target_attack_comparison.sh
#   bash experiments/run_target_attack_comparison.sh 0
#   bash experiments/run_target_attack_comparison.sh 0 3 12
#
# 常用环境变量:
#   CONFIG=configs/config.yaml
#   TAG=target_compare
#   OUTPUT_DIR=output/experiments_output/target_compare_run
#   CATEGORIES=shoe
#   DEFENSE_METHOD=geotrap
#   ATTACK_TARGET_DATASET=omni
#   DEFENSE_TARGET_DATASET=omni
#   ATTACK_STEPS=400
#   DEFENSE_STEPS=100
#   EVAL_EVERY_STEPS=-1
#   TRAP_LOSSES=position,scale,opacity,rotation,color
#   PIPELINE_EXTRA_ARGS="--trap_aggregation_method mean"

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${ROOT_DIR}/../venvs/3d-defense/bin/python" ]]; then
        PYTHON="${ROOT_DIR}/../venvs/3d-defense/bin/python"
    else
        PYTHON="python"
    fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "${MPLCONFIGDIR}"

GPU="${1:-0}"
ID_A="${2:-}"
ID_B="${3:-}"

CONFIG="${CONFIG:-configs/config.yaml}"
TAG="${TAG:-target_attack_comparison}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENTS_BASE}/target_attack_comparison_${TIMESTAMP}}"

mkdir -p "${OUTPUT_DIR}"
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_DIR}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"

echo "=========================================="
echo "一步导出 Target Attack Comparison"
echo "=========================================="
echo "GPU: ${GPU}"
echo "Config: ${CONFIG} (已复制)"
echo "Tag: ${TAG}"
echo "Output: ${OUTPUT_DIR}"
if [[ -n "${ID_A}" && -n "${ID_B}" ]]; then
    echo "选择编号: ${ID_A}, ${ID_B}"
else
    echo "选择编号: 运行后交互输入"
fi
echo "=========================================="
echo ""

PIPELINE_CMD=(
    "${PYTHON}" script/run_pipeline.py
    --gpu "${GPU}"
    --config "${CONFIG}"
    --tag "${TAG}"
    --output_dir "${OUTPUT_DIR}"
    --export_target_comparison_data
    --target_comparison_num_views 4
)

if [[ -n "${CATEGORIES:-}" ]]; then
    PIPELINE_CMD+=(--categories "${CATEGORIES}")
fi
if [[ -n "${DEFENSE_METHOD:-}" ]]; then
    PIPELINE_CMD+=(--defense_method "${DEFENSE_METHOD}")
fi
if [[ -n "${ATTACK_TARGET_DATASET:-}" ]]; then
    PIPELINE_CMD+=(--attack_target_dataset "${ATTACK_TARGET_DATASET}")
fi
if [[ -n "${DEFENSE_TARGET_DATASET:-}" ]]; then
    PIPELINE_CMD+=(--defense_target_dataset "${DEFENSE_TARGET_DATASET}")
fi
if [[ -n "${ATTACK_STEPS:-}" ]]; then
    PIPELINE_CMD+=(--attack_steps "${ATTACK_STEPS}")
fi
if [[ -n "${ATTACK_EPOCHS:-}" ]]; then
    PIPELINE_CMD+=(--attack_epochs "${ATTACK_EPOCHS}")
fi
if [[ -n "${DEFENSE_STEPS:-}" ]]; then
    PIPELINE_CMD+=(--defense_steps "${DEFENSE_STEPS}")
fi
if [[ -n "${DEFENSE_EPOCHS:-}" ]]; then
    PIPELINE_CMD+=(--defense_epochs "${DEFENSE_EPOCHS}")
fi
if [[ -n "${EVAL_EVERY_STEPS:-}" ]]; then
    PIPELINE_CMD+=(--eval_every_steps "${EVAL_EVERY_STEPS}")
fi
if [[ -n "${TRAP_LOSSES:-}" ]]; then
    PIPELINE_CMD+=(--trap_losses "${TRAP_LOSSES}")
fi
if [[ -n "${PIPELINE_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${PIPELINE_EXTRA_ARGS} )
    PIPELINE_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "执行 Pipeline:"
printf '  %q' "${PIPELINE_CMD[@]}"
echo
echo ""
"${PIPELINE_CMD[@]}"

EXPORT_CMD=(
    bash experiments/export_target_attack_comparison.sh
    "${OUTPUT_DIR}"
)

if [[ -n "${ID_A}" || -n "${ID_B}" ]]; then
    if [[ -z "${ID_A}" || -z "${ID_B}" ]]; then
        echo "错误: 如果指定编号，必须同时提供 <id_a> 和 <id_b>"
        exit 1
    fi
    EXPORT_CMD+=("${ID_A}" "${ID_B}")
fi

echo ""
echo "执行 Target Comparison 导出:"
printf '  %q' "${EXPORT_CMD[@]}"
echo
echo ""
exec "${EXPORT_CMD[@]}"
