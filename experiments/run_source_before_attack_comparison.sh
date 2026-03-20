#!/bin/bash
# 一步跑完：
# 1. 运行 pipeline，仅做 clean baseline / defense 后的 source 预导出（两者都在 attack 前）
# 2. 导出 source clean baseline / defense 对比预览
# 3. 交互输入两个编号，或直接传入两个编号，输出最终 PNG/PDF
#
# 用法:
#   bash experiments/run_source_before_attack_comparison.sh
#   bash experiments/run_source_before_attack_comparison.sh 0
#   bash experiments/run_source_before_attack_comparison.sh 0 3 12
#
# 常用环境变量:
#   CONFIG=configs/config.yaml
#   TAG=source_before_attack_comparison
#   OUTPUT_DIR=output/experiments_output/source_before_attack_comparison_xxx
#   CATEGORIES=shoe
#   DEFENSE_METHOD=geotrap
#   ATTACK_TARGET_DATASET=omni
#   DEFENSE_TARGET_DATASET=omni
#   DEFENSE_STEPS=100
#   DEFENSE_EPOCHS=25
#   TRAP_LOSSES=position,scale,opacity,rotation,color
#   DEFENSE_CACHE_MODE=registry
#   SOURCE_COMPARISON_NUM_VIEWS=4
#   SOURCE_COMPARISON_MAX_SAMPLES=50
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
TAG="${TAG:-source_before_attack_comparison}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENTS_BASE}/source_before_attack_comparison_${TIMESTAMP}}"
SOURCE_COMPARISON_NUM_VIEWS="${SOURCE_COMPARISON_NUM_VIEWS:-4}"

mkdir -p "${OUTPUT_DIR}"
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_DIR}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"

echo "=========================================="
echo "一步导出 Source Before-Attack Comparison"
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
    --skip_baseline
    --skip_postdefense_attack
    --export_source_comparison_data
    --source_comparison_num_views "${SOURCE_COMPARISON_NUM_VIEWS}"
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
if [[ -n "${DEFENSE_STEPS:-}" ]]; then
    PIPELINE_CMD+=(--defense_steps "${DEFENSE_STEPS}")
fi
if [[ -n "${DEFENSE_EPOCHS:-}" ]]; then
    PIPELINE_CMD+=(--defense_epochs "${DEFENSE_EPOCHS}")
fi
if [[ -n "${DEFENSE_CACHE_MODE:-}" ]]; then
    PIPELINE_CMD+=(--defense_cache_mode "${DEFENSE_CACHE_MODE}")
fi
if [[ -n "${DEFENSE_BATCH_SIZE:-}" ]]; then
    PIPELINE_CMD+=(--defense_batch_size "${DEFENSE_BATCH_SIZE}")
fi
if [[ -n "${DEFENSE_GRAD_ACCUM:-}" ]]; then
    PIPELINE_CMD+=(--defense_grad_accumulation_steps "${DEFENSE_GRAD_ACCUM}")
fi
if [[ -n "${TRAP_LOSSES:-}" ]]; then
    PIPELINE_CMD+=(--trap_losses "${TRAP_LOSSES}")
fi
if [[ -n "${SOURCE_COMPARISON_MAX_SAMPLES:-}" ]]; then
    PIPELINE_CMD+=(--source_comparison_max_samples "${SOURCE_COMPARISON_MAX_SAMPLES}")
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
    bash experiments/export_source_before_attack_comparison.sh
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
echo "执行 Source Comparison 导出:"
printf '  %q' "${EXPORT_CMD[@]}"
echo
echo ""
exec "${EXPORT_CMD[@]}"
