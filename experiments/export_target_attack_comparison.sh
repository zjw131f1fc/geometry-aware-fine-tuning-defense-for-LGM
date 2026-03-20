#!/bin/bash
# 导出 target 的 baseline / post-defense 攻击对比图。
#
# 用法:
#   bash experiments/export_target_attack_comparison.sh <workspace_dir>
#     先生成编号预览，再交互输入两个编号
#
#   bash experiments/export_target_attack_comparison.sh <workspace_dir> 3 12
#     直接选择编号 3 和 12，跳过交互
#
# 常用环境变量:
#   OUTPUT_DIR=output/my_compare
#   PREVIEW_LIMIT=50
#   BASELINE_DIR=/path/to/phase1_baseline_attack/target_comparison_export
#   POSTDEF_DIR=/path/to/phase3_postdefense_attack/target_comparison_export
#   CELL_GAP=18
#   GROUP_GAP=44
#   ROW_GAP=40
#   HEADER_HEIGHT=64

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

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

if [[ $# -lt 1 ]]; then
    echo "错误: 缺少 <workspace_dir>"
    echo ""
    echo "用法:"
    echo "  bash experiments/export_target_attack_comparison.sh <workspace_dir>"
    echo "  bash experiments/export_target_attack_comparison.sh <workspace_dir> <id_a> <id_b>"
    exit 1
fi

WORKSPACE="$1"
ID_A="${2:-}"
ID_B="${3:-}"

OUTPUT_DIR="${OUTPUT_DIR:-}"
PREVIEW_LIMIT="${PREVIEW_LIMIT:-}"
BASELINE_DIR="${BASELINE_DIR:-}"
POSTDEF_DIR="${POSTDEF_DIR:-}"
CELL_GAP="${CELL_GAP:-}"
GROUP_GAP="${GROUP_GAP:-}"
ROW_GAP="${ROW_GAP:-}"
HEADER_HEIGHT="${HEADER_HEIGHT:-}"

CMD=(
    "${PYTHON}" script/export_target_attack_comparison.py
    "${WORKSPACE}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
    CMD+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${PREVIEW_LIMIT}" ]]; then
    CMD+=(--preview-limit "${PREVIEW_LIMIT}")
fi

if [[ -n "${BASELINE_DIR}" ]]; then
    CMD+=(--baseline-dir "${BASELINE_DIR}")
fi

if [[ -n "${POSTDEF_DIR}" ]]; then
    CMD+=(--postdef-dir "${POSTDEF_DIR}")
fi

if [[ -n "${CELL_GAP}" ]]; then
    CMD+=(--cell-gap "${CELL_GAP}")
fi

if [[ -n "${GROUP_GAP}" ]]; then
    CMD+=(--group-gap "${GROUP_GAP}")
fi

if [[ -n "${ROW_GAP}" ]]; then
    CMD+=(--row-gap "${ROW_GAP}")
fi

if [[ -n "${HEADER_HEIGHT}" ]]; then
    CMD+=(--header-height "${HEADER_HEIGHT}")
fi

if [[ -n "${ID_A}" || -n "${ID_B}" ]]; then
    if [[ -z "${ID_A}" || -z "${ID_B}" ]]; then
        echo "错误: 如果指定编号，必须同时提供 <id_a> 和 <id_b>"
        exit 1
    fi
    CMD+=(--select "${ID_A}" "${ID_B}")
fi

echo "执行命令: ${CMD[*]}"
exec "${CMD[@]}"
