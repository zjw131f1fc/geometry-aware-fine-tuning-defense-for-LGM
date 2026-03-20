#!/bin/bash
# 导出 target 的 baseline / post-defense before-after 对比图。
#
# 用法:
#   bash experiments/export_target_before_after.sh <workspace_dir>
#   bash experiments/export_target_before_after.sh <workspace_dir> horizontal
#   bash experiments/export_target_before_after.sh <workspace_dir> vertical output/my_compare

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
    echo "用法: bash experiments/export_target_before_after.sh <workspace_dir> [layout] [output_dir]"
    exit 1
fi

WORKSPACE="$1"
LAYOUT="${2:-${LAYOUT:-horizontal}}"
OUTPUT_DIR="${3:-${OUTPUT_DIR:-}}"
LIMIT="${LIMIT:-}"
PREFER="${PREFER:-paired}"
SPACER="${SPACER:-24}"

CMD=(
    "${PYTHON}" script/export_target_before_after.py
    "${WORKSPACE}"
    --layout "${LAYOUT}"
    --prefer "${PREFER}"
    --spacer "${SPACER}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
    CMD+=(--output_dir "${OUTPUT_DIR}")
fi

if [[ -n "${LIMIT}" ]]; then
    CMD+=(--limit "${LIMIT}")
fi

echo "执行命令: ${CMD[*]}"
exec "${CMD[@]}"
