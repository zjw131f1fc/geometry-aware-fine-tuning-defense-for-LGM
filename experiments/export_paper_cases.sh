#!/bin/bash
# 导出论文案例图：原图 + Defense 后渲染图成对保存
#
# 用法:
#   bash experiments/export_paper_cases.sh <workspace_dir>                    # 默认 GPU=0, 5个样本, 导出 source+target
#   bash experiments/export_paper_cases.sh <workspace_dir> 1                  # 指定 GPU=1
#   bash experiments/export_paper_cases.sh <workspace_dir> 0 10               # GPU=0, 10个样本
#   NUM_SAMPLES=8 bash experiments/export_paper_cases.sh <workspace_dir>      # 自定义样本数
#   DATA_TYPE=source bash experiments/export_paper_cases.sh <workspace_dir>  # 只导出 source（Defense后未攻击）
#   DATA_TYPE=target bash experiments/export_paper_cases.sh <workspace_dir>  # 只导出 target（Post-Defense Attack）
#   DATA_TYPE=both bash experiments/export_paper_cases.sh <workspace_dir>    # 导出两者（默认）
#   OUTPUT_DIR=my_figures bash experiments/export_paper_cases.sh <workspace_dir>  # 自定义输出目录
#
# 示例:
#   bash experiments/export_paper_cases.sh output/experiments_output/single_20260305_123456

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 自动检测并使用 venv Python
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${ROOT_DIR}/../venvs/3d-defense/bin/python" ]]; then
        PYTHON="${ROOT_DIR}/../venvs/3d-defense/bin/python"
        echo "✓ 使用 venv Python: ${PYTHON}"
    else
        PYTHON="python"
        echo "⚠ venv 未找到，使用系统 Python"
    fi
fi

# 环境变量配置
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "${MPLCONFIGDIR}"

# 参数配置
if [[ $# -lt 1 ]]; then
    echo "错误: 缺少必需参数 <workspace_dir>"
    echo ""
    echo "用法:"
    echo "  bash experiments/export_paper_cases.sh <workspace_dir> [gpu] [num_samples]"
    echo ""
    echo "示例:"
    echo "  bash experiments/export_paper_cases.sh output/experiments_output/single_20260305_123456"
    echo "  bash experiments/export_paper_cases.sh output/experiments_output/single_20260305_123456 1 10"
    exit 1
fi

WORKSPACE="$1"
GPU="${2:-${GPU:-0}}"
NUM_SAMPLES="${3:-${NUM_SAMPLES:-5}}"
DATA_TYPE="${DATA_TYPE:-both}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "==========================================="
echo "导出论文案例图"
echo "==========================================="
echo "工作目录: ${WORKSPACE}"
echo "GPU: ${GPU}"
echo "样本数: ${NUM_SAMPLES}"
echo "数据类型: ${DATA_TYPE}"
if [[ -n "${OUTPUT_DIR}" ]]; then
    echo "输出目录: ${OUTPUT_DIR}"
else
    echo "输出目录: ${WORKSPACE}/paper_cases (默认)"
fi
echo "==========================================="
echo ""

# 构建命令
CMD=(
    "${PYTHON}" script/export_paper_cases.py
    "${WORKSPACE}"
    --gpu "${GPU}"
    --num_samples "${NUM_SAMPLES}"
    --data_type "${DATA_TYPE}"
)

# 添加可选参数
if [[ -n "${OUTPUT_DIR}" ]]; then
    CMD+=(--output_dir "${OUTPUT_DIR}")
fi

# 执行
echo "执行命令: ${CMD[*]}"
echo ""
exec "${CMD[@]}"
