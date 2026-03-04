#!/bin/bash
# 单任务快速运行脚本
# 自动加载 venv 环境，使用配置文件默认值运行单个实验
#
# 用法:
#   bash experiments/run_single.sh                    # 默认 GPU=0
#   bash experiments/run_single.sh 1                  # 指定 GPU=1
#   SKIP_BASELINE=1 bash experiments/run_single.sh 0  # 跳过 baseline
#   TAG=my_test bash experiments/run_single.sh 0      # 自定义标签

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
GPU="${1:-0}"
CONFIG="${CONFIG:-configs/config.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TAG="${TAG:-single_test}"
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENTS_BASE}/single_${TIMESTAMP}}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"

# 创建输出目录并复制配置文件
mkdir -p "${OUTPUT_DIR}"
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_DIR}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"

echo "=========================================="
echo "单任务快速运行"
echo "=========================================="
echo "GPU: ${GPU}"
echo "Config: ${CONFIG} (已复制)"
echo "Tag: ${TAG}"
echo "Output: ${OUTPUT_DIR}"
if [[ "${SKIP_BASELINE}" == "1" ]]; then
    echo "模式: 跳过 Baseline Attack"
fi
echo "=========================================="
echo ""

# 构建命令
CMD=(
    "${PYTHON}" script/run_pipeline.py
    --gpu "${GPU}"
    --config "${CONFIG}"
    --tag "${TAG}"
    --output_dir "${OUTPUT_DIR}"
)

# 添加可选参数
if [[ "${SKIP_BASELINE}" == "1" ]]; then
    CMD+=(--skip_baseline)
fi

# 执行
echo "执行命令: ${CMD[*]}"
echo ""
exec "${CMD[@]}"
