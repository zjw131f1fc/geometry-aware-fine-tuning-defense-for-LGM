#!/bin/bash
# 可视化 Baseline Attack vs Post-Defense Attack 的高斯分布对比
#
# 用法:
#   bash experiments/visualize_distribution.sh <metrics_json_path>                    # 默认输出到同目录
#   bash experiments/visualize_distribution.sh <metrics_json_path> <output_png>       # 指定输出路径
#   OUTPUT=my_viz.png bash experiments/visualize_distribution.sh <metrics_json_path>  # 环境变量指定输出
#
# 示例:
#   bash experiments/visualize_distribution.sh output/experiments_output/single_20260305_123456/metrics.json
#   bash experiments/visualize_distribution.sh output/experiments_output/single_20260305_123456/metrics.json viz.png

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

# 环境变量配置（matplotlib 相关）
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "${MPLCONFIGDIR}"

# 参数配置
if [[ $# -lt 1 ]]; then
    echo "错误: 缺少必需参数 <metrics_json_path>"
    echo ""
    echo "用法:"
    echo "  bash experiments/visualize_distribution.sh <metrics_json_path> [output_png]"
    echo ""
    echo "示例:"
    echo "  bash experiments/visualize_distribution.sh output/experiments_output/single_xxx/metrics.json"
    echo "  bash experiments/visualize_distribution.sh output/experiments_output/single_xxx/metrics.json viz.png"
    exit 1
fi

METRICS_PATH="$1"
OUTPUT_PATH="${2:-${OUTPUT:-}}"

# 如果没有指定输出路径，默认输出到 metrics.json 同目录
if [[ -z "${OUTPUT_PATH}" ]]; then
    METRICS_DIR="$(dirname "${METRICS_PATH}")"
    OUTPUT_PATH="${METRICS_DIR}/gaussian_distribution_comparison.png"
fi

echo "==========================================="
echo "可视化高斯分布对比"
echo "==========================================="
echo "Metrics: ${METRICS_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "==========================================="
echo ""

# 构建命令
CMD=(
    "${PYTHON}" experiments/visualize_attack_defense_distribution.py
    --metrics "${METRICS_PATH}"
    --output "${OUTPUT_PATH}"
)

# 执行
echo "执行命令: ${CMD[*]}"
echo ""
exec "${CMD[@]}"
