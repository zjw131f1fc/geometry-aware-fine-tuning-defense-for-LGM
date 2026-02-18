#!/bin/bash
# 防御训练 + 攻击评估 启动脚本
#
# 使用方法:
#   ./script/run_defense_eval.sh <GPU_ID> [CONFIG_FILE] [OPTIONS]
# 示例:
#   ./script/run_defense_eval.sh 7
#   ./script/run_defense_eval.sh 7 configs/config.yaml
#   ./script/run_defense_eval.sh 7 configs/config.yaml --output_dir output/my_exp

if [ $# -lt 1 ]; then
    echo "使用方法: $0 <GPU_ID> [CONFIG_FILE] [OPTIONS]"
    echo "示例: $0 7 configs/config.yaml"
    exit 1
fi

GPU_ID=$1
shift
CONFIG_FILE=${1:-"configs/config.yaml"}
if [[ "$1" != --* ]] && [ -n "$1" ]; then
    shift
fi
EXTRA_ARGS="$@"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/LGM:${PYTHONPATH}"

echo "=========================================="
echo "GeoTrap 防御训练 + 攻击评估"
echo "=========================================="
echo "GPU ID: $GPU_ID"
echo "配置文件: $CONFIG_FILE"
echo "额外参数: $EXTRA_ARGS"
echo "=========================================="

python script/train_defense_with_eval.py \
    --config $CONFIG_FILE \
    --gpu $GPU_ID \
    $EXTRA_ARGS
