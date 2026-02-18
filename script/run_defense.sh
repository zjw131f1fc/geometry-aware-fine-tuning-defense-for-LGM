#!/bin/bash
# 防御训练启动脚本

# 使用方法:
# ./script/run_defense.sh <GPU_ID> [CONFIG_FILE] [OPTIONS]
# 示例:
#   ./script/run_defense.sh 0
#   ./script/run_defense.sh 0 configs/config.yaml
#   ./script/run_defense.sh 0 configs/config.yaml --num_epochs 20 --target_layers "conv.weight,unet.conv_in.weight"

if [ $# -lt 1 ]; then
    echo "使用方法: $0 <GPU_ID> [CONFIG_FILE] [OPTIONS]"
    echo "示例: $0 0 configs/config.yaml --num_epochs 20"
    exit 1
fi

GPU_ID=$1
shift
CONFIG_FILE=${1:-"configs/config.yaml"}
# 如果第二个参数不是以 -- 开头，说明是配置文件，消费掉
if [[ "$1" != --* ]] && [ -n "$1" ]; then
    shift
fi
EXTRA_ARGS="$@"

# 设置项目根目录到 PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/LGM:${PYTHONPATH}"

echo "=========================================="
echo "GeoTrap 防御训练启动"
echo "=========================================="
echo "GPU ID: $GPU_ID"
echo "配置文件: $CONFIG_FILE"
echo "额外参数: $EXTRA_ARGS"
echo "=========================================="

python script/train_defense.py \
    --config $CONFIG_FILE \
    --gpu $GPU_ID \
    $EXTRA_ARGS
