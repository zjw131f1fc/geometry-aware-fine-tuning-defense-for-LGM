#!/bin/bash
# DDP 训练启动脚本

# 使用方法:
# ./script/run_attack_ddp.sh 0,1,2,3 config.yaml

if [ $# -lt 1 ]; then
    echo "使用方法: $0 <GPU_IDS> [CONFIG_FILE]"
    echo "示例: $0 0,1,2,3 config.yaml"
    exit 1
fi

GPU_IDS=$1
CONFIG_FILE=${2:-"configs/config.yaml"}

# 设置可见 GPU
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 设置项目根目录到 PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/lib/LGM:${PYTHONPATH}"

# 计算 GPU 数量
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

echo "=========================================="
echo "DDP 训练启动"
echo "=========================================="
echo "GPU IDs: $GPU_IDS"
echo "GPU 数量: $NUM_GPUS"
echo "配置文件: $CONFIG_FILE"
echo "=========================================="

# 使用 accelerate 启动
accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    script/attack_train_ddp.py \
    --config $CONFIG_FILE
