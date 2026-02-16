#!/bin/bash
# 3D Defense 攻击训练 - DDP 分布式训练启动脚本
#
# 使用方法:
#   ./scripts/run_attack_ddp.sh [GPU列表] [配置文件]
#
# 示例:
#   ./scripts/run_attack_ddp.sh 0,1,2,3                        # 使用 GPU 0,1,2,3，默认配置
#   ./scripts/run_attack_ddp.sh 6,7 configs/attack_config.yaml # 使用 GPU 6,7，指定配置
#   ./scripts/run_attack_ddp.sh 4,5,6,7                        # 使用 GPU 4,5,6,7，默认配置

set -e  # 遇到错误立即退出

# 默认参数
DEFAULT_GPUS="0,1,2,3"
DEFAULT_CONFIG="configs/attack_config.yaml"

# 解析参数
GPUS=${1:-$DEFAULT_GPUS}
CONFIG=${2:-$DEFAULT_CONFIG}

# 计算GPU数量
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=========================================="
echo "3D Defense DDP 分布式训练"
echo "=========================================="
echo "GPU列表: $GPUS"
echo "GPU数量: $NUM_GPUS"
echo "配置文件: $CONFIG"
echo "=========================================="

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

# 检查训练脚本是否存在
TRAIN_SCRIPT="scripts/attack_test_ddp.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "错误: 训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

# 使用统一的accelerate配置模板
ACCELERATE_CONFIG="configs/accelerate.yaml"

if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "错误: Accelerate配置文件不存在: $ACCELERATE_CONFIG"
    exit 1
fi

echo "Accelerate配置: $ACCELERATE_CONFIG"
echo "进程数: $NUM_GPUS"
echo "=========================================="
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPUS
export HF_ENDPOINT="https://hf-mirror.com"

# 启动训练（使用 --num_processes 覆盖配置文件中的进程数）
echo "启动命令: accelerate launch --config_file $ACCELERATE_CONFIG --num_processes $NUM_GPUS $TRAIN_SCRIPT --config $CONFIG"
accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes $NUM_GPUS \
    "$TRAIN_SCRIPT" \
    --config "$CONFIG"

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
