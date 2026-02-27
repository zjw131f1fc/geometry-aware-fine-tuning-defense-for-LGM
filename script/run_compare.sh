#!/bin/bash

# 比较随机初始化 vs 预训练LGM的启动脚本
#
# 用法:
#   ./script/run_compare.sh [GPU_ID] [CONFIG] [ATTACK_EPOCHS]
#
# 示例:
#   ./script/run_compare.sh 0 configs/config.yaml 5

GPU=${1:-0}
CONFIG=${2:-configs/config.yaml}
ATTACK_EPOCHS=${3:-5}

echo "=========================================="
echo "比较随机初始化 vs 预训练LGM"
echo "=========================================="
echo "GPU: $GPU"
echo "配置文件: $CONFIG"
echo "攻击Epochs: $ATTACK_EPOCHS"
echo "=========================================="

python script/compare_random_vs_pretrained.py \
    --config "$CONFIG" \
    --attack_epochs "$ATTACK_EPOCHS" \
    --gpu "$GPU" \
    --num_render 3 \
    --eval_every_steps 10

echo ""
echo "完成！查看 output/compare_random_vs_pretrained/ 目录获取结果"
