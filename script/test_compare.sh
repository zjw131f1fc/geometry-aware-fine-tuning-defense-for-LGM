#!/bin/bash

# 快速测试脚本 - 使用少量epoch验证功能
#
# 用法: ./script/test_compare.sh [GPU_ID]

GPU=${1:-0}
CONFIG="configs/config.yaml"
ATTACK_EPOCHS=1  # 只训练1个epoch用于快速测试

echo "=========================================="
echo "快速测试：比较随机初始化 vs 预训练LGM"
echo "=========================================="
echo "GPU: $GPU"
echo "配置文件: $CONFIG"
echo "攻击Epochs: $ATTACK_EPOCHS (测试模式)"
echo "=========================================="
echo ""
echo "注意：这是快速测试模式，只训练1个epoch"
echo "完整实验请使用: ./script/run_compare.sh"
echo ""

python script/compare_random_vs_pretrained.py \
    --config "$CONFIG" \
    --attack_epochs "$ATTACK_EPOCHS" \
    --gpu "$GPU" \
    --num_render 1 \
    --eval_every_steps 5 \
    --output_dir output/compare_test

echo ""
echo "测试完成！查看 output/compare_test/ 目录"
