#!/bin/bash
# 快速测试效率测量功能
# 使用小规模参数快速验证集成是否正常

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=========================================="
echo "效率测量功能快速测试"
echo "=========================================="
echo ""
echo "这个测试会运行一个小规模实验来验证效率测量功能"
echo "预计耗时: 5-10分钟"
echo ""
read -p "按 Enter 继续，或 Ctrl+C 取消..."
echo ""

# 使用小规模参数
MEASURE_EFFICIENCY=1 \
TAG=efficiency_quick_test \
python script/run_pipeline.py \
    --gpu 0 \
    --config configs/config.yaml \
    --attack_steps 50 \
    --defense_steps 50 \
    --eval_every_steps 10 \
    --num_render 1

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "请检查输出目录中的 efficiency_report.json 文件"
echo "如果看到了效率对比报告，说明集成成功！"
echo ""
