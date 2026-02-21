#!/bin/bash
# C(4,2)=6 种 trap loss 组合完整 sweep
# 用法: bash script/sweep_trap_combos.sh [GPU_ID]
# 默认 GPU 6，输出日志保存到 output/sweep_logs/

set -e

GPU=${1:-6}
LOG_DIR="output/sweep_logs"
mkdir -p "$LOG_DIR"

source /mnt/huangjiaxin/venvs/3d-defense/bin/activate

COMBOS=(
    "position,scale"
    "position,opacity"
    "position,rotation"
    "scale,opacity"
    "scale,rotation"
    "opacity,rotation"
)

echo "=========================================="
echo "  Trap Loss Sweep: C(4,2) = 6 组合"
echo "  GPU: $GPU"
echo "  日志目录: $LOG_DIR"
echo "=========================================="

for combo in "${COMBOS[@]}"; do
    tag="sweep_${combo//,/_}"
    log_file="${LOG_DIR}/${tag}.log"

    echo ""
    echo ">>> [$tag] 开始 ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo ">>> 日志: $log_file"

    python script/run_pipeline.py \
        --gpu "$GPU" \
        --config configs/config.yaml \
        --trap_losses "$combo" \
        --tag "$tag" \
        2>&1 | tee "$log_file"

    echo ">>> [$tag] 完成 ($(date '+%Y-%m-%d %H:%M:%S'))"
done

echo ""
echo "=========================================="
echo "  全部完成！日志在 $LOG_DIR"
echo "=========================================="
