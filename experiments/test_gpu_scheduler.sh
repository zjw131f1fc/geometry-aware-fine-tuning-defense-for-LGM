#!/bin/bash
# GPU调度器测试脚本
# 用于验证多卡调度功能是否正常工作

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 加载GPU调度器
source experiments/lib/gpu_scheduler.sh

echo "=========================================="
echo "GPU调度器测试"
echo "=========================================="

# 模拟任务函数（睡眠几秒钟）
mock_task() {
    local gpu=$1
    local task_id=$2
    local duration=$3

    echo "[GPU ${gpu}] 开始执行 Task ${task_id} (预计 ${duration}s)"
    sleep "$duration"
    echo "[GPU ${gpu}] 完成 Task ${task_id}"
}

# 测试1: 单卡模式
echo ""
echo "测试1: 单卡模式 (GPU 0)"
echo "----------------------------------------"
init_gpu_pool "0"

for i in {1..3}; do
    submit_task mock_task "$i" 2
done

wait_all_tasks
echo "测试1 完成"
echo ""

# 测试2: 多卡模式
echo ""
echo "测试2: 多卡模式 (GPU 0,1)"
echo "----------------------------------------"
init_gpu_pool "0,1"

for i in {1..5}; do
    submit_task mock_task "$i" 3
done

wait_all_tasks
echo "测试2 完成"
echo ""

echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
