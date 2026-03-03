#!/bin/bash
# 简单的多卡测试

set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source experiments/lib/gpu_scheduler.sh

mock_task() {
    local gpu=$1
    local task_id=$2
    echo "[$(date +%H:%M:%S)] GPU ${gpu} 开始 Task ${task_id}"
    sleep 2
    echo "[$(date +%H:%M:%S)] GPU ${gpu} 完成 Task ${task_id}"
}

echo "测试: 5个任务，2张卡"
init_gpu_pool "0,1"

for i in {1..5}; do
    echo "提交 Task $i..."
    submit_task mock_task "$i"
done

wait_all_tasks
