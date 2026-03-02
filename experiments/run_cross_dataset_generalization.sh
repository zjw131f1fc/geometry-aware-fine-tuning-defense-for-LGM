#!/bin/bash
# 跨数据集泛化实验：
# - Attack 阶段 target 使用 GSO
# - Defense 阶段 target 使用 OmniObject3D
#
# 每个 pipeline 自动产生 Undefended（Phase 1）和对应防御方法（Phase 3）的结果
#
# 用法: bash experiments/run_cross_dataset_generalization.sh 0,1,2,3

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Prefer project venv python if available; allow override via $PYTHON
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${ROOT_DIR}/../venvs/3d-defense/bin/python" ]]; then
        PYTHON="${ROOT_DIR}/../venvs/3d-defense/bin/python"
    else
        PYTHON="python"
    fi
fi

# Avoid OpenMP env issues + make matplotlib cache writable (important for multiprocessing)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "${MPLCONFIGDIR}"

if [ $# -eq 0 ]; then
    echo "用法: bash experiments/run_cross_dataset_generalization.sh GPU_LIST"
    echo "示例: bash experiments/run_cross_dataset_generalization.sh 0,1,2,3"
    exit 1
fi

# 解析GPU列表
IFS=',' read -ra GPUS <<< "$1"
NUM_GPUS=${#GPUS[@]}

echo "使用 ${NUM_GPUS} 张GPU: ${GPUS[@]}"

CATEGORIES=(shoe plant dish bowl box)
METHODS=(geotrap naive_unlearning)

ATTACK_TARGET_DATASET="gso"
DEFENSE_TARGET_DATASET="omni"

CONFIG="configs/config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 默认把实验输出放到 repo 的 output/ 下（本环境通常会把 output/ 链接到系统盘，避免写满数据盘）
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/cross_dataset_omni_defense_gso_attack_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
echo "=========================================="
echo "跨数据集泛化实验: 5类别 × 2方法 = 10个pipeline"
echo "Attack target dataset:  ${ATTACK_TARGET_DATASET}"
echo "Defense target dataset: ${DEFENSE_TARGET_DATASET}"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

# 生成任务列表
TASKS=()
for method in "${METHODS[@]}"; do
    for category in "${CATEGORIES[@]}"; do
        TASKS+=("${category}:${method}")
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo "总任务数: ${TOTAL_TASKS}"
echo ""

# 任务分配函数
run_task() {
    local task_idx=$1
    local gpu_idx=$((task_idx % NUM_GPUS))
    local gpu=${GPUS[$gpu_idx]}
    local task=${TASKS[$task_idx]}

    IFS=':' read -r category method <<< "$task"
    local tag="${category}_${method}_omni2gso"
    local log="${OUTPUT_ROOT}/${tag}.log"

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

    {
        echo "=== GPU ${gpu}: ${tag} ==="
        "${PYTHON}" script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            --categories "${category}" \
            --defense_method "${method}" \
            --attack_target_dataset "${ATTACK_TARGET_DATASET}" \
            --defense_target_dataset "${DEFENSE_TARGET_DATASET}" \
            --tag "${tag}" \
            --output_dir "${OUTPUT_ROOT}/${tag}"
    } > "${log}" 2>&1 &

    echo "[GPU ${gpu}] PID: $!, log: ${log}"
}

# 启动所有任务（自动分配到GPU）
for i in $(seq 0 $((TOTAL_TASKS-1))); do
    run_task $i

    # 每启动NUM_GPUS个任务后等待，避免同时启动太多
    if [ $(((i+1) % NUM_GPUS)) -eq 0 ] && [ $((i+1)) -lt ${TOTAL_TASKS} ]; then
        echo ""
        echo "已启动 $((i+1)) 个任务，等待当前批次完成..."
        wait
        echo "当前批次完成，继续启动..."
        echo ""
    fi
done

echo ""
echo "所有任务已启动，等待完成..."
echo "查看进度: tail -f ${OUTPUT_ROOT}/*.log"
wait
echo ""
echo "全部完成！"

# 汇总结果（同时输出到终端和文件）
SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"

{
echo ""
echo "=========================================="
echo "汇总结果（Defense=Omni, Attack=GSO）"
echo "=========================================="

for category in "${CATEGORIES[@]}"; do
    echo ""
    echo "=== ${category} ==="
    echo ""

    for method in "${METHODS[@]}"; do
        tag="${category}_${method}_omni2gso"
        metrics="${OUTPUT_ROOT}/${tag}/metrics.json"

        if [ -f "$metrics" ]; then
            echo "--- ${method} ---"
            python -c "
import json
with open('${metrics}') as f:
    m = json.load(f)

# Undefended = baseline
bt_base = m.get('baseline_target') or {}
bs_base = m.get('baseline_source') or {}

# 防御方法 = postdefense
bt_def = m.get('postdefense_target') or {}
bs_def = m.get('postdefense_source') or {}

print(f'  Target LPIPS: {bt_base.get(\"lpips\", 0):.4f} → {bt_def.get(\"lpips\", 0):.4f} (Δ={bt_def.get(\"lpips\", 0)-bt_base.get(\"lpips\", 0):+.4f})')
print(f'  Target PSNR:  {bt_base.get(\"psnr\", 0):.2f} → {bt_def.get(\"psnr\", 0):.2f} (Δ={bt_def.get(\"psnr\", 0)-bt_base.get(\"psnr\", 0):+.2f})')
print(f'  Source PSNR:  {bs_base.get(\"psnr\", 0):.2f} → {bs_def.get(\"psnr\", 0):.2f} (Δ={bs_def.get(\"psnr\", 0)-bs_base.get(\"psnr\", 0):+.2f})')
print(f'  Source LPIPS: {bs_base.get(\"lpips\", 0):.4f} → {bs_def.get(\"lpips\", 0):.4f} (Δ={bs_def.get(\"lpips\", 0)-bs_base.get(\"lpips\", 0):+.4f})')
"
        else
            echo "--- ${method} --- (未完成)"
        fi
    done

    # 打印Undefended（使用第一个方法的baseline）
    first_method="${METHODS[0]}"
    tag="${category}_${first_method}_omni2gso"
    metrics="${OUTPUT_ROOT}/${tag}/metrics.json"

    if [ -f "$metrics" ]; then
        echo ""
        echo "--- Undefended (baseline) ---"
        python -c "
import json
with open('${metrics}') as f:
    m = json.load(f)
bt = m.get('baseline_target') or {}
bs = m.get('baseline_source') or {}
print(f'  Target LPIPS: {bt.get(\"lpips\", 0):.4f}')
print(f'  Target PSNR:  {bt.get(\"psnr\", 0):.2f}')
print(f'  Source PSNR:  {bs.get(\"psnr\", 0):.2f}')
print(f'  Source LPIPS: {bs.get(\"lpips\", 0):.4f}')
"
    fi
done

echo ""
echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"
