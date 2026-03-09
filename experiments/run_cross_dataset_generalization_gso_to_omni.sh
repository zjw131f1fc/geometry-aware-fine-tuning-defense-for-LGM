#!/bin/bash
# 跨数据集泛化实验（反向）：
# - Attack 阶段 target 使用 OmniObject3D
# - Defense 阶段 target 使用 GSO
#
# 每个 pipeline 自动产生 Undefended（Phase 1）和对应防御方法（Phase 3）的结果
#
# 用法:
#   bash experiments/run_cross_dataset_generalization_gso_to_omni.sh            # 默认 GPU=0 (单卡顺序执行)
#   bash experiments/run_cross_dataset_generalization_gso_to_omni.sh 0          # 指定 GPU=0 (单卡顺序执行)
#   bash experiments/run_cross_dataset_generalization_gso_to_omni.sh 0,1        # 多卡并行: 2张卡动态调度任务

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 加载GPU调度器
source experiments/lib/gpu_scheduler.sh

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

# 解析GPU列表
GPU_LIST="${1:-0}"
echo "GPU列表: ${GPU_LIST}"

CATEGORIES=(shoe plant dish bowl box)
METHODS=(naive_unlearning geotrap)

ATTACK_TARGET_DATASET="omni"
DEFENSE_TARGET_DATASET="gso"
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-registry}"
DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-}"
DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-10}"

CONFIG="configs/config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 默认把实验输出放到 repo 的 output/ 下（本地目录）
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/cross_dataset_gso_defense_omni_attack_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
# 复制配置文件到输出目录，避免后续修改影响实验参数
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_ROOT}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"

echo "=========================================="
echo "跨数据集泛化实验（反向）: 5类别 × 2方法 = 10个pipeline"
echo "Attack target dataset:  ${ATTACK_TARGET_DATASET}"
echo "Defense target dataset: ${DEFENSE_TARGET_DATASET}"
echo "Config: ${CONFIG} (已复制)"
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

# 任务执行函数
# 参数: $1=GPU_ID, $2=task_idx
run_task() {
    local gpu=$1
    local task_idx=$2
    local task=${TASKS[$task_idx]}

    IFS=':' read -r category method <<< "$task"
    local tag="${category}_${method}_gso2omni"
    local log="${OUTPUT_ROOT}/${tag}.log"

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

    if {
        echo "=== GPU ${gpu}: ${tag} ==="
        extra_args=()
        if [[ -n "${DEFENSE_BATCH_SIZE}" ]]; then
            extra_args+=(--defense_batch_size "${DEFENSE_BATCH_SIZE}")
        fi
        if [[ -n "${DEFENSE_GRAD_ACCUM}" ]]; then
            extra_args+=(--defense_grad_accumulation_steps "${DEFENSE_GRAD_ACCUM}")
        fi
        "${PYTHON}" script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            --categories "${category}" \
            --defense_method "${method}" \
            --attack_target_dataset "${ATTACK_TARGET_DATASET}" \
            --defense_target_dataset "${DEFENSE_TARGET_DATASET}" \
            --defense_cache_mode "${DEFENSE_CACHE_MODE}" \
            --eval_every_steps "${EVAL_EVERY_STEPS}" \
            --tag "${tag}" \
            --output_dir "${OUTPUT_ROOT}/${tag}" \
            "${extra_args[@]}"
    } > "${log}" 2>&1; then
        echo "[GPU ${gpu}] 完成: ${tag}"
        return 0
    fi

    exit_code=$?
    echo "[GPU ${gpu}] 失败: ${tag} (exit=${exit_code}), log: ${log}"
    return "${exit_code}"
}

# 初始化GPU池并提交所有任务
init_gpu_pool "${GPU_LIST}"

echo ""
if [[ "${SCHEDULER_ENABLED}" == "true" ]]; then
    echo "多卡并行已启动：动态调度 ${TOTAL_TASKS} 个任务到 GPU [${GPU_LIST}]"
else
    echo "单卡顺序执行已启动：逐个任务运行"
fi
echo "查看进度: tail -f ${OUTPUT_ROOT}/*.log"
echo ""

for i in $(seq 0 $((TOTAL_TASKS-1))); do
    submit_task run_task "${i}"
done

wait_all_tasks

FAILED=${#FAILED_TASKS[@]}

echo ""
echo "全部完成！成功: $((TOTAL_TASKS - FAILED)), 失败: ${FAILED}"

# 汇总结果（同时输出到终端和文件）
SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"

{
echo ""
echo "=========================================="
echo "汇总结果（Defense=GSO, Attack=Omni）"
echo "=========================================="

for category in "${CATEGORIES[@]}"; do
    echo ""
    echo "=== ${category} ==="
    echo ""

    for method in "${METHODS[@]}"; do
        tag="${category}_${method}_gso2omni"
        metrics="${OUTPUT_ROOT}/${tag}/metrics.json"

        if [ -f "$metrics" ]; then
            echo "--- ${method} ---"
            "${PYTHON}" script/print_attack_step_report.py --metrics "$metrics"
        else
            echo "--- ${method} --- (未完成)"
        fi
    done

    first_method="${METHODS[0]}"
    tag="${category}_${first_method}_gso2omni"
    metrics="${OUTPUT_ROOT}/${tag}/metrics.json"

    if [ -f "$metrics" ]; then
        echo ""
        echo "--- Undefended (baseline checkpoints) ---"
        "${PYTHON}" script/print_attack_step_report.py --metrics "$metrics" --phase baseline
    fi
done

echo ""
echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"
