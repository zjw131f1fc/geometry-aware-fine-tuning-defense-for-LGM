#!/bin/bash
# 防御类别数消融实验（OmniObject3D target）
#
# 目的：
#   在同一 target 数据集 OmniObject3D 上（attack/defense 都是 omni），
#   分别用 2 / 4 个类别进行防御训练并跑完整 pipeline（baseline→defense→postdefense）。
#   同时测试 naive_unlearning 和 geotrap 两种防御方法。
#
# 用法:
#   bash experiments/run_ablation_defense_num_categories.sh            # 默认 GPU=0 (单卡顺序执行)
#   bash experiments/run_ablation_defense_num_categories.sh 0          # 指定 GPU=0 (单卡顺序执行)
#   bash experiments/run_ablation_defense_num_categories.sh 0,1,2,3    # 多卡并行: 4张卡动态调度任务
#
# 默认类别集合（可按需改）：
#   K=2: shoe,plant
#   K=4: shoe,plant,dish,bowl
#
# 可选环境变量：
#   CONFIG=configs/config.yaml
#   EXPERIMENTS_BASE=output/experiments_output
#   DEFENSE_CACHE_MODE=registry
#   DEFENSE_BATCH_SIZE=2
#   DEFENSE_GRAD_ACCUM=2
#   ATTACK_STEPS=200
#   DEFENSE_STEPS=200
#   EVAL_EVERY_STEPS=10
#   NUM_RENDER=1

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

CONFIG="${CONFIG:-configs/config.yaml}"
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="${EXPERIMENTS_BASE}/ablation_defense_num_categories_${TIMESTAMP}"
mkdir -p "${OUTPUT_ROOT}"
# 复制配置文件到输出目录，避免后续修改影响实验参数
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_ROOT}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"

# 两种防御方法
METHODS=(naive_unlearning geotrap)
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-registry}"
DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-}"
DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-}"

ATTACK_STEPS="${ATTACK_STEPS:-}"
DEFENSE_STEPS="${DEFENSE_STEPS:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-10}"
NUM_RENDER="${NUM_RENDER:-1}"

# 两组类别集合（防御类别数 2/4）
SETS_NAME=(k2 k4)
SETS_CATS=("shoe,plant" "shoe,plant,dish,bowl")

# 生成任务列表：2个类别集合 × 2个方法 = 4个任务
TASKS=()
for method in "${METHODS[@]}"; do
    for i in "${!SETS_NAME[@]}"; do
        TASKS+=("${i}:${method}")
    done
done

TOTAL_TASKS=${#TASKS[@]}

echo "=========================================="
echo "防御类别数消融（target=omni，attack/defense 同 dataset）"
echo "Config: ${CONFIG}"
echo "Defense methods: ${METHODS[*]}"
echo "Defense cache mode: ${DEFENSE_CACHE_MODE}"
echo "Output: ${OUTPUT_ROOT}"
echo "Tasks: ${TOTAL_TASKS} (2/4 categories × 2 methods)"
echo "=========================================="

# 任务执行函数
# 参数: $1=GPU_ID, $2=task_idx
launch_on_gpu() {
    local gpu=$1
    local task_idx=$2
    local task=${TASKS[$task_idx]}

    IFS=':' read -r idx method <<< "$task"
    local name=${SETS_NAME[$idx]}
    local cats=${SETS_CATS[$idx]}

    local tag="defcats_${name}_${method}_omni"
    local log="${OUTPUT_ROOT}/${tag}.log"
    local out_dir="${OUTPUT_ROOT}/${tag}"

    extra_args=()
    if [[ -n "${DEFENSE_BATCH_SIZE}" ]]; then
        extra_args+=(--defense_batch_size "${DEFENSE_BATCH_SIZE}")
    fi
    if [[ -n "${DEFENSE_GRAD_ACCUM}" ]]; then
        extra_args+=(--defense_grad_accumulation_steps "${DEFENSE_GRAD_ACCUM}")
    fi
    if [[ -n "${ATTACK_STEPS}" ]]; then
        extra_args+=(--attack_steps "${ATTACK_STEPS}")
    fi
    if [[ -n "${DEFENSE_STEPS}" ]]; then
        extra_args+=(--defense_steps "${DEFENSE_STEPS}")
    fi

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag} (cats=${cats}, method=${method})"

    if {
        echo "=== GPU ${gpu}: ${tag} ==="
        "${PYTHON}" script/run_pipeline.py \
        --gpu "${gpu}" \
        --config "${CONFIG}" \
        --attack_target_dataset omni \
        --defense_target_dataset omni \
        --categories "${cats}" \
        --defense_method "${method}" \
        --defense_cache_mode "${DEFENSE_CACHE_MODE}" \
        --eval_every_steps "${EVAL_EVERY_STEPS}" \
        --num_render "${NUM_RENDER}" \
        --tag "${tag}" \
        --output_dir "${out_dir}" \
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
    submit_task launch_on_gpu "${i}"
done

wait_all_tasks

FAILED=${#FAILED_TASKS[@]}

echo ""
echo "全部完成！成功: $((TOTAL_TASKS - FAILED)), 失败: ${FAILED}"
echo "结果保存在: ${OUTPUT_ROOT}"
