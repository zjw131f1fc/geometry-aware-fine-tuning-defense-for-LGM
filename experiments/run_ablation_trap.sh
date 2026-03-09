#!/bin/bash
# Trap组合消融实验（Section 5.1）：garlic
# 5.1.1 单属性Trap (5种) × 1类别 = 5
# 5.1.2 两两组合Trap (10种) × 1类别 = 10
# 5.1.3 三属性组合Trap (10种) × 1类别 = 10（当前未启用）
# 5.1.4 四属性组合Trap (5种) × 1类别 = 5
# 5.1.5 五属性全组合Trap (1种) × 1类别 = 1
# 当前启用: 21个实验
#
# 单卡顺序执行（不做 GPU 空闲检查）
# 用法:
#   bash experiments/run_ablation_trap.sh            # 默认 GPU=0
#   bash experiments/run_ablation_trap.sh 0          # 指定 GPU=0
#   bash experiments/run_ablation_trap.sh 0,1,2,3    # 兼容旧 GPU_LIST，仅使用第一个 GPU

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

# 单卡选择：优先使用 CLI 参数，其次 GPU_ID/GPU 环境变量，最后默认 0
GPU_ID="${GPU_ID:-${GPU:-0}}"
if [ $# -ge 1 ]; then
    GPU_ARG="$1"
    if [[ "${GPU_ARG}" == *","* ]]; then
        GPU_ID="${GPU_ARG%%,*}"
        echo "检测到 GPU_LIST='${GPU_ARG}'，单卡模式仅使用第一个 GPU: ${GPU_ID}"
    else
        GPU_ID="${GPU_ARG}"
    fi
fi

if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID 必须是非负整数，当前: '${GPU_ID}'"
    exit 1
fi

GPU="${GPU_ID}"
echo "单卡顺序执行: GPU=${GPU}"

CONFIG="configs/config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 默认把实验输出放到 repo 的 output/ 下（本地目录）
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/ablation_trap_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
# 复制配置文件到输出目录，避免后续修改影响实验参数
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_ROOT}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"

echo "=========================================="
echo "Trap组合消融实验 (Section 5.1)"
echo "测试类别: garlic"
echo "Config: ${CONFIG} (已复制)"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

CATEGORIES=(garlic)
ATTACK_EPOCHS=2
DEFENSE_EPOCHS=15
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-registry}"
DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-}"
DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-10}"

# ============================================================================
# 任务列表
# ============================================================================

TASKS=()

# 5.1.1 单属性Trap
for cat in "${CATEGORIES[@]}"; do
    TASKS+=("5.1.1:single_position_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.1:single_scale_${cat}:--categories ${cat} --defense_method geotrap --trap_losses scale --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.1:single_rotation_${cat}:--categories ${cat} --defense_method geotrap --trap_losses rotation --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.1:single_opacity_${cat}:--categories ${cat} --defense_method geotrap --trap_losses opacity --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.1:single_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
done

# 5.1.2 两两组合Trap
for cat in "${CATEGORIES[@]}"; do
    TASKS+=("5.1.2:combo_position_scale_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,scale --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.2:combo_position_rotation_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,rotation --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.2:combo_position_opacity_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,opacity --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.2:combo_position_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.2:combo_scale_rotation_${cat}:--categories ${cat} --defense_method geotrap --trap_losses scale,rotation --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.2:combo_scale_opacity_${cat}:--categories ${cat} --defense_method geotrap --trap_losses scale,opacity --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.2:combo_scale_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses scale,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.2:combo_rotation_opacity_${cat}:--categories ${cat} --defense_method geotrap --trap_losses rotation,opacity --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.2:combo_rotation_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses rotation,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.2:combo_opacity_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses opacity,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
done

# # 5.1.3 三属性组合Trap
# for cat in "${CATEGORIES[@]}"; do
#     TASKS+=("5.1.3:triple_position_scale_rotation_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,scale,rotation --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
#     TASKS+=("5.1.3:triple_position_scale_opacity_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,scale,opacity --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
#     TASKS+=("5.1.3:triple_position_scale_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,scale,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
#     TASKS+=("5.1.3:triple_position_rotation_opacity_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,rotation,opacity --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
#     TASKS+=("5.1.3:triple_position_rotation_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,rotation,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
#     TASKS+=("5.1.3:triple_position_opacity_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,opacity,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
#     TASKS+=("5.1.3:triple_scale_rotation_opacity_${cat}:--categories ${cat} --defense_method geotrap --trap_losses scale,rotation,opacity --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
#     TASKS+=("5.1.3:triple_scale_rotation_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses scale,rotation,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
#     TASKS+=("5.1.3:triple_scale_opacity_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses scale,opacity,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
#     TASKS+=("5.1.3:triple_rotation_opacity_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses rotation,opacity,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
# done

# # 5.1.4 四属性组合Trap
for cat in "${CATEGORIES[@]}"; do
    TASKS+=("5.1.4:quad_position_scale_rotation_opacity_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,scale,rotation,opacity --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.4:quad_position_scale_rotation_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,scale,rotation,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.4:quad_position_scale_opacity_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,scale,opacity,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.4:quad_position_rotation_opacity_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,rotation,opacity,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
    TASKS+=("5.1.4:quad_scale_rotation_opacity_color_${cat}:--categories ${cat} --defense_method geotrap --trap_losses scale,rotation,opacity,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
done

# # 5.1.5 五属性全组合Trap
for cat in "${CATEGORIES[@]}"; do
    TASKS+=("5.1.5:penta_all_${cat}:--categories ${cat} --defense_method geotrap --trap_losses position,scale,rotation,opacity,color --attack_epochs ${ATTACK_EPOCHS} --defense_epochs ${DEFENSE_EPOCHS}")
done

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "总任务数: ${TOTAL_TASKS}"
echo "  - Section 5.1.1 (单属性): 5 × 1类别 = 5 个任务"
echo "  - Section 5.1.2 (两两组合): 10 × 1类别 = 10 个任务"
echo "  - Section 5.1.3 (三属性组合): 当前未启用"
echo "  - Section 5.1.4 (四属性组合): 5 × 1类别 = 5 个任务"
echo "  - Section 5.1.5 (五属性全组合): 1 × 1类别 = 1 个任务"
echo ""

# ============================================================================
# 任务执行函数
# ============================================================================

run_task() {
    local task_idx=$1
    local task=${TASKS[$task_idx]}

    IFS=':' read -r section tag params <<< "$task"

    local log="${OUTPUT_ROOT}/${section}_${tag}.log"
    local output_dir="${OUTPUT_ROOT}/${section}_${tag}"

    echo "[GPU ${GPU}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${section} - ${tag}"

    if {
        echo "=== GPU ${GPU}: ${section} - ${tag} ==="
        echo "Params: ${params}"
        echo ""

        extra_args=()
        if [[ -n "${DEFENSE_BATCH_SIZE}" ]]; then
            extra_args+=(--defense_batch_size "${DEFENSE_BATCH_SIZE}")
        fi
        if [[ -n "${DEFENSE_GRAD_ACCUM}" ]]; then
            extra_args+=(--defense_grad_accumulation_steps "${DEFENSE_GRAD_ACCUM}")
        fi
        "${PYTHON}" script/run_pipeline.py \
            --gpu "${GPU}" \
            --config "${CONFIG}" \
            ${params} \
            --defense_cache_mode "${DEFENSE_CACHE_MODE}" \
            --eval_every_steps "${EVAL_EVERY_STEPS}" \
            --tag "${section}_${tag}" \
            --output_dir "${output_dir}" \
            "${extra_args[@]}"
    } > "${log}" 2>&1; then
        echo "[GPU ${GPU}] 完成: ${section} - ${tag}"
        return 0
    fi

    exit_code=$?
    echo "[GPU ${GPU}] 失败: ${section} - ${tag} (exit=${exit_code}), log: ${log}"
    return "${exit_code}"
}

# ============================================================================
# 启动所有任务
# ============================================================================

echo ""
echo "顺序执行已启动：单卡逐个任务运行（不检查 GPU 空闲）"
echo "查看进度: tail -f ${OUTPUT_ROOT}/*.log"
echo ""

FAILED=0
for i in $(seq 0 $((TOTAL_TASKS-1))); do
    if ! run_task "${i}"; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "全部完成！成功: $((TOTAL_TASKS - FAILED)), 失败: ${FAILED}"

# ============================================================================
# 汇总结果
# ============================================================================

SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"

{
echo ""
echo "=========================================="
echo "Trap组合消融结果汇总 (Section 5.1)"
echo "=========================================="

for section_label in "5.1.1:单属性Trap" "5.1.2:两两组合Trap" "5.1.3:三属性组合Trap" "5.1.4:四属性组合Trap" "5.1.5:五属性全组合Trap"; do
    IFS=':' read -r section section_name <<< "$section_label"
    echo ""
    echo "=== Section ${section}: ${section_name} ==="
    echo ""

    for task in "${TASKS[@]}"; do
        IFS=':' read -r tsection tag params <<< "$task"
        [ "$tsection" != "$section" ] && continue

        metrics="${OUTPUT_ROOT}/${tsection}_${tag}/metrics.json"
        echo "--- ${tag} ---"

        if [ -f "$metrics" ]; then
            "${PYTHON}" script/print_attack_step_report.py --metrics "$metrics"
        else
            echo "  (未完成或失败)"
        fi
        echo ""
    done
done

echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

echo ""
echo "全部完成！"
