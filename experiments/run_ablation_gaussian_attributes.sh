#!/bin/bash
# 高斯属性消融实验：测试5种高斯属性的 w/o
# 测试属性：position, scale, opacity, rotation, color
#
# 用法:
#   bash experiments/run_ablation_gaussian_attributes.sh            # 默认 GPU=0 (单卡顺序执行)
#   bash experiments/run_ablation_gaussian_attributes.sh 0          # 指定 GPU=0 (单卡顺序执行)
#   bash experiments/run_ablation_gaussian_attributes.sh 0,1        # 多卡并行: 2张卡动态调度任务
#
# 环境变量:
#   DEFENSE_CACHE_MODE               # 防御缓存模式（可选，不设置则使用程序默认值）
#   EVAL_EVERY_STEPS                 # 评估间隔步数（可选，不设置则使用程序默认值）
#   TRAP_TASK_SET                    # 任务集选择: wo_attrs(默认) / single_combo(Trap单开&组合补全)

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

# Avoid OpenMP env issues + make matplotlib cache writable
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
if [[ -d "/root/autodl-tmp" ]]; then
    export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/.cache}"
    export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/.cache/torch}"
    export HF_HOME="${HF_HOME:-/root/autodl-tmp/.cache/huggingface}"
    export WANDB_DIR="${WANDB_DIR:-/root/autodl-tmp/.cache/wandb}"
    mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${HF_HOME}" "${WANDB_DIR}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl}"
mkdir -p "${MPLCONFIGDIR}"

# 解析GPU列表
GPU_LIST="${1:-0}"
echo "GPU列表: ${GPU_LIST}"

CONFIG="configs/config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
TRAP_TASK_SET="${TRAP_TASK_SET:-wo_attrs}"
if [[ "${TRAP_TASK_SET}" == "single_combo" ]]; then
    OUTPUT_ROOT="${EXPERIMENTS_BASE}/ablation_trap_single_combo_${TIMESTAMP}"
else
    OUTPUT_ROOT="${EXPERIMENTS_BASE}/ablation_gaussian_attrs_${TIMESTAMP}"
fi

mkdir -p "${OUTPUT_ROOT}"
# 复制配置文件到输出目录，避免后续修改影响实验参数
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_ROOT}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"

echo "=========================================="
echo "高斯属性消融实验"
echo "测试类别: 使用 config.yaml 中的配置"
echo "Config: ${CONFIG} (已复制)"
echo "Output: ${OUTPUT_ROOT}"
echo "TRAP_TASK_SET: ${TRAP_TASK_SET}"
echo "=========================================="

# 精细指标口径：Defense 仅训练 50 step（用于达标步数分析）
DEFENSE_STEPS=50

# ============================================================================
# 任务列表
# ============================================================================

TASKS=()

if [[ "${TRAP_TASK_SET}" == "single_combo" ]]; then
    # Trap 消融补全：单 trap + trap 组合（>=3 traps）
    # 说明：使用 --trap_losses 精确控制启用的 traps；不跳过 baseline 以便计算达标步数。

    # 单 trap
    TASKS+=("single_position:--defense_method geotrap --trap_losses position")
    TASKS+=("single_scale:--defense_method geotrap --trap_losses scale")
    TASKS+=("single_opacity:--defense_method geotrap --trap_losses opacity")
    TASKS+=("single_rotation:--defense_method geotrap --trap_losses rotation")
    TASKS+=("single_color:--defense_method geotrap --trap_losses color")

    # trap 组合（覆盖 >=3 traps 的组合）
    TASKS+=("combo_pos_scale_opacity:--defense_method geotrap --trap_losses position,scale,opacity")
    TASKS+=("combo_scale_opacity_color:--defense_method geotrap --trap_losses scale,opacity,color")
    TASKS+=("combo_pos_rotation_color:--defense_method geotrap --trap_losses position,rotation,color")
    TASKS+=("combo_all5:--defense_method geotrap --trap_losses position,scale,opacity,rotation,color")
else
    # 高斯属性消融：测试5种高斯属性的 w/o
    # 说明：不跳过 baseline，以便计算达标步数。

    # 1. w/o position (关闭 position trap)
    TASKS+=("wo_position:--defense_method geotrap --trap_position_static false")

    # 2. w/o scale (关闭 scale trap)
    TASKS+=("wo_scale:--defense_method geotrap --trap_scale_static false")

    # 3. w/o opacity (关闭 opacity trap)
    TASKS+=("wo_opacity:--defense_method geotrap --trap_opacity_static false")

    # 4. w/o rotation (关闭 rotation trap)
    TASKS+=("wo_rotation:--defense_method geotrap --trap_rotation_static false")

    # 5. w/o color (关闭 color trap)
    TASKS+=("wo_color:--defense_method geotrap --trap_color_static false")
fi

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "总任务数: ${TOTAL_TASKS}"
if [[ "${TRAP_TASK_SET}" == "single_combo" ]]; then
    echo "  - 单 trap: position/scale/opacity/rotation/color"
    echo "  - trap 组合: pos+scale+opacity / scale+opacity+color / pos+rotation+color / all5"
else
    echo "  - w/o position"
    echo "  - w/o scale"
    echo "  - w/o opacity"
    echo "  - w/o rotation"
    echo "  - w/o color"
fi
echo "  - 不跳过 baseline（用于 analysis.postdefense_attack_steps_to_baseline_effect）"
echo ""

# ============================================================================
# 任务执行函数
# ============================================================================

run_task() {
    local gpu=$1
    local task_idx=$2
    local task=${TASKS[$task_idx]}

    IFS=':' read -r tag params <<< "$task"

    local log="${OUTPUT_ROOT}/${tag}.log"
    local output_dir="${OUTPUT_ROOT}/${tag}"

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

    if {
        echo "=== GPU ${gpu}: ${tag} ==="
        echo "Params: ${params}"
        echo ""

        # 执行 pipeline
        "${PYTHON}" script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG}" \
            ${params} \
            ${DEFENSE_CACHE_MODE:+--defense_cache_mode "${DEFENSE_CACHE_MODE}"} \
            ${EVAL_EVERY_STEPS:+--eval_every_steps "${EVAL_EVERY_STEPS}"} \
            --defense_steps "${DEFENSE_STEPS}" \
            --tag "${tag}" \
            --output_dir "${output_dir}"
    } > "${log}" 2>&1; then
        echo "[GPU ${gpu}] 完成: ${tag}"
        return 0
    else
        exit_code=$?
        echo "[GPU ${gpu}] 失败: ${tag} (exit=${exit_code}), log: ${log}"
        return "${exit_code}"
    fi
}

# ============================================================================
# 启动所有任务
# ============================================================================

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

# ============================================================================
# 汇总结果
# ============================================================================

SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"

{
echo ""
echo "=========================================="
echo "高斯属性消融结果汇总"
echo "=========================================="
echo "任务集: ${TRAP_TASK_SET}"
echo ""

printf "%-20s %-12s %-8s %-10s %-10s\n" \
    "实验配置" "达标步数" "阈值step" "阈值PSNR" "阈值LPIPS"
echo "--------------------------------------------------------------"

for task in "${TASKS[@]}"; do
    IFS=':' read -r tag params <<< "$task"

    metrics="${OUTPUT_ROOT}/${tag}/metrics.json"

    if [ -f "$metrics" ]; then
        "${PYTHON}" -c "
import json
with open('${metrics}') as f:
    m = json.load(f)

analysis = m.get('analysis') or {}
steps = analysis.get('postdefense_attack_steps_to_baseline_effect')
baseline = analysis.get('baseline_attack_effect_at_end') or {}
b_step = baseline.get('step')
b_psnr = baseline.get('masked_psnr')
b_lpips = baseline.get('masked_lpips')

name = '${tag}'
def fmt(v, nd=2):
    return 'NA' if v is None else f'{v:.{nd}f}'

steps_s = 'NA' if steps is None else str(int(steps))
b_step_s = 'NA' if b_step is None else str(int(b_step))
print(f'{name:<20s} {steps_s:>12s} {b_step_s:>8s} {fmt(b_psnr,2):>10s} {fmt(b_lpips,4):>10s}')
"
    else
        printf "%-20s (未完成或失败)\n" "${tag}"
    fi
done

echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

echo ""
echo "全部完成！"
