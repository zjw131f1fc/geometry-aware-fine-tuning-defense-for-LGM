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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "${MPLCONFIGDIR}"

# 解析GPU列表
GPU_LIST="${1:-0}"
echo "GPU列表: ${GPU_LIST}"

CONFIG="configs/config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/ablation_gaussian_attrs_${TIMESTAMP}"

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
echo "=========================================="

# ============================================================================
# 任务列表：测试5种高斯属性的 w/o（全部跳过 baseline）
# ============================================================================

TASKS=()

# 1. w/o position (关闭 position trap)
TASKS+=("wo_position:--defense_method geotrap --trap_position_static false --skip_baseline")

# 2. w/o scale (关闭 scale trap)
TASKS+=("wo_scale:--defense_method geotrap --trap_scale_static false --skip_baseline")

# 3. w/o opacity (关闭 opacity trap)
TASKS+=("wo_opacity:--defense_method geotrap --trap_opacity_static false --skip_baseline")

# 4. w/o rotation (关闭 rotation trap)
TASKS+=("wo_rotation:--defense_method geotrap --trap_rotation_static false --skip_baseline")

# 5. w/o color (关闭 color trap)
TASKS+=("wo_color:--defense_method geotrap --trap_color_static false --skip_baseline")

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "总任务数: ${TOTAL_TASKS}"
echo "  - w/o position"
echo "  - w/o scale"
echo "  - w/o opacity"
echo "  - w/o rotation"
echo "  - w/o color"
echo "  - 全部跳过 baseline"
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
            --tag "${tag}" \
            --output_dir "${output_dir}"
    } > "${log}" 2>&1; then
        echo "[GPU ${gpu}] 完成: ${tag}"
        return 0
    fi

    exit_code=$?
    echo "[GPU ${gpu}] 失败: ${tag} (exit=${exit_code}), log: ${log}"
    return "${exit_code}"
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
echo ""

for task in "${TASKS[@]}"; do
    IFS=':' read -r tag params <<< "$task"

    metrics="${OUTPUT_ROOT}/${tag}/metrics.json"

    echo "--- ${tag} ---"

    if [ -f "$metrics" ]; then
        "${PYTHON}" -c "
import json
with open('${metrics}') as f:
    m = json.load(f)

target = m.get('postdefense_target', {})
print(f'  Target LPIPS: {target.get(\"lpips\", 0):.4f}')
print(f'  Target PSNR:  {target.get(\"psnr\", 0):.2f}')
"
    else
        echo "  (未完成或失败)"
    fi
    echo ""
done

echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

echo ""
echo "全部完成！"


