#!/bin/bash
# 防御机制消融实验：grad surgery, energy梯度裁剪, 陷阱聚合
# Baseline 全启用，然后 w/o 形式去掉每个机制
#
# 单卡顺序执行（不做 GPU 空闲检查）
# 用法:
#   bash experiments/run_ablation_mechanisms.sh            # 默认 GPU=0
#   bash experiments/run_ablation_mechanisms.sh 0          # 指定 GPU=0
#   bash experiments/run_ablation_mechanisms.sh 0,1,2,3    # 兼容旧 GPU_LIST，仅使用第一个 GPU

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

CONFIG="configs/config_geotrap_v34_plant_fusehead_lora_retention.yaml"
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-registry}"
ATTACK_STEPS="${ATTACK_STEPS:-60}"
DEFENSE_STEPS="${DEFENSE_STEPS:-60}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-10}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 默认把实验输出放到 repo 的 output/ 下（本地目录）
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/ablation_mechanisms_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
echo "=========================================="
echo "防御机制消融实验"
echo "测试配置: ${CONFIG}"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

# ============================================================================
# 任务列表（Baseline 全启用，然后单项消融：每次只去掉一个机制）
# ============================================================================

TASKS=()

# Baseline：PCGrad + Energy梯度裁剪 + bottleneck_logsumexp 陷阱聚合 + freeze_head
TASKS+=("baseline_all:--grad_surgery_enabled true --grad_surgery_mode pcgrad --grad_clip_mode energy --grad_energy_mult 5.0 --trap_aggregation_method bottleneck_logsumexp --trap_bottleneck_tau 0.25 --freeze_head true")

# w/o PCGrad (禁用梯度手术)
TASKS+=("wo_pcgrad:--grad_surgery_enabled false --grad_clip_mode energy --grad_energy_mult 5.0 --trap_aggregation_method bottleneck_logsumexp --trap_bottleneck_tau 0.25 --freeze_head true")

# w/o Energy梯度裁剪 (改用普通 norm 裁剪)
TASKS+=("wo_energy_clip:--grad_surgery_enabled true --grad_surgery_mode pcgrad --grad_clip_mode norm --trap_aggregation_method bottleneck_logsumexp --trap_bottleneck_tau 0.25 --freeze_head true")

# w/o bottleneck聚合 (改用简单平均)
TASKS+=("wo_bottleneck_agg:--grad_surgery_enabled true --grad_surgery_mode pcgrad --grad_clip_mode energy --grad_energy_mult 5.0 --trap_aggregation_method mean --freeze_head true")

# w/o freeze_head (不冻结头部卷积层)
TASKS+=("wo_freeze_head:--grad_surgery_enabled true --grad_surgery_mode pcgrad --grad_clip_mode energy --grad_energy_mult 5.0 --trap_aggregation_method bottleneck_logsumexp --trap_bottleneck_tau 0.25 --freeze_head false")

# w/o 所有四个机制
TASKS+=("wo_all_four:--grad_surgery_enabled false --grad_clip_mode norm --trap_aggregation_method mean --freeze_head false")

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "总任务数: ${TOTAL_TASKS}"
echo "  1. Baseline (PCGrad + Energy裁剪 + Bottleneck聚合 + Freeze Head)"
echo "  2. w/o PCGrad"
echo "  3. w/o Energy梯度裁剪"
echo "  4. w/o Bottleneck聚合"
echo "  5. w/o Freeze Head"
echo "  6. w/o 所有四个机制"
echo ""

# ============================================================================
# 任务执行函数
# ============================================================================

run_task() {
    local task_idx=$1
    local task=${TASKS[$task_idx]}

    IFS=':' read -r tag params <<< "$task"

    local log="${OUTPUT_ROOT}/${tag}.log"
    local output_dir="${OUTPUT_ROOT}/${tag}"

    echo "[GPU ${GPU}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

    if {
        echo "=== GPU ${GPU}: ${tag} ==="
        echo "Params: ${params}"
        echo ""

        XFORMERS_DISABLED=1 "${PYTHON}" script/run_pipeline.py \
            --gpu "${GPU}" \
            --config "${CONFIG}" \
            ${params} \
            --attack_steps "${ATTACK_STEPS}" \
            --defense_steps "${DEFENSE_STEPS}" \
            --defense_cache_mode "${DEFENSE_CACHE_MODE}" \
            --eval_every_steps "${EVAL_EVERY_STEPS}" \
            --tag "${tag}" \
            --output_dir "${output_dir}"
    } > "${log}" 2>&1; then
        echo "[GPU ${GPU}] 完成: ${tag}"
        return 0
    fi

    exit_code=$?
    echo "[GPU ${GPU}] 失败: ${tag} (exit=${exit_code}), log: ${log}"
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
echo "防御机制消融结果汇总"
echo "=========================================="
echo ""

printf "%-30s %-15s %-15s %-15s %-15s\n" \
    "实验配置" "Target LPIPS↑" "Target PSNR↓" "Source PSNR↑" "Source LPIPS↓"
echo "----------------------------------------------------------------------------------------------------"

for task in "${TASKS[@]}"; do
    IFS=':' read -r tag params <<< "$task"

    metrics="${OUTPUT_ROOT}/${tag}/metrics.json"

    if [ -f "$metrics" ]; then
        "${PYTHON}" -c "
import json
with open('${metrics}') as f:
    m = json.load(f)

bt = m.get('postdefense_target') or m.get('baseline_target') or {}
bs = m.get('postdefense_source') or m.get('baseline_source') or {}

print(f'${tag:<30s} {bt.get(\"lpips\", 0):>13.4f}   {bt.get(\"psnr\", 0):>13.2f}   {bs.get(\"psnr\", 0):>13.2f}   {bs.get(\"lpips\", 0):>13.4f}')
"
    else
        printf "%-30s (未完成或失败)\n" "${tag}"
    fi
done

echo ""
echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

echo ""
echo "全部完成！"
