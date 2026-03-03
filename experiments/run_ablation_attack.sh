#!/bin/bash
# 攻击实验消融（Section 5.3）：测试类别=coconut
# 包含所有消融实验，优先分配语义偏转攻击
#
# 单卡顺序执行（不做 GPU 空闲检查）
# 用法:
#   bash experiments/run_ablation_attack.sh            # 默认 GPU=0
#   bash experiments/run_ablation_attack.sh 0          # 指定 GPU=0
#   bash experiments/run_ablation_attack.sh 0,1,2,3    # 兼容旧 GPU_LIST，仅使用第一个 GPU

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
OUTPUT_ROOT="${EXPERIMENTS_BASE}/ablation_attack_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
echo "=========================================="
echo "攻击实验消融 (Section 5.3)"
echo "测试类别: coconut"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

# 实验配置
TEST_CAT="coconut"
ATTACK_EPOCHS=5
DEFENSE_EPOCHS=60
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-registry}"
DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-}"
DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-10}"

# ============================================================================
# 任务列表（优先级排序：语义偏转 > 其他）
# ============================================================================

TASKS=()

# ========== 优先级1: Section 5.3.1 微调方式 ==========
# 全量微调 (baseline): 已有结果 (LPIPS=0.4655, PSNR=13.5424)，跳过
# LoRA r=8, alpha=16
TASKS+=("5.3.1:finetune_lora_r8:--categories ${TEST_CAT} --defense_method geotrap --training_mode lora --lora_r 8 --lora_alpha 16 --attack_epochs ${ATTACK_EPOCHS}")
# LoRA r=16, alpha=32
TASKS+=("5.3.1:finetune_lora_r16:--categories ${TEST_CAT} --defense_method geotrap --training_mode lora --lora_r 16 --lora_alpha 32 --attack_epochs ${ATTACK_EPOCHS}")

# ========== 优先级3: Section 5.3.2 优化器与学习率 (成对: AdamW vs SGD, 小/中/大) ==========
# AdamW lr=1e-5 (小)
TASKS+=("5.3.2:optimizer_adamw_1e5:--categories ${TEST_CAT} --defense_method geotrap --lr 1e-5 --attack_epochs ${ATTACK_EPOCHS}")
# SGD+momentum(0.9) lr=1e-4 (小)
TASKS+=("5.3.2:optimizer_sgd_1e4:--categories ${TEST_CAT} --defense_method geotrap --optimizer sgd --lr 1e-4 --attack_epochs ${ATTACK_EPOCHS}")
# AdamW lr=5e-5 (中, baseline): 已有结果 (LPIPS=0.4655, PSNR=13.5424)，跳过
# SGD+momentum(0.9) lr=1e-3 (中)
TASKS+=("5.3.2:optimizer_sgd_1e3:--categories ${TEST_CAT} --defense_method geotrap --optimizer sgd --lr 1e-3 --attack_epochs ${ATTACK_EPOCHS}")
# AdamW lr=2e-4 (大)
TASKS+=("5.3.2:optimizer_adamw_2e4:--categories ${TEST_CAT} --defense_method geotrap --lr 2e-4 --attack_epochs ${ATTACK_EPOCHS}")
# SGD+momentum(0.9) lr=1e-2 (大)
TASKS+=("5.3.2:optimizer_sgd_1e2:--categories ${TEST_CAT} --defense_method geotrap --optimizer sgd --lr 1e-2 --attack_epochs ${ATTACK_EPOCHS}")

# ========== 优先级4: Section 5.3.3 攻击时长 ==========
# 2 epochs
TASKS+=("5.3.3:duration_2ep:--categories ${TEST_CAT} --defense_method geotrap --attack_epochs 2")
# 5 epochs (baseline): 已有结果 (LPIPS=0.4655, PSNR=13.5424)，跳过
# 10 epochs
TASKS+=("5.3.3:duration_10ep:--categories ${TEST_CAT} --defense_method geotrap --attack_epochs 10")
# 20 epochs
TASKS+=("5.3.3:duration_20ep:--categories ${TEST_CAT} --defense_method geotrap --attack_epochs 20")

TOTAL_TASKS=${#TASKS[@]}
echo ""
echo "总任务数: ${TOTAL_TASKS}"
echo "  - Section 5.3.1 (微调方式): 2 个任务 (baseline已有结果)"
echo "  - Section 5.3.2 (优化器): 5 个任务"
echo "  - Section 5.3.3 (攻击时长): 3 个任务"
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

        # 执行 pipeline
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
            --defense_epochs "${DEFENSE_EPOCHS}" \
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
echo "攻击实验消融结果汇总 (Section 5.3)"
echo "=========================================="
echo ""

# ========== Section 5.3.1: 微调方式 ==========
echo "=== Section 5.3.1: 微调方式 ==="
echo ""

for task in "${TASKS[@]}"; do
    IFS=':' read -r section tag params <<< "$task"

    if [ "$section" != "5.3.1" ]; then
        continue
    fi

    metrics="${OUTPUT_ROOT}/${section}_${tag}/metrics.json"

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

# ========== Section 5.3.2: 优化器与学习率 ==========
echo "=== Section 5.3.2: 优化器与学习率 ==="
echo ""

for task in "${TASKS[@]}"; do
    IFS=':' read -r section tag params <<< "$task"

    if [ "$section" != "5.3.2" ]; then
        continue
    fi

    metrics="${OUTPUT_ROOT}/${section}_${tag}/metrics.json"

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

# ========== Section 5.3.3: 攻击时长 ==========
echo "=== Section 5.3.3: 攻击时长 ==="
echo ""

for task in "${TASKS[@]}"; do
    IFS=':' read -r section tag params <<< "$task"

    if [ "$section" != "5.3.3" ]; then
        continue
    fi

    metrics="${OUTPUT_ROOT}/${section}_${tag}/metrics.json"

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
