#!/bin/bash
# 仅重跑两条怀疑 baseline cache 有问题的主实验（只看 GeoTrap）：
# 1. Omni -> Omni：dish
# 2. GSO -> Omni：box
#
# 口径与 run_main_all_three.sh 保持一致：
# - attack_steps = 400
# - defense_steps = 100
# - eval_every_steps = -1（自动 4 段 checkpoint）
# - omni -> omni: use_object_split=false, defense.target.split_ratio=0.4
# - gso -> omni（项目命名）实际为 attack_target_dataset=omni, defense_target_dataset=gso
#
# 默认额外开启：
# - --no_baseline_cache
#   目的是强制重跑 Phase 1，避免再次命中 baseline cache
#
# 用法:
#   bash experiments/run_rerun_baseline_cache_suspects.sh 0
#   bash experiments/run_rerun_baseline_cache_suspects.sh 0,1

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source experiments/lib/gpu_scheduler.sh

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${ROOT_DIR}/../venvs/3d-defense/bin/python" ]]; then
        PYTHON="${ROOT_DIR}/../venvs/3d-defense/bin/python"
    else
        PYTHON="python"
    fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "${MPLCONFIGDIR}"

GPU_LIST="${1:-0}"
echo "GPU列表: ${GPU_LIST}"

CONFIG="${CONFIG:-configs/config.yaml}"
ATTACK_STEPS="${ATTACK_STEPS:-400}"
DEFENSE_STEPS="${DEFENSE_STEPS:-100}"
DEFENSE_SPLIT_RATIO="${DEFENSE_SPLIT_RATIO:-0.4}"
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-registry}"
DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-}"
DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:--1}"
NO_BASELINE_CACHE="${NO_BASELINE_CACHE:-1}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/rerun_baseline_cache_suspects_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"

prepare_config() {
    local config_path="$1"
    local attack_dataset="$2"
    local defense_dataset="$3"
    local use_object_split="$4"
    local split_ratio="$5"

    cp "${CONFIG}" "${config_path}"

    "${PYTHON}" - "${config_path}" "${attack_dataset}" "${defense_dataset}" "${use_object_split}" "${split_ratio}" <<'PY'
import sys
import yaml

config_path, attack_dataset, defense_dataset, use_object_split_raw, split_ratio_raw = sys.argv[1:6]

use_object_split = str(use_object_split_raw).strip().lower() in ("1", "true", "yes", "on")
split_ratio_raw = str(split_ratio_raw).strip().lower()
split_ratio = None if split_ratio_raw in ("", "none", "null") else float(split_ratio_raw)

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("data", {})
cfg["data"]["use_object_split"] = use_object_split
cfg["data"].setdefault("target", {})
cfg["data"]["target"]["dataset"] = attack_dataset

cfg.setdefault("defense", {})
cfg["defense"].setdefault("target", {})
cfg["defense"]["target"]["dataset"] = defense_dataset
cfg["defense"]["target"].pop("object_split", None)

if split_ratio is None:
    cfg["defense"]["target"].pop("split_ratio", None)
else:
    cfg["defense"]["target"]["split_ratio"] = split_ratio

with open(config_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY
}

CONFIG_OMNI2OMNI="${OUTPUT_ROOT}/config_omni2omni_random40.yaml"
CONFIG_GSO2OMNI="${OUTPUT_ROOT}/config_gso2omni_full_defense.yaml"

prepare_config "${CONFIG_OMNI2OMNI}" "omni" "omni" "false" "${DEFENSE_SPLIT_RATIO}"
prepare_config "${CONFIG_GSO2OMNI}" "omni" "gso" "true" "none"

TASKS=(
    "omni2omni_random40:dish:geotrap:${CONFIG_OMNI2OMNI}:omni:omni:omni_to_omni_random40"
    "gso2omni_full_defense:box:geotrap:${CONFIG_GSO2OMNI}:omni:gso:gso_to_omni_full_defense"
)

TOTAL_TASKS=${#TASKS[@]}

echo "=========================================="
echo "重跑 baseline cache 可疑任务（GeoTrap only）"
echo "任务数: ${TOTAL_TASKS}"
echo "Attack steps: ${ATTACK_STEPS}"
echo "Defense steps: ${DEFENSE_STEPS}"
echo "Eval every steps: ${EVAL_EVERY_STEPS}"
echo "Defense cache mode: ${DEFENSE_CACHE_MODE}"
echo "No baseline cache: ${NO_BASELINE_CACHE}"
echo "Config (omni->omni): ${CONFIG_OMNI2OMNI}"
echo "Config (gso->omni): ${CONFIG_GSO2OMNI}"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="
echo ""

run_task() {
    local gpu="$1"
    local task_idx="$2"
    local task="${TASKS[$task_idx]}"
    local scenario
    local category
    local method
    local config_path
    local attack_dataset
    local defense_dataset
    local scenario_subdir
    local tag
    local output_dir
    local log

    IFS=':' read -r scenario category method config_path attack_dataset defense_dataset scenario_subdir <<< "${task}"

    tag="${scenario}_${category}_${method}"
    output_dir="${OUTPUT_ROOT}/${scenario_subdir}/${tag}"
    log="${OUTPUT_ROOT}/${scenario_subdir}_${category}_${method}.log"

    mkdir -p "${OUTPUT_ROOT}/${scenario_subdir}"

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
        if [[ "${NO_BASELINE_CACHE}" == "1" || "${NO_BASELINE_CACHE}" == "true" || "${NO_BASELINE_CACHE}" == "TRUE" ]]; then
            extra_args+=(--no_baseline_cache)
        fi

        "${PYTHON}" script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${config_path}" \
            --categories "${category}" \
            --defense_method "${method}" \
            --attack_target_dataset "${attack_dataset}" \
            --defense_target_dataset "${defense_dataset}" \
            --attack_steps "${ATTACK_STEPS}" \
            --defense_steps "${DEFENSE_STEPS}" \
            --defense_cache_mode "${DEFENSE_CACHE_MODE}" \
            --eval_every_steps "${EVAL_EVERY_STEPS}" \
            --tag "${tag}" \
            --output_dir "${output_dir}" \
            "${extra_args[@]}"
    } > "${log}" 2>&1; then
        echo "[GPU ${gpu}] 完成: ${tag}"
        return 0
    fi

    exit_code=$?
    echo "[GPU ${gpu}] 失败: ${tag} (exit=${exit_code}), log: ${log}"
    return "${exit_code}"
}

init_gpu_pool "${GPU_LIST}"

echo "开始提交任务..."
for i in $(seq 0 $((TOTAL_TASKS-1))); do
    submit_task run_task "$i"
done

echo ""
wait_all_tasks
scheduler_exit_code=$?

SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"

{
echo "=========================================="
echo "汇总结果"
echo "=========================================="

for task in "${TASKS[@]}"; do
    IFS=':' read -r scenario category method _ _ _ scenario_subdir <<< "${task}"
    tag="${scenario}_${category}_${method}"
    metrics="${OUTPUT_ROOT}/${scenario_subdir}/${tag}/metrics.json"

    echo ""
    echo "=== ${tag} ==="
    if [[ -f "${metrics}" ]]; then
        "${PYTHON}" script/print_attack_step_report.py --metrics "${metrics}"
        echo ""
        echo "--- Undefended (baseline checkpoints) ---"
        "${PYTHON}" script/print_attack_step_report.py --metrics "${metrics}" --phase baseline
    else
        echo "未完成"
    fi
done

echo ""
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

exit "${scheduler_exit_code}"
