#!/bin/bash
# 从 run_main_all_three.sh 摘出的子集实验：
# 1. gso -> omni 的 shoe / plant 攻击消融（22 个 pipeline）
# 2. gso -> omni 的 shoe / plant single-trap 消融（10 个 pipeline）
# 3. gso -> omni 的 shoe / plant multi-trap 代表链（6 个 pipeline）
#
# 项目命名里的 gso -> omni，当前实际对应：
# - attack_target_dataset = omni
# - defense_target_dataset = gso
# - defense_target 使用全部数据
#
# 固定设置：
# - attack_steps = 400（攻击消融里的 attack_800 / attack_1600 任务除外）
# - defense_steps = 100
# - eval_every_steps = -1（自动 4 段 checkpoint）
# - 全部任务都使用 gso -> omni 口径
#
# 用法:
#   bash experiments/run_ablation_attack_and_traps.sh 0
#   bash experiments/run_ablation_attack_and_traps.sh 0,1,2,3

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source experiments/lib/gpu_scheduler.sh

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${ROOT_DIR}/../venvs/3d-defense/bin/python" ]]; then
        PYTHON="${ROOT_DIR}/../venvs/3d-defense/bin/python"
    else
        PYTHON="python"
    fi
fi

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

GPU_LIST="${1:-0}"
echo "GPU列表: ${GPU_LIST}"

TARGET_CATEGORIES=(shoe plant)
TRAP_COMBO_ATTRS=(position scale opacity rotation color)
TRAP_COMBO_SELECTED_COMBOS=(
    "scale,opacity"
    "position,scale,opacity"
    "position,scale,opacity,color"
)

CONFIG="${CONFIG:-configs/config.yaml}"
ATTACK_STEPS="${ATTACK_STEPS:-400}"
DEFENSE_STEPS="${DEFENSE_STEPS:-100}"
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-registry}"
DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-}"
DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:--1}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/ablation_attack_and_traps_${TIMESTAMP}"

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

ATTACK_ABLATION_DIRNAME="attack_ablation"
CONFIG_GSO2OMNI="${OUTPUT_ROOT}/config_gso2omni_full_defense.yaml"
CONFIG_ATTACK_ABLATION="${OUTPUT_ROOT}/config_attack_ablation_gso2omni.yaml"

prepare_config "${CONFIG_GSO2OMNI}" "omni" "gso" "true" "none"
cp "${CONFIG_GSO2OMNI}" "${CONFIG_ATTACK_ABLATION}"

echo "=========================================="
echo "攻击消融 + Trap 子集实验"
echo "场景: gso -> omni (attack_target_dataset=omni, defense_target_dataset=gso)"
echo "Attack steps: ${ATTACK_STEPS}"
echo "Defense steps: ${DEFENSE_STEPS}"
echo "类别: ${TARGET_CATEGORIES[*]}"
echo "Defense target: full dataset"
echo "Config (attack ablation): ${CONFIG_ATTACK_ABLATION}"
echo "Config (trap): ${CONFIG_GSO2OMNI}"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

ATTACK_ABLATION_TASKS=()
for category in "${TARGET_CATEGORIES[@]}"; do
    ATTACK_ABLATION_TASKS+=("ablation:${category}:robust_default:--categories ${category} --defense_method geotrap --attack_target_dataset omni --defense_target_dataset gso --defense_steps ${DEFENSE_STEPS}")
    ATTACK_ABLATION_TASKS+=("ablation:${category}:mode_lora8:--categories ${category} --defense_method geotrap --attack_target_dataset omni --defense_target_dataset gso --training_mode lora --lora_r 8 --lora_alpha 8 --defense_steps ${DEFENSE_STEPS}")
    ATTACK_ABLATION_TASKS+=("ablation:${category}:mode_lora32:--categories ${category} --defense_method geotrap --attack_target_dataset omni --defense_target_dataset gso --training_mode lora --lora_r 32 --lora_alpha 32 --defense_steps ${DEFENSE_STEPS}")
    ATTACK_ABLATION_TASKS+=("ablation:${category}:mode_full:--categories ${category} --defense_method geotrap --attack_target_dataset omni --defense_target_dataset gso --training_mode full --defense_steps ${DEFENSE_STEPS}")
    ATTACK_ABLATION_TASKS+=("ablation:${category}:optimizer_adamw_3e6:--categories ${category} --defense_method geotrap --attack_target_dataset omni --defense_target_dataset gso --optimizer adamw --lr 3e-6 --defense_steps ${DEFENSE_STEPS}")
    ATTACK_ABLATION_TASKS+=("ablation:${category}:optimizer_adamw_3e4:--categories ${category} --defense_method geotrap --attack_target_dataset omni --defense_target_dataset gso --optimizer adamw --lr 3e-4 --defense_steps ${DEFENSE_STEPS}")
    ATTACK_ABLATION_TASKS+=("ablation:${category}:optimizer_sgd_3e5:--categories ${category} --defense_method geotrap --attack_target_dataset omni --defense_target_dataset gso --optimizer sgd --lr 3e-5 --defense_steps ${DEFENSE_STEPS}")
    ATTACK_ABLATION_TASKS+=("ablation:${category}:optimizer_sgd_3e4:--categories ${category} --defense_method geotrap --attack_target_dataset omni --defense_target_dataset gso --optimizer sgd --lr 3e-4 --defense_steps ${DEFENSE_STEPS}")
    ATTACK_ABLATION_TASKS+=("ablation:${category}:optimizer_sgd_3e3:--categories ${category} --defense_method geotrap --attack_target_dataset omni --defense_target_dataset gso --optimizer sgd --lr 3e-3 --defense_steps ${DEFENSE_STEPS}")
    ATTACK_ABLATION_TASKS+=("ablation:${category}:steps_attack_800:--categories ${category} --defense_method geotrap --attack_target_dataset omni --defense_target_dataset gso --attack_steps 800 --defense_steps ${DEFENSE_STEPS}")
    ATTACK_ABLATION_TASKS+=("ablation:${category}:steps_attack_1600:--categories ${category} --defense_method geotrap --attack_target_dataset omni --defense_target_dataset gso --attack_steps 1600 --defense_steps ${DEFENSE_STEPS}")
done

SINGLE_TRAP_TASKS=()
for category in "${TARGET_CATEGORIES[@]}"; do
    for attr in "${TRAP_COMBO_ATTRS[@]}"; do
        SINGLE_TRAP_TASKS+=("singletrap:${category}:single_${attr}:${attr}")
    done
done

TRAP_COMBO_TASKS=()
for category in "${TARGET_CATEGORIES[@]}"; do
    for combo in "${TRAP_COMBO_SELECTED_COMBOS[@]}"; do
        combo_tag="combo_${combo//,/_}"
        TRAP_COMBO_TASKS+=("trapcombo:${category}:${combo_tag}:${combo}")
    done
done

TASKS=(
    "${ATTACK_ABLATION_TASKS[@]}"
    "${SINGLE_TRAP_TASKS[@]}"
    "${TRAP_COMBO_TASKS[@]}"
)

TOTAL_TASKS=${#TASKS[@]}
echo "总任务数: ${TOTAL_TASKS}"
echo "  - shoe / plant 攻击消融: ${#ATTACK_ABLATION_TASKS[@]} 个任务"
echo "  - single-trap ablation: ${#SINGLE_TRAP_TASKS[@]} 个任务"
echo "  - multi-trap combos: ${#TRAP_COMBO_TASKS[@]} 个任务"
echo ""

run_task() {
    local gpu="$1"
    local task_idx="$2"
    local task="${TASKS[$task_idx]}"
    local task_type
    local field1
    local field2
    local field3
    local category
    local combo_tag
    local single_trap_attr
    local trap_losses
    local ablation_tag
    local params
    local tag
    local output_dir
    local log
    local extra_args

    IFS=':' read -r task_type field1 field2 field3 <<< "${task}"

    extra_args=()
    if [[ -n "${DEFENSE_BATCH_SIZE}" ]]; then
        extra_args+=(--defense_batch_size "${DEFENSE_BATCH_SIZE}")
    fi
    if [[ -n "${DEFENSE_GRAD_ACCUM}" ]]; then
        extra_args+=(--defense_grad_accumulation_steps "${DEFENSE_GRAD_ACCUM}")
    fi

    if [[ "${task_type}" == "singletrap" ]]; then
        category="${field1}"
        combo_tag="${field2}"
        single_trap_attr="${field3}"
        tag="singletrap_${category}_${single_trap_attr}"
        output_dir="${OUTPUT_ROOT}/single_traps/${category}/${combo_tag}"
        log="${OUTPUT_ROOT}/singletrap_${category}_${single_trap_attr}.log"

        mkdir -p "${OUTPUT_ROOT}/single_traps/${category}"

        echo "[GPU ${gpu}] 任务 $((task_idx + 1))/${TOTAL_TASKS}: ${tag}"

        if {
            echo "=== GPU ${gpu}: ${tag} ==="
            "${PYTHON}" script/run_pipeline.py \
                --gpu "${gpu}" \
                --config "${CONFIG_GSO2OMNI}" \
                --categories "${category}" \
                --defense_method geotrap \
                --attack_target_dataset omni \
                --defense_target_dataset gso \
                --attack_steps "${ATTACK_STEPS}" \
                --defense_steps "${DEFENSE_STEPS}" \
                --trap_losses "${single_trap_attr}" \
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
    fi

    if [[ "${task_type}" == "trapcombo" ]]; then
        category="${field1}"
        combo_tag="${field2}"
        trap_losses="${field3}"
        tag="trapcombo_${category}_${combo_tag}"
        output_dir="${OUTPUT_ROOT}/trap_combos/${category}/${combo_tag}"
        log="${OUTPUT_ROOT}/trapcombo_${category}_${combo_tag}.log"

        mkdir -p "${OUTPUT_ROOT}/trap_combos/${category}"

        echo "[GPU ${gpu}] 任务 $((task_idx + 1))/${TOTAL_TASKS}: ${tag}"

        if {
            echo "=== GPU ${gpu}: ${tag} ==="
            "${PYTHON}" script/run_pipeline.py \
                --gpu "${gpu}" \
                --config "${CONFIG_GSO2OMNI}" \
                --categories "${category}" \
                --defense_method geotrap \
                --attack_target_dataset omni \
                --defense_target_dataset gso \
                --attack_steps "${ATTACK_STEPS}" \
                --defense_steps "${DEFENSE_STEPS}" \
                --trap_losses "${trap_losses}" \
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
    fi

    IFS=':' read -r task_type category ablation_tag params <<< "${task}"
    tag="attack_ablation_${category}_${ablation_tag}"
    output_dir="${OUTPUT_ROOT}/${ATTACK_ABLATION_DIRNAME}/${category}/${ablation_tag}"
    log="${OUTPUT_ROOT}/${ATTACK_ABLATION_DIRNAME}_${category}_${ablation_tag}.log"

    mkdir -p "${OUTPUT_ROOT}/${ATTACK_ABLATION_DIRNAME}/${category}"

    echo "[GPU ${gpu}] 任务 $((task_idx + 1))/${TOTAL_TASKS}: ${tag}"

    if {
        echo "=== GPU ${gpu}: ${tag} ==="
        echo "Params: ${params}"
        echo ""
        "${PYTHON}" script/run_pipeline.py \
            --gpu "${gpu}" \
            --config "${CONFIG_ATTACK_ABLATION}" \
            ${params} \
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

echo ""
if [[ "${SCHEDULER_ENABLED}" == "true" ]]; then
    echo "多卡并行已启动：动态调度 ${TOTAL_TASKS} 个任务到 GPU [${GPU_LIST}]"
else
    echo "单卡顺序执行已启动：逐个任务运行"
fi
echo "查看进度: tail -f ${OUTPUT_ROOT}/*.log"
echo ""

for i in $(seq 0 $((TOTAL_TASKS - 1))); do
    submit_task run_task "${i}"
done

scheduler_exit_code=0
wait_all_tasks || scheduler_exit_code=$?
FAILED=${#FAILED_TASKS[@]}

echo ""
echo "全部完成！成功: $((TOTAL_TASKS - FAILED)), 失败: ${FAILED}"

SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"

{
echo ""
echo "=========================================="
echo "攻击消融 + Trap 子集实验汇总"
echo "=========================================="
echo "Attack steps: ${ATTACK_STEPS}"
echo "Defense steps: ${DEFENSE_STEPS}"
echo ""

echo "##########################################"
echo "攻击消融: shoe / plant"
echo "目录: ${OUTPUT_ROOT}/${ATTACK_ABLATION_DIRNAME}"
echo "口径: gso -> omni (attack_target_dataset=omni, defense_target_dataset=gso)"
echo "##########################################"
echo ""

for category in "${TARGET_CATEGORIES[@]}"; do
    echo "=== ${category} ==="
    echo ""
    for task in "${ATTACK_ABLATION_TASKS[@]}"; do
        IFS=':' read -r _ task_category ablation_tag params <<< "${task}"
        [[ "${task_category}" != "${category}" ]] && continue
        metrics="${OUTPUT_ROOT}/${ATTACK_ABLATION_DIRNAME}/${category}/${ablation_tag}/metrics.json"

        echo "--- ${ablation_tag} ---"
        if [[ -f "${metrics}" ]]; then
            "${PYTHON}" script/print_attack_step_report.py --metrics "${metrics}"
        else
            echo "(未完成或失败)"
        fi
        echo ""
    done
done

echo "##########################################"
echo "Single-Trap Ablation: shoe / plant"
echo "目录: ${OUTPUT_ROOT}/single_traps"
echo "口径: gso -> omni, geotrap, defense target uses full gso dataset"
echo "##########################################"
echo ""

current_single_trap_category=""
for task in "${SINGLE_TRAP_TASKS[@]}"; do
    IFS=':' read -r _ category combo_tag trap_losses <<< "${task}"
    if [[ "${current_single_trap_category}" != "${category}" ]]; then
        current_single_trap_category="${category}"
        echo "=== ${category} ==="
        echo ""
    fi

    metrics="${OUTPUT_ROOT}/single_traps/${category}/${combo_tag}/metrics.json"
    echo "--- ${combo_tag} (${trap_losses}) ---"
    if [[ -f "${metrics}" ]]; then
        "${PYTHON}" script/print_attack_step_report.py --metrics "${metrics}"
    else
        echo "(未完成或失败)"
    fi
    echo ""
done

echo "##########################################"
echo "Multi-Trap Combos: shoe / plant"
echo "目录: ${OUTPUT_ROOT}/trap_combos"
echo "口径: gso -> omni, geotrap, 仅保留含 opacity 的 2/3/4-trap 代表链"
echo "链路: scale+opacity -> position+scale+opacity -> position+scale+opacity+color"
echo "##########################################"
echo ""

current_trap_combo_category=""
for task in "${TRAP_COMBO_TASKS[@]}"; do
    IFS=':' read -r _ category combo_tag trap_losses <<< "${task}"
    if [[ "${current_trap_combo_category}" != "${category}" ]]; then
        current_trap_combo_category="${category}"
        echo "=== ${category} ==="
        echo ""
    fi

    metrics="${OUTPUT_ROOT}/trap_combos/${category}/${combo_tag}/metrics.json"
    echo "--- ${combo_tag} (${trap_losses}) ---"
    if [[ -f "${metrics}" ]]; then
        "${PYTHON}" script/print_attack_step_report.py --metrics "${metrics}"
    else
        echo "(未完成或失败)"
    fi
    echo ""
done

echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

exit "${scheduler_exit_code}"
