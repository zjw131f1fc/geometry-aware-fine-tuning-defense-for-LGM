#!/bin/bash
# 复合主实验：
# 1. Omni -> Omni：不按 object_split，defense_target 随机取 40% 物体
# 2. Omni -> GSO：defense_target 使用全部数据
# 3. GSO -> Omni：defense_target 使用全部数据
#
# 每组主场景都跑 5 个类别 × 2 个方法，共 30 个 pipeline。
# 另外附带：
# - bowl 攻击消融（geotrap）共 11 个 pipeline
# - 防御类别数消融（2/3 类别 × 2 方法）共 4 个 pipeline
# - w/o input noise（shoe/plant × 3 场景，geotrap）共 6 个 pipeline
# - random init vs pretrained（5 类别，attack-only）共 5 个任务
# - shoe/plant 的单 trap 消融（omni->omni，position/scale/opacity/rotation/color）共 10 个 pipeline
# - shoe/plant 的多 trap 代表链（omni->omni，2/3/4-trap）共 6 个 pipeline，放在最后
#
# 固定设置：
# - attack_steps = 400
# - defense_steps = 100
# - eval_every_steps = -1（自动 4 段 checkpoint）
# - random vs pretrained: omni -> omni, attack-only, attack_steps = 400
#
# 用法:
#   bash experiments/run_main_all_three.sh 0
#   bash experiments/run_main_all_three.sh 0,1,2,3

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

CATEGORIES=(shoe plant dish bowl box)
METHODS=(geotrap naive_unlearning)
SCENARIOS=(omni2omni_random40 omni2gso_full_defense gso2omni_full_defense)
ATTACK_ABLATION_CATEGORY="${ATTACK_ABLATION_CATEGORY:-bowl}"
DEFENSE_NUM_CATEGORIES_SET_NAMES=(k2 k3)
DEFENSE_NUM_CATEGORIES_SET_CATS=("bowl,shoe" "shoe,dish,bowl")
NO_INPUT_NOISE_CATEGORIES=(shoe plant)
TRAP_COMBO_CATEGORIES=(shoe plant)
TRAP_COMBO_ATTRS=(position scale opacity rotation color)
TRAP_COMBO_SELECTED_COMBOS=(
    "scale,opacity"
    "position,scale,opacity"
    "position,scale,opacity,color"
)
COMPARE_RANDOM_PRETRAINED_DATASET="${COMPARE_RANDOM_PRETRAINED_DATASET:-omni}"
COMPARE_NUM_RENDER="${COMPARE_NUM_RENDER:-1}"

CONFIG="${CONFIG:-configs/config.yaml}"
ATTACK_STEPS="${ATTACK_STEPS:-400}"
DEFENSE_STEPS="${DEFENSE_STEPS:-100}"
DEFENSE_SPLIT_RATIO="${DEFENSE_SPLIT_RATIO:-0.4}"
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-registry}"
DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-}"
DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:--1}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/main_all_three_${TIMESTAMP}"

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
CONFIG_OMNI2GSO="${OUTPUT_ROOT}/config_omni2gso_full_defense.yaml"
CONFIG_GSO2OMNI="${OUTPUT_ROOT}/config_gso2omni_full_defense.yaml"
CONFIG_ATTACK_ABLATION="${OUTPUT_ROOT}/config_attack_ablation_bowl.yaml"
CONFIG_COMPARE_RANDOM_PRETRAINED="${OUTPUT_ROOT}/config_compare_random_vs_pretrained.yaml"

prepare_config "${CONFIG_OMNI2OMNI}" "omni" "omni" "false" "${DEFENSE_SPLIT_RATIO}"
prepare_config "${CONFIG_OMNI2GSO}" "gso" "omni" "true" "none"
prepare_config "${CONFIG_GSO2OMNI}" "omni" "gso" "true" "none"
cp "${CONFIG}" "${CONFIG_ATTACK_ABLATION}"
prepare_config "${CONFIG_COMPARE_RANDOM_PRETRAINED}" "${COMPARE_RANDOM_PRETRAINED_DATASET}" "omni" "true" "none"

echo "=========================================="
echo "复合主实验：3 组主场景 + bowl 攻击消融 + 防御类别数消融 + w/o input noise + random vs pretrained + single-trap ablation + multi-trap combos"
echo "Attack steps: ${ATTACK_STEPS}"
echo "Defense steps: ${DEFENSE_STEPS}"
echo "Omni -> Omni: use_object_split=false, defense split_ratio=${DEFENSE_SPLIT_RATIO}"
echo "Omni -> GSO: defense_target uses full defense dataset"
echo "GSO -> Omni: defense_target uses full defense dataset"
echo "Bowl 攻击消融: 保持默认 config，仅固定 defense_steps=${DEFENSE_STEPS}"
echo "Random vs Pretrained: attack-only, dataset=${COMPARE_RANDOM_PRETRAINED_DATASET}, attack_steps=${ATTACK_STEPS}"
echo "Config (omni->omni): ${CONFIG_OMNI2OMNI}"
echo "Config (omni->gso): ${CONFIG_OMNI2GSO}"
echo "Config (gso->omni): ${CONFIG_GSO2OMNI}"
echo "Config (attack ablation): ${CONFIG_ATTACK_ABLATION}"
echo "Config (random vs pretrained): ${CONFIG_COMPARE_RANDOM_PRETRAINED}"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

MAIN_TASKS=()
for scenario in "${SCENARIOS[@]}"; do
    for method in "${METHODS[@]}"; do
        for category in "${CATEGORIES[@]}"; do
            MAIN_TASKS+=("main:${scenario}:${category}:${method}")
        done
    done
done

ATTACK_ABLATION_TASKS=()
ATTACK_ABLATION_TASKS+=("ablation:robust:default:--categories ${ATTACK_ABLATION_CATEGORY} --defense_method geotrap --defense_steps ${DEFENSE_STEPS}")
ATTACK_ABLATION_TASKS+=("ablation:mode:lora8:--categories ${ATTACK_ABLATION_CATEGORY} --defense_method geotrap --training_mode lora --lora_r 8 --lora_alpha 8 --defense_steps ${DEFENSE_STEPS}")
ATTACK_ABLATION_TASKS+=("ablation:mode:lora32:--categories ${ATTACK_ABLATION_CATEGORY} --defense_method geotrap --training_mode lora --lora_r 32 --lora_alpha 32 --defense_steps ${DEFENSE_STEPS}")
ATTACK_ABLATION_TASKS+=("ablation:mode:full:--categories ${ATTACK_ABLATION_CATEGORY} --defense_method geotrap --training_mode full --defense_steps ${DEFENSE_STEPS}")
ATTACK_ABLATION_TASKS+=("ablation:optimizer:adamw_3e6:--categories ${ATTACK_ABLATION_CATEGORY} --defense_method geotrap --optimizer adamw --lr 3e-6 --defense_steps ${DEFENSE_STEPS}")
ATTACK_ABLATION_TASKS+=("ablation:optimizer:adamw_3e4:--categories ${ATTACK_ABLATION_CATEGORY} --defense_method geotrap --optimizer adamw --lr 3e-4 --defense_steps ${DEFENSE_STEPS}")
ATTACK_ABLATION_TASKS+=("ablation:optimizer:sgd_3e5:--categories ${ATTACK_ABLATION_CATEGORY} --defense_method geotrap --optimizer sgd --lr 3e-5 --defense_steps ${DEFENSE_STEPS}")
ATTACK_ABLATION_TASKS+=("ablation:optimizer:sgd_3e4:--categories ${ATTACK_ABLATION_CATEGORY} --defense_method geotrap --optimizer sgd --lr 3e-4 --defense_steps ${DEFENSE_STEPS}")
ATTACK_ABLATION_TASKS+=("ablation:optimizer:sgd_3e3:--categories ${ATTACK_ABLATION_CATEGORY} --defense_method geotrap --optimizer sgd --lr 3e-3 --defense_steps ${DEFENSE_STEPS}")
ATTACK_ABLATION_TASKS+=("ablation:steps:attack_800:--categories ${ATTACK_ABLATION_CATEGORY} --defense_method geotrap --attack_steps 800 --defense_steps ${DEFENSE_STEPS}")
ATTACK_ABLATION_TASKS+=("ablation:steps:attack_1600:--categories ${ATTACK_ABLATION_CATEGORY} --defense_method geotrap --attack_steps 1600 --defense_steps ${DEFENSE_STEPS}")

DEFENSE_NUM_CATEGORIES_TASKS=()
for method in "${METHODS[@]}"; do
    for i in "${!DEFENSE_NUM_CATEGORIES_SET_NAMES[@]}"; do
        DEFENSE_NUM_CATEGORIES_TASKS+=("defcats:${DEFENSE_NUM_CATEGORIES_SET_NAMES[$i]}:${method}:${DEFENSE_NUM_CATEGORIES_SET_CATS[$i]}")
    done
done

NO_INPUT_NOISE_TASKS=()
for scenario in "${SCENARIOS[@]}"; do
    for category in "${NO_INPUT_NOISE_CATEGORIES[@]}"; do
        NO_INPUT_NOISE_TASKS+=("no_input_noise:${scenario}:${category}")
    done
done

COMPARE_RANDOM_PRETRAINED_TASKS=()
for category in "${CATEGORIES[@]}"; do
    COMPARE_RANDOM_PRETRAINED_TASKS+=("compare:${category}")
done

SINGLE_TRAP_TASKS=()
for category in "${TRAP_COMBO_CATEGORIES[@]}"; do
    for attr in "${TRAP_COMBO_ATTRS[@]}"; do
        SINGLE_TRAP_TASKS+=("singletrap:${category}:single_${attr}:${attr}")
    done
done

TRAP_COMBO_TASKS=()
for category in "${TRAP_COMBO_CATEGORIES[@]}"; do
    for combo in "${TRAP_COMBO_SELECTED_COMBOS[@]}"; do
        combo_tag="combo_${combo//,/_}"
        TRAP_COMBO_TASKS+=("trapcombo:${category}:${combo_tag}:${combo}")
    done
done

TASKS=(
    "${MAIN_TASKS[@]}"
    "${ATTACK_ABLATION_TASKS[@]}"
    "${DEFENSE_NUM_CATEGORIES_TASKS[@]}"
    "${NO_INPUT_NOISE_TASKS[@]}"
    "${COMPARE_RANDOM_PRETRAINED_TASKS[@]}"
    "${SINGLE_TRAP_TASKS[@]}"
    "${TRAP_COMBO_TASKS[@]}"
)

TOTAL_TASKS=${#TASKS[@]}
echo "总任务数: ${TOTAL_TASKS}"
echo "  - 主场景: ${#MAIN_TASKS[@]} 个任务"
echo "  - bowl 攻击消融: ${#ATTACK_ABLATION_TASKS[@]} 个任务"
echo "  - 防御类别数消融: ${#DEFENSE_NUM_CATEGORIES_TASKS[@]} 个任务"
echo "  - w/o input noise: ${#NO_INPUT_NOISE_TASKS[@]} 个任务"
echo "  - random vs pretrained: ${#COMPARE_RANDOM_PRETRAINED_TASKS[@]} 个任务"
echo "  - single-trap ablation: ${#SINGLE_TRAP_TASKS[@]} 个任务"
echo "  - multi-trap combos: ${#TRAP_COMBO_TASKS[@]} 个任务"
echo ""

scenario_config() {
    case "$1" in
        omni2omni_random40)
            printf '%s\n' "${CONFIG_OMNI2OMNI}"
            ;;
        omni2gso_full_defense)
            printf '%s\n' "${CONFIG_OMNI2GSO}"
            ;;
        gso2omni_full_defense)
            printf '%s\n' "${CONFIG_GSO2OMNI}"
            ;;
        *)
            return 1
            ;;
    esac
}

scenario_attack_dataset() {
    case "$1" in
        omni2omni_random40) printf '%s\n' "omni" ;;
        omni2gso_full_defense) printf '%s\n' "gso" ;;
        gso2omni_full_defense) printf '%s\n' "omni" ;;
        *) return 1 ;;
    esac
}

scenario_defense_dataset() {
    case "$1" in
        omni2omni_random40) printf '%s\n' "omni" ;;
        omni2gso_full_defense) printf '%s\n' "omni" ;;
        gso2omni_full_defense) printf '%s\n' "gso" ;;
        *) return 1 ;;
    esac
}

scenario_dir() {
    case "$1" in
        omni2omni_random40) printf '%s\n' "omni_to_omni_random40" ;;
        omni2gso_full_defense) printf '%s\n' "omni_to_gso_full_defense" ;;
        gso2omni_full_defense) printf '%s\n' "gso_to_omni_full_defense" ;;
        *) return 1 ;;
    esac
}

scenario_label() {
    case "$1" in
        omni2omni_random40) printf '%s\n' "Omni -> Omni (random 40% defense)" ;;
        omni2gso_full_defense) printf '%s\n' "Omni -> GSO (full defense target)" ;;
        gso2omni_full_defense) printf '%s\n' "GSO -> Omni (full defense target)" ;;
        *) return 1 ;;
    esac
}

baseline_metrics_path() {
    local scenario="$1"
    local category="$2"
    local scenario_subdir
    local method
    local tag
    local metrics

    scenario_subdir="$(scenario_dir "${scenario}")"
    for method in "${METHODS[@]}"; do
        tag="${scenario}_${category}_${method}"
        metrics="${OUTPUT_ROOT}/${scenario_subdir}/${tag}/metrics.json"
        if [[ -f "${metrics}" ]]; then
            printf '%s\n' "${metrics}"
            return 0
        fi
    done
    return 1
}

run_task() {
    local gpu="$1"
    local task_idx="$2"
    local task="${TASKS[$task_idx]}"
    local task_type
    local field1
    local field2
    local field3
    local scenario
    local category
    local method
    local section
    local ablation_tag
    local params
    local config_path
    local attack_dataset
    local defense_dataset
    local scenario_subdir
    local tag
    local output_dir
    local log
    local compare_category
    local compare_base_dir
    local set_name
    local cats
    local no_input_noise_scenario
    local no_input_noise_category
    local single_trap_attr
    local combo_tag
    local trap_losses

    IFS=':' read -r task_type field1 field2 field3 <<< "${task}"

    extra_args=()
    if [[ -n "${DEFENSE_BATCH_SIZE}" ]]; then
        extra_args+=(--defense_batch_size "${DEFENSE_BATCH_SIZE}")
    fi
    if [[ -n "${DEFENSE_GRAD_ACCUM}" ]]; then
        extra_args+=(--defense_grad_accumulation_steps "${DEFENSE_GRAD_ACCUM}")
    fi

    if [[ "${task_type}" == "main" ]]; then
        scenario="${field1}"
        category="${field2}"
        method="${field3}"
        config_path="$(scenario_config "${scenario}")"
        attack_dataset="$(scenario_attack_dataset "${scenario}")"
        defense_dataset="$(scenario_defense_dataset "${scenario}")"
        scenario_subdir="$(scenario_dir "${scenario}")"
        tag="${scenario}_${category}_${method}"
        output_dir="${OUTPUT_ROOT}/${scenario_subdir}/${tag}"
        log="${OUTPUT_ROOT}/${scenario_subdir}_${category}_${method}.log"

        mkdir -p "${OUTPUT_ROOT}/${scenario_subdir}"

        echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

        if {
            echo "=== GPU ${gpu}: ${tag} ==="
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
    fi

    if [[ "${task_type}" == "defcats" ]]; then
        set_name="${field1}"
        method="${field2}"
        cats="${field3}"
        tag="defcats_${set_name}_${method}_omni"
        output_dir="${OUTPUT_ROOT}/defense_num_categories/${tag}"
        log="${OUTPUT_ROOT}/defense_num_categories_${set_name}_${method}.log"

        mkdir -p "${OUTPUT_ROOT}/defense_num_categories"

        echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

        if {
            echo "=== GPU ${gpu}: ${tag} ==="
            "${PYTHON}" script/run_pipeline.py \
                --gpu "${gpu}" \
                --config "${CONFIG_OMNI2OMNI}" \
                --categories "${cats}" \
                --defense_method "${method}" \
                --attack_target_dataset omni \
                --defense_target_dataset omni \
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
    fi

    if [[ "${task_type}" == "no_input_noise" ]]; then
        no_input_noise_scenario="${field1}"
        no_input_noise_category="${field2}"
        config_path="$(scenario_config "${no_input_noise_scenario}")"
        attack_dataset="$(scenario_attack_dataset "${no_input_noise_scenario}")"
        defense_dataset="$(scenario_defense_dataset "${no_input_noise_scenario}")"
        scenario_subdir="$(scenario_dir "${no_input_noise_scenario}")"
        tag="${no_input_noise_scenario}_${no_input_noise_category}_geotrap_wo_input_noise"
        output_dir="${OUTPUT_ROOT}/wo_input_noise/${scenario_subdir}/${no_input_noise_category}"
        log="${OUTPUT_ROOT}/wo_input_noise_${no_input_noise_scenario}_${no_input_noise_category}.log"

        mkdir -p "${OUTPUT_ROOT}/wo_input_noise/${scenario_subdir}"

        echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

        if {
            echo "=== GPU ${gpu}: ${tag} ==="
            "${PYTHON}" script/run_pipeline.py \
                --gpu "${gpu}" \
                --config "${config_path}" \
                --categories "${no_input_noise_category}" \
                --defense_method geotrap \
                --attack_target_dataset "${attack_dataset}" \
                --defense_target_dataset "${defense_dataset}" \
                --attack_steps "${ATTACK_STEPS}" \
                --defense_steps "${DEFENSE_STEPS}" \
                --input_noise_enabled false \
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

        echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

        if {
            echo "=== GPU ${gpu}: ${tag} ==="
            "${PYTHON}" script/run_pipeline.py \
                --gpu "${gpu}" \
                --config "${CONFIG_OMNI2OMNI}" \
                --categories "${category}" \
                --defense_method geotrap \
                --attack_target_dataset omni \
                --defense_target_dataset omni \
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

    if [[ "${task_type}" == "singletrap" ]]; then
        category="${field1}"
        combo_tag="${field2}"
        single_trap_attr="${field3}"
        tag="singletrap_${category}_${single_trap_attr}"
        output_dir="${OUTPUT_ROOT}/single_traps/${category}/${combo_tag}"
        log="${OUTPUT_ROOT}/singletrap_${category}_${single_trap_attr}.log"

        mkdir -p "${OUTPUT_ROOT}/single_traps/${category}"

        echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

        if {
            echo "=== GPU ${gpu}: ${tag} ==="
            "${PYTHON}" script/run_pipeline.py \
                --gpu "${gpu}" \
                --config "${CONFIG_OMNI2OMNI}" \
                --categories "${category}" \
                --defense_method geotrap \
                --attack_target_dataset omni \
                --defense_target_dataset omni \
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

    if [[ "${task_type}" == "compare" ]]; then
        compare_category="${field1}"
        tag="compare_random_vs_pretrained_${compare_category}"
        compare_base_dir="${OUTPUT_ROOT}/random_vs_pretrained/${compare_category}"
        log="${OUTPUT_ROOT}/random_vs_pretrained_${compare_category}.log"

        mkdir -p "${compare_base_dir}"

        echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

        if {
            echo "=== GPU ${gpu}: ${tag} ==="
            "${PYTHON}" script/compare_random_vs_pretrained.py \
                --config "${CONFIG_COMPARE_RANDOM_PRETRAINED}" \
                --gpu "${gpu}" \
                --categories "${compare_category}" \
                --attack_steps "${ATTACK_STEPS}" \
                --eval_every_steps "${EVAL_EVERY_STEPS}" \
                --num_render "${COMPARE_NUM_RENDER}" \
                --output_dir "${compare_base_dir}"
        } > "${log}" 2>&1; then
            echo "[GPU ${gpu}] 完成: ${tag}"
            return 0
        fi

        exit_code=$?
        echo "[GPU ${gpu}] 失败: ${tag} (exit=${exit_code}), log: ${log}"
        return "${exit_code}"
    fi

    IFS=':' read -r task_type section ablation_tag params <<< "${task}"
    tag="attack_ablation_${section}_${ablation_tag}"
    output_dir="${OUTPUT_ROOT}/attack_ablation_bowl/${section}_${ablation_tag}"
    log="${OUTPUT_ROOT}/attack_ablation_bowl_${section}_${ablation_tag}.log"

    mkdir -p "${OUTPUT_ROOT}/attack_ablation_bowl"

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

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
echo "复合主实验汇总"
echo "=========================================="
echo "Attack steps: ${ATTACK_STEPS}"
echo "Defense steps: ${DEFENSE_STEPS}"
echo ""

for scenario in "${SCENARIOS[@]}"; do
    scenario_subdir="$(scenario_dir "${scenario}")"
    echo "##########################################"
    echo "场景: $(scenario_label "${scenario}")"
    echo "目录: ${OUTPUT_ROOT}/${scenario_subdir}"
    echo "##########################################"
    echo ""

    for category in "${CATEGORIES[@]}"; do
        echo "=== ${category} ==="
        echo ""

        for method in "${METHODS[@]}"; do
            tag="${scenario}_${category}_${method}"
            metrics="${OUTPUT_ROOT}/${scenario_subdir}/${tag}/metrics.json"

            if [[ -f "${metrics}" ]]; then
                echo "--- ${method} (Post-Defense) ---"
                "${PYTHON}" script/print_attack_step_report.py --metrics "${metrics}" --phase postdefense
            else
                echo "--- ${method} --- (未完成)"
            fi
            echo ""
        done

        baseline_metrics="$(baseline_metrics_path "${scenario}" "${category}" || true)"
        if [[ -n "${baseline_metrics}" ]]; then
            echo "--- Undefended (baseline checkpoints) ---"
            "${PYTHON}" script/print_attack_step_report.py --metrics "${baseline_metrics}" --phase baseline
            echo ""
        fi
    done
done

echo "##########################################"
echo "攻击消融: bowl"
echo "目录: ${OUTPUT_ROOT}/attack_ablation_bowl"
echo "##########################################"
echo ""

for task in "${ATTACK_ABLATION_TASKS[@]}"; do
    IFS=':' read -r _ section ablation_tag params <<< "${task}"
    metrics="${OUTPUT_ROOT}/attack_ablation_bowl/${section}_${ablation_tag}/metrics.json"

    if [[ -f "${metrics}" ]]; then
        echo "--- ${section}_${ablation_tag} ---"
        "${PYTHON}" script/print_attack_step_report.py --metrics "${metrics}"
        echo ""
    else
        echo "--- ${section}_${ablation_tag} ---"
        echo "(未完成或失败)"
        echo ""
    fi
done

echo "##########################################"
echo "防御类别数消融"
echo "目录: ${OUTPUT_ROOT}/defense_num_categories"
echo "口径: omni -> omni, use_object_split=false, defense split_ratio=${DEFENSE_SPLIT_RATIO}"
echo "##########################################"
echo ""

for method in "${METHODS[@]}"; do
    echo "=== ${method} ==="
    echo ""
    for i in "${!DEFENSE_NUM_CATEGORIES_SET_NAMES[@]}"; do
        set_name="${DEFENSE_NUM_CATEGORIES_SET_NAMES[$i]}"
        cats="${DEFENSE_NUM_CATEGORIES_SET_CATS[$i]}"
        tag="defcats_${set_name}_${method}_omni"
        metrics="${OUTPUT_ROOT}/defense_num_categories/${tag}/metrics.json"

        echo "--- ${tag} (cats=${cats}) ---"
        if [[ -f "${metrics}" ]]; then
            "${PYTHON}" script/print_attack_step_report.py --metrics "${metrics}"
        else
            echo "(未完成或失败)"
        fi
        echo ""
    done
done

echo "##########################################"
echo "w/o Input Noise: shoe / plant"
echo "目录: ${OUTPUT_ROOT}/wo_input_noise"
echo "口径: geotrap, input_noise_enabled=false"
echo "##########################################"
echo ""

for scenario in "${SCENARIOS[@]}"; do
    scenario_subdir="$(scenario_dir "${scenario}")"
    echo "=== $(scenario_label "${scenario}") ==="
    echo ""
    for category in "${NO_INPUT_NOISE_CATEGORIES[@]}"; do
        tag="${scenario}_${category}_geotrap_wo_input_noise"
        metrics="${OUTPUT_ROOT}/wo_input_noise/${scenario_subdir}/${category}/metrics.json"

        echo "--- ${category} ---"
        if [[ -f "${metrics}" ]]; then
            "${PYTHON}" script/print_attack_step_report.py --metrics "${metrics}"
        else
            echo "(未完成或失败)"
        fi
        echo ""
    done
done

echo "##########################################"
echo "Random Init vs Pretrained: 5 类别"
echo "目录: ${OUTPUT_ROOT}/random_vs_pretrained"
echo "##########################################"
echo ""

for category in "${CATEGORIES[@]}"; do
    compare_base_dir="${OUTPUT_ROOT}/random_vs_pretrained/${category}"
    compare_dir="$(find "${compare_base_dir}" -mindepth 1 -maxdepth 1 -type d -name 'compare_*' 2>/dev/null | sort | tail -n 1)"

    echo "--- ${category} ---"
    if [[ -n "${compare_dir}" && -f "${compare_dir}/comparison_data.json" ]]; then
        echo "结果目录: ${compare_dir}"
        "${PYTHON}" - "${compare_dir}/comparison_data.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

rand_target = data.get("random_init", {}).get("target_metrics", {}) or {}
pre_target = data.get("pretrained", {}).get("target_metrics", {}) or {}

rand_lpips = rand_target.get("lpips", 0.0)
pre_lpips = pre_target.get("lpips", 0.0)
rand_psnr = rand_target.get("psnr", 0.0)
pre_psnr = pre_target.get("psnr", 0.0)

print(f"Random target:     LPIPS={rand_lpips:.4f}, PSNR={rand_psnr:.2f}")
print(f"Pretrained target: LPIPS={pre_lpips:.4f}, PSNR={pre_psnr:.2f}")
print(f"Delta (pre-rand):  LPIPS={pre_lpips-rand_lpips:+.4f}, PSNR={pre_psnr-rand_psnr:+.2f}")
PY
        echo "报告: ${compare_dir}/comparison_report.txt"
echo "图像: ${compare_dir}/comparison_plot.png"
        echo ""
    else
        echo "(未完成或失败)"
        echo ""
    fi
done

echo "##########################################"
echo "Single-Trap Ablation: shoe / plant"
echo "目录: ${OUTPUT_ROOT}/single_traps"
echo "口径: omni -> omni, geotrap, 仅补 1-trap 结果"
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
echo "口径: omni -> omni, geotrap, 仅保留含 opacity 的 2/3/4-trap 代表链"
echo "链路: scale+opacity -> position+scale+opacity -> position+scale+opacity+color"
echo "说明: 用于对照 single_opacity 与主实验默认 5-trap geotrap，验证多 trap 下 source 更稳"
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
