#!/bin/bash
# 比较 Trap 聚合方式：mean vs bottleneck_logsumexp
#
# 目标：
# - 固定一组范式（defense_target_dataset, attack_target_dataset）
# - 固定一个类别和可选的物体索引
# - 固定一组 trap 组合（默认单组合：5 种 trap 全开）
# - 比较平均聚合和 bottleneck_logsumexp 聚合的区别
#
# 默认输出 4 个 attack checkpoint（由 run_pipeline.py 自动均分）
#
# 用法:
#   bash experiments/run_compare_trap_aggregation.sh 0
#   CATEGORY=shoe OBJECT_INDICES=1,3,7 \
#   TRAP_LOSSES=opacity,color ATTACK_TARGET_DATASET=omni DEFENSE_TARGET_DATASET=omni \
#   bash experiments/run_compare_trap_aggregation.sh 0
#   CATEGORY=shoe ATTACK_TARGET_DATASET=gso DEFENSE_TARGET_DATASET=omni \
#   TRAP_LOSSES_LIST='opacity,color;opacity,rotation;opacity,scale' NUM_TARGET_LAYERS=80 \
#   bash experiments/run_compare_trap_aggregation.sh 0,1
#
# 主要环境变量:
#   CATEGORY=shoe                         # 默认单类别，对应 data.target / defense.target / attack.malicious_content
#   OBJECT_INDICES=1,3,7                 # 指定物体索引（同数据集时 attack 自动使用剩余物体）
#   DEFENSE_OBJECTS=1,3,7                # OBJECT_INDICES 的兼容别名
#   ATTACK_TARGET_DATASET=omni           # omni / gso / objaverse
#   DEFENSE_TARGET_DATASET=omni          # omni / gso / objaverse
#   TRAP_LOSSES=position,scale,rotation,opacity,color
#                                       # 默认：单组合 5 种 trap 全开；也可手动改成其他单组合
#   TRAP_LOSSES_LIST='opacity,color;opacity,rotation;opacity,scale'
#                                       # 可选：批量跑多个组合；用分号分隔
#   TRAP_COMBO=position+scale+rotation+opacity+color
#                                       # 单组合时显式指定层选择用 trap_combo
#   TRAP_COMBO_LIST='opacity+color;opacity+rotation;scale+opacity'
#                                       # 多组合时显式指定 trap_combo 列表；顺序需与 TRAP_LOSSES_LIST 对齐
#   NUM_TARGET_LAYERS=80                 # 可选；若设置则配合 trap_combo 自动选层
#   ATTACK_STEPS=400                     # 可选；优先于 ATTACK_EPOCHS
#   DEFENSE_STEPS=50                     # 默认 50；优先于 DEFENSE_EPOCHS
#   ATTACK_EPOCHS=                        # 可选；仅在 ATTACK_STEPS 未设置时生效
#   DEFENSE_EPOCHS=                       # 可选；仅在 DEFENSE_STEPS 未设置时生效
#   TRAP_BOTTLENECK_TAU=0.25             # bottleneck_logsumexp 的 tau
#   DEFENSE_CACHE_MODE=none              # 默认 none，避免旧防御缓存混入
#   EVAL_EVERY_STEPS=-1                  # 默认只在自动均分的 4 个 step 节点评估
#   SKIP_BASELINE=0                      # 可选；1=跳过 baseline attack
#   EXPERIMENTS_BASE=output/experiments_output

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

TRAP_LOSSES_RAW="${TRAP_LOSSES-}"
TRAP_LOSSES_LIST_RAW="${TRAP_LOSSES_LIST-}"
CONFIG="${CONFIG:-configs/config.yaml}"
CATEGORY="${CATEGORY:-shoe}"
OBJECT_INDICES="${OBJECT_INDICES:-${DEFENSE_OBJECTS:-}}"
ATTACK_TARGET_DATASET="${ATTACK_TARGET_DATASET:-omni}"
DEFENSE_TARGET_DATASET="${DEFENSE_TARGET_DATASET:-omni}"
TRAP_LOSSES="${TRAP_LOSSES:-position,scale,rotation,opacity,color}"
TRAP_LOSSES_LIST="${TRAP_LOSSES_LIST:-}"
TRAP_COMBO="${TRAP_COMBO:-}"
TRAP_COMBO_LIST="${TRAP_COMBO_LIST:-}"
NUM_TARGET_LAYERS="${NUM_TARGET_LAYERS:-}"
ATTACK_STEPS="${ATTACK_STEPS:-}"
DEFENSE_STEPS="${DEFENSE_STEPS:-100}"
ATTACK_EPOCHS="${ATTACK_EPOCHS:-}"
DEFENSE_EPOCHS="${DEFENSE_EPOCHS:-}"
TRAP_BOTTLENECK_TAU="${TRAP_BOTTLENECK_TAU:-0.25}"
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-none}"
DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-}"
DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-}"
ATTACK_GRAD_ACCUM="${ATTACK_GRAD_ACCUM:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:--1}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/compare_trap_aggregation_${CATEGORY}_${TIMESTAMP}"

mkdir -p "${OUTPUT_ROOT}"
ORIGINAL_CONFIG="${CONFIG}"
CONFIG="${OUTPUT_ROOT}/config.yaml"
cp "${ORIGINAL_CONFIG}" "${CONFIG}"

trim_combo() {
    local combo="$1"
    combo="${combo//[[:space:]]/}"
    printf '%s\n' "${combo}"
}

combo_slug_from_losses() {
    local combo
    combo="$(trim_combo "$1")"
    combo="${combo//,/_}"
    combo="${combo//+/_}"
    printf '%s\n' "${combo}"
}

combo_tag_from_combo() {
    local combo
    combo="$(trim_combo "$1")"
    combo="${combo//,/-}"
    combo="${combo//+/-}"
    printf '%s\n' "${combo}"
}

declare -a TRAP_LOSSES_ITEMS=()
declare -a TRAP_COMBO_ITEMS=()
declare -a COMBO_SLUGS=()

if [[ -n "${TRAP_LOSSES_RAW}" || -z "${TRAP_LOSSES_LIST_RAW}" ]]; then
    if [[ -n "${TRAP_LOSSES_LIST_RAW}" ]]; then
        echo "注意: 已设置 TRAP_LOSSES，忽略 TRAP_LOSSES_LIST，按单组合模式运行。"
    fi
    TRAP_LOSSES_ITEMS+=("$(trim_combo "${TRAP_LOSSES}")")
    if [[ -n "${TRAP_COMBO}" ]]; then
        TRAP_COMBO_ITEMS+=("$(trim_combo "${TRAP_COMBO}")")
    else
        TRAP_COMBO_ITEMS+=("$(trim_combo "${TRAP_LOSSES}")")
        TRAP_COMBO_ITEMS[0]="${TRAP_COMBO_ITEMS[0]//,/+}"
    fi
else
    if [[ -n "${TRAP_COMBO}" ]]; then
        echo "错误: 多组合模式请使用 TRAP_COMBO_LIST，不要单独设置 TRAP_COMBO。"
        exit 1
    fi

    IFS=';' read -r -a raw_trap_losses_items <<< "${TRAP_LOSSES_LIST}"
    for raw_combo in "${raw_trap_losses_items[@]}"; do
        combo="$(trim_combo "${raw_combo}")"
        if [[ -n "${combo}" ]]; then
            TRAP_LOSSES_ITEMS+=("${combo}")
        fi
    done

    if [[ ${#TRAP_LOSSES_ITEMS[@]} -eq 0 ]]; then
        echo "错误: TRAP_LOSSES_LIST 为空，无法创建任务。"
        exit 1
    fi

    if [[ -n "${TRAP_COMBO_LIST}" ]]; then
        IFS=';' read -r -a raw_trap_combo_items <<< "${TRAP_COMBO_LIST}"
        for raw_combo in "${raw_trap_combo_items[@]}"; do
            combo="$(trim_combo "${raw_combo}")"
            if [[ -n "${combo}" ]]; then
                TRAP_COMBO_ITEMS+=("${combo}")
            fi
        done
        if [[ ${#TRAP_COMBO_ITEMS[@]} -ne ${#TRAP_LOSSES_ITEMS[@]} ]]; then
            echo "错误: TRAP_COMBO_LIST 与 TRAP_LOSSES_LIST 数量不一致。"
            echo "      losses=${#TRAP_LOSSES_ITEMS[@]}, combos=${#TRAP_COMBO_ITEMS[@]}"
            exit 1
        fi
    else
        for combo in "${TRAP_LOSSES_ITEMS[@]}"; do
            trap_combo="${combo//,/+}"
            TRAP_COMBO_ITEMS+=("${trap_combo}")
        done
    fi
fi

for combo in "${TRAP_LOSSES_ITEMS[@]}"; do
    COMBO_SLUGS+=("$(combo_slug_from_losses "${combo}")")
done

if [[ -n "${OBJECT_INDICES}" && "${ATTACK_TARGET_DATASET}" != "${DEFENSE_TARGET_DATASET}" ]]; then
    echo "注意: 当前 DataManager 在跨数据集时 defense_target 不按 object_split 划分。"
    echo "      因此 OBJECT_INDICES 在跨数据集场景下不会限制 defense 侧训练物体。"
fi

"${PYTHON}" - "${CONFIG}" "${CATEGORY}" "${ATTACK_TARGET_DATASET}" "${DEFENSE_TARGET_DATASET}" "${OBJECT_INDICES}" <<'PY'
import sys
import yaml

config_path, category, attack_dataset, defense_dataset, object_indices = sys.argv[1:6]

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("data", {})
cfg["data"].setdefault("target", {})
cfg["data"]["target"]["dataset"] = attack_dataset
cfg["data"]["target"]["categories"] = [category]

cfg.setdefault("attack", {})
cfg["attack"].setdefault("malicious_content", {})
cfg["attack"]["malicious_content"]["malicious_categories"] = [category]

cfg.setdefault("defense", {})
cfg["defense"].setdefault("target", {})
cfg["defense"]["target"]["dataset"] = defense_dataset
cfg["defense"]["target"]["categories"] = [category]

if object_indices.strip():
    indices = sorted({int(x.strip()) for x in object_indices.split(",") if x.strip()})
    cfg["data"].setdefault("object_split", {})
    cfg["data"]["object_split"][category] = indices
    cfg["defense"]["target"]["object_split"] = {category: indices}
else:
    cfg["defense"]["target"].pop("object_split", None)

with open(config_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY

echo "=========================================="
echo "Trap聚合对比实验"
echo "Category: ${CATEGORY}"
echo "Attack target dataset:  ${ATTACK_TARGET_DATASET}"
echo "Defense target dataset: ${DEFENSE_TARGET_DATASET}"
if [[ -n "${OBJECT_INDICES}" ]]; then
    echo "Object indices: ${OBJECT_INDICES}"
else
    echo "Object indices: 使用 config 默认 object_split"
fi
echo "Trap combos (${#TRAP_LOSSES_ITEMS[@]}):"
for idx in "${!TRAP_LOSSES_ITEMS[@]}"; do
    echo "  [$((idx+1))] losses=${TRAP_LOSSES_ITEMS[$idx]} | combo=${TRAP_COMBO_ITEMS[$idx]}"
done
if [[ -n "${NUM_TARGET_LAYERS}" ]]; then
    echo "Num target layers: ${NUM_TARGET_LAYERS}"
fi
echo "Config: ${CONFIG} (已复制并定制)"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="

TASKS=()
for idx in "${!TRAP_LOSSES_ITEMS[@]}"; do
    TASKS+=("${idx}:mean")
    TASKS+=("${idx}:bottleneck_logsumexp")
done

run_task() {
    local gpu=$1
    local task_idx=$2
    local task=${TASKS[$task_idx]}
    local combo_idx=${task%%:*}
    local aggregation=${task#*:}
    local combo_losses=${TRAP_LOSSES_ITEMS[$combo_idx]}
    local combo_trap_combo=${TRAP_COMBO_ITEMS[$combo_idx]}
    local combo_slug=${COMBO_SLUGS[$combo_idx]}
    local combo_tag
    local short_tag
    local log
    local output_dir

    combo_tag="$(combo_tag_from_combo "${combo_trap_combo}")"

    if [[ "${aggregation}" == "mean" ]]; then
        short_tag="mean"
    else
        short_tag="logsumexp"
    fi

    log="${OUTPUT_ROOT}/${combo_slug}_${short_tag}.log"
    output_dir="${OUTPUT_ROOT}/${combo_slug}/${short_tag}"

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${#TASKS[@]}: ${combo_losses} / ${aggregation}"

    cmd=(
        "${PYTHON}" script/run_pipeline.py
        --gpu "${gpu}"
        --config "${CONFIG}"
        --categories "${CATEGORY}"
        --defense_method geotrap
        --attack_target_dataset "${ATTACK_TARGET_DATASET}"
        --defense_target_dataset "${DEFENSE_TARGET_DATASET}"
        --trap_losses "${combo_losses}"
        --trap_aggregation_method "${aggregation}"
        --defense_cache_mode "${DEFENSE_CACHE_MODE}"
        --eval_every_steps "${EVAL_EVERY_STEPS}"
        --tag "${short_tag}_${CATEGORY}_${DEFENSE_TARGET_DATASET}def_${ATTACK_TARGET_DATASET}atk_${combo_tag}"
        --output_dir "${output_dir}"
    )

    if [[ -n "${combo_trap_combo}" ]]; then
        cmd+=(--trap_combo "${combo_trap_combo}")
    fi
    if [[ -n "${NUM_TARGET_LAYERS}" ]]; then
        cmd+=(--num_target_layers "${NUM_TARGET_LAYERS}")
    fi
    if [[ -n "${ATTACK_STEPS}" ]]; then
        cmd+=(--attack_steps "${ATTACK_STEPS}")
    elif [[ -n "${ATTACK_EPOCHS}" ]]; then
        cmd+=(--attack_epochs "${ATTACK_EPOCHS}")
    fi
    if [[ -n "${DEFENSE_STEPS}" ]]; then
        cmd+=(--defense_steps "${DEFENSE_STEPS}")
    elif [[ -n "${DEFENSE_EPOCHS}" ]]; then
        cmd+=(--defense_epochs "${DEFENSE_EPOCHS}")
    fi
    if [[ -n "${DEFENSE_BATCH_SIZE}" ]]; then
        cmd+=(--defense_batch_size "${DEFENSE_BATCH_SIZE}")
    fi
    if [[ -n "${DEFENSE_GRAD_ACCUM}" ]]; then
        cmd+=(--defense_grad_accumulation_steps "${DEFENSE_GRAD_ACCUM}")
    fi
    if [[ -n "${ATTACK_GRAD_ACCUM}" ]]; then
        cmd+=(--attack_grad_accumulation_steps "${ATTACK_GRAD_ACCUM}")
    fi
    if [[ "${aggregation}" == "bottleneck_logsumexp" ]]; then
        cmd+=(--trap_bottleneck_tau "${TRAP_BOTTLENECK_TAU}")
    fi
    if [[ "${SKIP_BASELINE}" == "1" ]]; then
        cmd+=(--skip_baseline)
    fi

    if {
        echo "=== GPU ${gpu}: ${combo_losses} / ${aggregation} ==="
        printf 'Command:'
        printf ' %q' "${cmd[@]}"
        echo
        echo
        "${cmd[@]}"
    } > "${log}" 2>&1; then
        echo "[GPU ${gpu}] 完成: ${combo_losses} / ${aggregation}"
        return 0
    fi

    exit_code=$?
    echo "[GPU ${gpu}] 失败: ${combo_losses} / ${aggregation} (exit=${exit_code}), log: ${log}"
    return "${exit_code}"
}

init_gpu_pool "${GPU_LIST}"

echo ""
if [[ "${SCHEDULER_ENABLED}" == "true" ]]; then
    echo "多卡并行已启动：${#TASKS[@]} 个任务将动态调度到 GPU [${GPU_LIST}]"
else
    echo "单卡顺序执行已启动：按组合依次跑 mean / bottleneck_logsumexp"
fi
echo "查看进度: tail -f ${OUTPUT_ROOT}/*.log"
echo ""

for i in $(seq 0 $(( ${#TASKS[@]} - 1 ))); do
    submit_task run_task "${i}"
done

wait_all_tasks
FAILED=${#FAILED_TASKS[@]}

SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"
BASELINE_METRICS=""
for idx in "${!TRAP_LOSSES_ITEMS[@]}"; do
    candidate="${OUTPUT_ROOT}/${COMBO_SLUGS[$idx]}/mean/metrics.json"
    if [[ -f "${candidate}" ]]; then
        BASELINE_METRICS="${candidate}"
        break
    fi
done

{
echo ""
echo "=========================================="
echo "Trap聚合对比结果"
echo "=========================================="
echo "Category: ${CATEGORY}"
echo "Attack target dataset:  ${ATTACK_TARGET_DATASET}"
echo "Defense target dataset: ${DEFENSE_TARGET_DATASET}"
if [[ -n "${OBJECT_INDICES}" ]]; then
    echo "Object indices: ${OBJECT_INDICES}"
fi
echo "Trap combos (${#TRAP_LOSSES_ITEMS[@]}):"
for idx in "${!TRAP_LOSSES_ITEMS[@]}"; do
    echo "  [$((idx+1))] losses=${TRAP_LOSSES_ITEMS[$idx]} | combo=${TRAP_COMBO_ITEMS[$idx]}"
done
echo ""

if [[ -f "${BASELINE_METRICS}" ]]; then
    echo "--- Undefended baseline checkpoints ---"
    "${PYTHON}" script/print_attack_step_report.py --metrics "${BASELINE_METRICS}" --phase baseline
    echo ""
else
    echo "--- Undefended baseline checkpoints ---"
    echo "(未找到已完成的 mean 任务，无法读取 baseline)"
    echo ""
fi

for idx in "${!TRAP_LOSSES_ITEMS[@]}"; do
    combo_losses=${TRAP_LOSSES_ITEMS[$idx]}
    combo_trap_combo=${TRAP_COMBO_ITEMS[$idx]}
    combo_slug=${COMBO_SLUGS[$idx]}
    mean_metrics="${OUTPUT_ROOT}/${combo_slug}/mean/metrics.json"
    lse_metrics="${OUTPUT_ROOT}/${combo_slug}/logsumexp/metrics.json"

    echo "=== Trap combo: ${combo_losses} (${combo_trap_combo}) ==="

    if [[ -f "${mean_metrics}" ]]; then
        echo "--- Mean aggregation (Post-Defense) ---"
        "${PYTHON}" script/print_attack_step_report.py --metrics "${mean_metrics}" --phase postdefense
        echo ""
    else
        echo "--- Mean aggregation (Post-Defense) ---"
        echo "(未完成或失败)"
        echo ""
    fi

    if [[ -f "${lse_metrics}" ]]; then
        echo "--- Bottleneck logsumexp aggregation (Post-Defense) ---"
        "${PYTHON}" script/print_attack_step_report.py --metrics "${lse_metrics}" --phase postdefense
        echo ""
    else
        echo "--- Bottleneck logsumexp aggregation (Post-Defense) ---"
        echo "(未完成或失败)"
        echo ""
    fi
done

if [[ "${FAILED}" -gt 0 ]]; then
    echo "失败任务数: ${FAILED}"
    echo ""
fi

echo "=========================================="
echo "结果保存在: ${OUTPUT_ROOT}"
echo "汇总文件: ${SUMMARY_FILE}"
echo "=========================================="
} | tee "${SUMMARY_FILE}"

echo ""
echo "全部完成！成功: $(( ${#TASKS[@]} - FAILED )), 失败: ${FAILED}"

exit "${FAILED}"
