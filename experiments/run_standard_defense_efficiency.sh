#!/bin/bash
# gso -> omni setting 下的 Defense 效率测量：
# - 项目命名：gso -> omni
# - 当前实际口径：attack_target_dataset=omni, defense_target_dataset=gso
# - 口径：data.use_object_split=true, defense_target 使用全部数据（不设 split_ratio）
# - 覆盖：2 个任务（naive_unlearning / geotrap-5trap）
# - 指标：Training Time / Avg Step Time / Target Avg Batch Time / Peak GPU Memory
#
# 说明：
# - 默认 SKIP_BASELINE=1 和 SKIP_POSTDEFENSE_ATTACK=1，因为这里只测 Phase 2: Defense。
#   跳过 Phase 1 / Phase 3 不会改变 defense 本身的效率指标，只是减少整条 pipeline 的额外耗时。
# - 默认 DEFENSE_CACHE_MODE=none，避免命中缓存后没有真实 defense 训练。
#
# 用法：
#   bash experiments/run_standard_defense_efficiency.sh 0
#   bash experiments/run_standard_defense_efficiency.sh 0,1,2,3
#
# 常用环境变量：
#   ATTACK_STEPS=400
#   DEFENSE_STEPS=100
#   CATEGORY_CSV=shoe,plant,dish,bowl,box
#   OURS_TRAP_LOSSES=position,scale,opacity,rotation,color
#   SKIP_BASELINE=1
#   SKIP_POSTDEFENSE_ATTACK=1
#   DEFENSE_CACHE_MODE=none
#   DEFENSE_BATCH_SIZE=4
#   DEFENSE_GRAD_ACCUM=1

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
ORIGINAL_CONFIG="${CONFIG:-configs/config.yaml}"
ATTACK_STEPS="${ATTACK_STEPS:-400}"
DEFENSE_STEPS="${DEFENSE_STEPS:-100}"
DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-none}"
DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-}"
DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:--1}"
SKIP_BASELINE="${SKIP_BASELINE:-1}"
SKIP_POSTDEFENSE_ATTACK="${SKIP_POSTDEFENSE_ATTACK:-1}"
CATEGORY_CSV="${CATEGORY_CSV:-shoe,plant,dish,bowl,box}"
OURS_TRAP_LOSSES="${OURS_TRAP_LOSSES:-position,scale,opacity,rotation,color}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-output/experiments_output}"
OUTPUT_ROOT="${EXPERIMENTS_BASE}/standard_defense_efficiency_gso2omni_${TIMESTAMP}"
CONFIG_STANDARD="${OUTPUT_ROOT}/config_gso2omni_full_defense.yaml"
TASKS_FILE="${OUTPUT_ROOT}/tasks.tsv"
SUMMARY_FILE="${OUTPUT_ROOT}/summary.txt"
SUMMARY_CSV="${OUTPUT_ROOT}/efficiency_summary.csv"

mkdir -p "${OUTPUT_ROOT}"

prepare_config() {
    local config_path="$1"
    local attack_dataset="$2"
    local defense_dataset="$3"
    local use_object_split="$4"
    local split_ratio="$5"

    cp "${ORIGINAL_CONFIG}" "${config_path}"

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

prepare_config "${CONFIG_STANDARD}" "omni" "gso" "true" "none"

echo "=========================================="
echo "gso -> omni Defense 效率测量"
echo "=========================================="
echo "GPU列表: ${GPU_LIST}"
echo "Categories: ${CATEGORY_CSV}"
echo "Tasks: naive_unlearning / geotrap(5 traps)"
echo "Attack steps: ${ATTACK_STEPS}"
echo "Defense steps: ${DEFENSE_STEPS}"
echo "Attack target dataset: omni"
echo "Defense target dataset: gso"
echo "Defense target: full dataset"
echo "Defense cache mode: ${DEFENSE_CACHE_MODE}"
echo "Skip baseline: ${SKIP_BASELINE}"
echo "Skip post-defense attack: ${SKIP_POSTDEFENSE_ATTACK}"
echo "Ours trap losses: ${OURS_TRAP_LOSSES}"
echo "Config: ${CONFIG_STANDARD}"
echo "Output: ${OUTPUT_ROOT}"
echo "=========================================="
echo ""

TASKS=(
    "gso2omni_naive_unlearning_efficiency:naive_unlearning:${CATEGORY_CSV}:"
    "gso2omni_geotrap_5trap_efficiency:geotrap:${CATEGORY_CSV}:--trap_losses ${OURS_TRAP_LOSSES}"
)

: > "${TASKS_FILE}"
for task in "${TASKS[@]}"; do
    IFS=':' read -r tag method categories params <<< "${task}"
    printf '%s\t%s\t%s\t%s\n' "${tag}" "${method}" "${categories}" "${params}" >> "${TASKS_FILE}"
done

TOTAL_TASKS=${#TASKS[@]}
echo "总任务数: ${TOTAL_TASKS}"
echo ""

run_task() {
    local gpu="$1"
    local task_idx="$2"
    local task="${TASKS[$task_idx]}"
    local categories
    local method
    local tag
    local params
    local output_dir
    local log
    local extra_args
    local cmd

    IFS=':' read -r tag method categories params <<< "${task}"
    output_dir="${OUTPUT_ROOT}/${tag}"
    log="${OUTPUT_ROOT}/${tag}.log"

    extra_args=()
    if [[ -n "${DEFENSE_BATCH_SIZE}" ]]; then
        extra_args+=(--defense_batch_size "${DEFENSE_BATCH_SIZE}")
    fi
    if [[ -n "${DEFENSE_GRAD_ACCUM}" ]]; then
        extra_args+=(--defense_grad_accumulation_steps "${DEFENSE_GRAD_ACCUM}")
    fi

    echo "[GPU ${gpu}] 任务 $((task_idx+1))/${TOTAL_TASKS}: ${tag}"

    cmd=(
        "${PYTHON}" script/run_pipeline.py
        --gpu "${gpu}"
        --config "${CONFIG_STANDARD}"
        --categories "${categories}"
        --defense_method "${method}"
        --attack_target_dataset omni
        --defense_target_dataset gso
        --attack_steps "${ATTACK_STEPS}"
        --defense_steps "${DEFENSE_STEPS}"
        --defense_cache_mode "${DEFENSE_CACHE_MODE}"
        --eval_every_steps "${EVAL_EVERY_STEPS}"
        --measure_efficiency
        --tag "${tag}"
        --output_dir "${output_dir}"
    )
    if [[ -n "${params}" ]]; then
        # params currently only carries a single well-formed CLI fragment defined above.
        # shellcheck disable=SC2206
        param_parts=( ${params} )
        cmd+=("${param_parts[@]}")
    fi
    cmd+=("${extra_args[@]}")
    if [[ "${SKIP_BASELINE}" == "1" ]]; then
        cmd+=(--skip_baseline)
    fi
    if [[ "${SKIP_POSTDEFENSE_ATTACK}" == "1" ]]; then
        cmd+=(--skip_postdefense_attack)
    fi

    if {
        echo "=== GPU ${gpu}: ${tag} ==="
        echo "Method: ${method}"
        echo "Categories: ${categories}"
        if [[ -n "${params}" ]]; then
            echo "Extra params: ${params}"
        fi
        echo ""
        "${cmd[@]}"
    } > "${log}" 2>&1; then
        echo "[GPU ${gpu}] 完成: ${tag}"
        return 0
    else
        exit_code=$?
        echo "[GPU ${gpu}] 失败: ${tag} (exit=${exit_code}), log: ${log}"
        return "${exit_code}"
    fi
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

"${PYTHON}" - "${OUTPUT_ROOT}" "${TASKS_FILE}" "${SUMMARY_CSV}" <<'PY' | tee "${SUMMARY_FILE}"
import csv
import json
import os
import sys

output_root, tasks_file, summary_csv = sys.argv[1:4]


def fmt_float(value, digits):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


rows = []
with open(tasks_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 3:
            tag, method, categories = parts
            params = ""
        elif len(parts) == 4:
            tag, method, categories, params = parts
        else:
            raise ValueError(f"Unexpected task line: {line!r}")
        path = os.path.join(output_root, tag, "defense_efficiency.json")
        row = {
            "tag": tag,
            "method": method,
            "categories": categories,
            "params": params,
            "status": "ok" if os.path.isfile(path) else "missing",
            "steps": None,
            "time_hours": None,
            "avg_step_time_s": None,
            "target_avg_batch_time_s": None,
            "target_batch_count": None,
            "peak_memory_mb": None,
            "path": path,
        }
        if row["status"] == "ok":
            with open(path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            history = data.get("history") or []
            final = history[-1] if history else {}
            peak_memory_mb = data.get("peak_memory_mb")

            row.update({
                "steps": final.get("step"),
                "time_hours": (final.get("elapsed_time") or 0.0) / 3600 if final.get("elapsed_time") is not None else None,
                "avg_step_time_s": final.get("step_time"),
                "target_avg_batch_time_s": final.get("target_avg_batch_time_s"),
                "target_batch_count": final.get("target_batch_count"),
                "peak_memory_mb": peak_memory_mb,
            })
        rows.append(row)

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "tag",
            "method",
            "categories",
            "params",
            "status",
            "steps",
            "time_hours",
            "avg_step_time_s",
            "target_avg_batch_time_s",
            "target_batch_count",
            "peak_memory_mb",
            "path",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("")
print("==========================================")
print("gso -> omni Defense 效率汇总")
print("==========================================")
print("")
print("项目命名: gso -> omni (实际口径: attack_target_dataset=omni, defense_target_dataset=gso)")
print("")
print(f"{'Tag':<40} {'Time(h)':>8} {'Step(s)':>8} {'Target(s)':>10} {'Mem(MB)':>9}")
print("-" * 84)
for row in rows:
    if row["status"] != "ok":
        print(f"{row['tag']:<40} {'FAILED':>8} {'FAILED':>8} {'FAILED':>10} {'FAILED':>9}")
        continue
    print(
        f"{row['tag']:<40} "
        f"{fmt_float(row['time_hours'], 2):>8} "
        f"{fmt_float(row['avg_step_time_s'], 3):>8} "
        f"{fmt_float(row['target_avg_batch_time_s'], 3):>10} "
        f"{fmt_float(row['peak_memory_mb'], 0):>9}"
    )

print("")
print("口径说明")
print("-" * 84)
print("Step(s): defense optimizer-step 的平均时间")
print("Target(s): target micro-batch 的平均时间")

print("")
print("任务说明")
print("-" * 84)
for row in rows:
    extra = f", extra={row['params']}" if row["params"] else ""
    print(f"{row['tag']}: method={row['method']}, categories={row['categories']}{extra}")

print("")
print("结果目录:", output_root)
print("CSV 汇总:", summary_csv)
print("说明: 以上为 gso -> omni 口径下的 Phase 2 defense efficiency；若跳过 Phase 1 / Phase 3，只减少额外 attack 耗时，不改变 defense 指标。")
print("==========================================")
PY

exit "${scheduler_exit_code}"
