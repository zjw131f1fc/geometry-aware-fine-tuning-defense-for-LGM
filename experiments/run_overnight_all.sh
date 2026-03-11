#!/usr/bin/env bash
set -euo pipefail

# Overnight runner:
# 1) Re-render GSO (same pose, 100 views) on 2 GPUs
# 2) Run several experiments scripts (some are single-GPU; we run them in 2-GPU "lanes")
# 3) Shutdown the machine
#
# Usage:
#   bash experiments/run_overnight_all.sh           # default GPUs: 0,1
#   bash experiments/run_overnight_all.sh 0,1
#
# Optional env vars (recommended for AutoDL small data disk):
#   SYS_ROOT=./custom-output-dir                   # custom root for caches/outputs (default: repo root)
#   DO_SHUTDOWN=1                                  # 1=shutdown at end, 0=skip shutdown
#   SHUTDOWN_ON_FAIL=1                             # 1=shutdown even if failures
#   NUM_VIEWS=100 RESOLUTION=512

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_LIST="${1:-0,1}"
IFS=',' read -r GPU0 GPU1 _REST <<< "${GPU_LIST}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

if ! [[ "${GPU0}" =~ ^[0-9]+$ && "${GPU1}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] GPU list must be like '0,1' (got '${GPU_LIST}')."
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

SYS_ROOT="${SYS_ROOT:-${ROOT_DIR}}"
DO_SHUTDOWN="${DO_SHUTDOWN:-1}"
SHUTDOWN_ON_FAIL="${SHUTDOWN_ON_FAIL:-1}"

echo "=================================================="
echo "Overnight run started: ${TIMESTAMP}"
echo "Repo: ${ROOT_DIR}"
echo "GPUs: ${GPU0},${GPU1}"
echo "SYS_ROOT: ${SYS_ROOT}"
echo "=================================================="

ensure_symlink() {
  local repo_path="$1"
  local target_path="$2"

  mkdir -p "$(dirname "${repo_path}")"

  if [[ -L "${repo_path}" ]]; then
    # Already a symlink; keep it unless it's broken.
    if [[ ! -e "${repo_path}" ]]; then
      echo "[WARN] Broken symlink: ${repo_path} -> $(readlink "${repo_path}")"
      rm -f "${repo_path}"
    else
      return 0
    fi
  fi

  if [[ -e "${repo_path}" && ! -L "${repo_path}" ]]; then
    local backup="${repo_path}_local_${TIMESTAMP}"
    echo "[INFO] Move aside existing path: ${repo_path} -> ${backup}"
    mv "${repo_path}" "${backup}"
  fi

  echo "[INFO] Symlink: ${repo_path} -> ${target_path}"
  ln -s "${target_path}" "${repo_path}"
}

# Put large outputs/caches on system disk (AutoDL: data disk can be small).
mkdir -p "${SYS_ROOT}/output"
mkdir -p "${SYS_ROOT}/datas/GSO/render_same_pose_all_100v_512"
mkdir -p "${SYS_ROOT}/cache/xdg" "${SYS_ROOT}/cache/torch" "${SYS_ROOT}/cache/hf" "${SYS_ROOT}/cache/pip"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SYS_ROOT}/cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-${SYS_ROOT}/cache/torch}"
export HF_HOME="${HF_HOME:-${SYS_ROOT}/cache/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SYS_ROOT}/cache/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "${MPLCONFIGDIR}"

ensure_symlink "output" "${SYS_ROOT}/output"
ensure_symlink "datas/GSO/render_same_pose_all_100v_512" "${SYS_ROOT}/datas/GSO/render_same_pose_all_100v_512"

mkdir -p "output/overnight_runs"
RUN_ROOT="output/overnight_runs/run_${TIMESTAMP}"
mkdir -p "${RUN_ROOT}"
LOG_DIR="${RUN_ROOT}/_logs"
mkdir -p "${LOG_DIR}"

echo "[INFO] Run dir: ${RUN_ROOT}"

STATUS=0
record_fail() {
  local code="$1"
  local step="$2"
  echo "[FAIL] ${step} (exit=${code})" | tee -a "${RUN_ROOT}/status.txt"
  STATUS=1
}
record_ok() {
  local step="$1"
  echo "[OK] ${step}" | tee -a "${RUN_ROOT}/status.txt"
}

BG_PIDS=()
cleanup_children() {
  echo "[WARN] Caught signal; stopping background jobs..." >&2
  for pid in "${BG_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup_children INT TERM

run_bg() {
  local name="$1"
  local __pid_var="$2"
  shift 2
  local log="${LOG_DIR}/${name}.log"
  echo "[INFO] Start: ${name}" >&2
  echo "[INFO] Log: ${log}" >&2
  (
    echo "=== ${name} ==="
    echo "Time: $(date -Iseconds)"
    echo "Cmd: $*"
    echo ""
    "$@"
  ) > "${log}" 2>&1 &
  local pid=$!
  BG_PIDS+=("${pid}")
  printf -v "${__pid_var}" '%s' "${pid}"
}

wait_job() {
  local pid="$1"
  local name="$2"
  local rc=0
  if wait "${pid}"; then
    record_ok "${name}"
    return 0
  else
    rc=$?
    record_fail "${rc}" "${name}"
    return "${rc}"
  fi
}

echo ""
echo "===================="
echo "1) Render (GSO)"
echo "===================="

export OUT_ROOT="${OUT_ROOT:-datas/GSO/render_same_pose_all_100v_512}"
export INPUT_ROOT="${INPUT_ROOT:-datas/GSO/loose_overlap_clean5}"
export NUM_VIEWS="${NUM_VIEWS:-100}"
export RESOLUTION="${RESOLUTION:-512}"
export NORMALIZE_MAX_DIM="${NORMALIZE_MAX_DIM:-2.7}"
export CYCLES_DEVICE="${CYCLES_DEVICE:-GPU}"
export AUTO_ORIENT="${AUTO_ORIENT:-omni5}"
export AUTO_YAW="${AUTO_YAW:-camera_right}"
export CLEAN_PARTIAL="${CLEAN_PARTIAL:-1}"

MANIFEST0="${MANIFEST0:-datas/GSO/loose_overlap_clean5/manifest_rerender_30shoe.gpu0.tsv}"
MANIFEST1="${MANIFEST1:-datas/GSO/loose_overlap_clean5/manifest_rerender_30shoe.gpu1.tsv}"

if [[ ! -f "${MANIFEST0}" || ! -f "${MANIFEST1}" ]]; then
  echo "[ERROR] Missing manifest(s):"
  echo "  - ${MANIFEST0}"
  echo "  - ${MANIFEST1}"
  exit 1
fi

run_bg "render_gpu${GPU0}" RENDER_PID0 bash script/render_gso_same_pose_worker.sh "${MANIFEST0}" "${GPU0}"
run_bg "render_gpu${GPU1}" RENDER_PID1 bash script/render_gso_same_pose_worker.sh "${MANIFEST1}" "${GPU1}"

wait_job "${RENDER_PID0}" "render_gpu${GPU0}" || true
wait_job "${RENDER_PID1}" "render_gpu${GPU1}" || true

# Quick completeness check: expect each object to have NUM_VIEWS images + transforms.json
CHECK_LOG="${LOG_DIR}/render_check.log"
(
  echo "=== render_check ==="
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "NUM_VIEWS=${NUM_VIEWS}"
  echo "Time: $(date -Iseconds)"
  echo ""

  expected_total=$(( $(wc -l < "${MANIFEST0}") + $(wc -l < "${MANIFEST1}") ))
  echo "Expected objects (rows): ${expected_total}"

  ok=0
  bad=0
  for d in "${OUT_ROOT}"/*; do
    [[ -d "${d}" ]] || continue
    [[ "$(basename "${d}")" == "_logs" ]] && continue
    img_dir="${d}/render/images"
    tf="${d}/render/transforms.json"
    c=0
    if [[ -d "${img_dir}" ]]; then
      c="$(find "${img_dir}" -maxdepth 1 -type f -name 'r_*.png' | wc -l)"
    fi
    if [[ "${c}" -eq "${NUM_VIEWS}" && -f "${tf}" ]]; then
      ok=$((ok + 1))
    else
      bad=$((bad + 1))
      echo "[BAD] $(basename "${d}") images=${c} transforms=$( [[ -f "${tf}" ]] && echo yes || echo no )"
    fi
  done
  echo ""
  echo "Render complete objects: ${ok}"
  echo "Render incomplete objects: ${bad}"
  if [[ "${bad}" -gt 0 ]]; then
    exit 2
  fi
) > "${CHECK_LOG}" 2>&1 || record_fail "$?" "render_check"

echo ""
echo "===================="
echo "2) Experiments"
echo "===================="

# Common knobs (keep as env vars so each experiments/*.sh inherits them)
# Default to "readonly" to avoid writing many full model checkpoints to disk overnight.
# You can override to "registry" if you explicitly want on-disk caching:
#   DEFENSE_CACHE_MODE=registry bash experiments/run_overnight_all.sh 0,1
export DEFENSE_CACHE_MODE="${DEFENSE_CACHE_MODE:-readonly}"
export DEFENSE_BATCH_SIZE="${DEFENSE_BATCH_SIZE:-2}"
export DEFENSE_GRAD_ACCUM="${DEFENSE_GRAD_ACCUM:-2}"
export EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:--1}"

run_script_bg() {
  local __pid_var="$1"
  local name="$2"
  local script_path="$3"
  local gpu="$4"
  run_bg "${name}" "${__pid_var}" bash "${script_path}" "${gpu}"
}

# Lane A on GPU0, Lane B on GPU1 (two-at-a-time to reduce conflicts).
PID_A=""
PID_B=""

run_script_bg PID_A "exp_defense_num_categories_gpu${GPU0}" experiments/run_ablation_defense_num_categories.sh "${GPU0}"
run_script_bg PID_B "exp_all_experiments_gpu${GPU1}" experiments/run_all_experiments.sh "${GPU1}"
wait_job "${PID_A}" "exp_defense_num_categories_gpu${GPU0}" || true
wait_job "${PID_B}" "exp_all_experiments_gpu${GPU1}" || true

run_script_bg PID_A "exp_ablation_attack_gpu${GPU0}" experiments/run_ablation_attack.sh "${GPU0}"
run_script_bg PID_B "exp_ablation_coupling_gpu${GPU1}" experiments/run_ablation_coupling.sh "${GPU1}"
wait_job "${PID_A}" "exp_ablation_attack_gpu${GPU0}" || true
wait_job "${PID_B}" "exp_ablation_coupling_gpu${GPU1}" || true

run_script_bg PID_A "exp_ablation_trap_gpu${GPU0}" experiments/run_ablation_trap.sh "${GPU0}"
wait_job "${PID_A}" "exp_ablation_trap_gpu${GPU0}" || true

echo ""
echo "===================="
echo "3) Finish"
echo "===================="

{
  echo "=================================================="
  echo "Finished at: $(date -Iseconds)"
  echo "STATUS=${STATUS} (0=all ok, 1=has failures)"
  echo "Run dir: ${RUN_ROOT}"
  echo "Logs: ${LOG_DIR}"
  echo "=================================================="
} | tee -a "${RUN_ROOT}/status.txt"

sync || true

if [[ "${DO_SHUTDOWN}" == "1" ]]; then
  if [[ "${STATUS}" != "0" && "${SHUTDOWN_ON_FAIL}" != "1" ]]; then
    echo "[INFO] Skip shutdown because failures happened (STATUS=${STATUS})."
    exit "${STATUS}"
  fi
  echo "[INFO] Shutdown requested. Powering off in 10 seconds..."
  sleep 10
  shutdown -h now || poweroff || halt
fi

exit "${STATUS}"
