#!/usr/bin/env bash
set -euo pipefail

# Dual-GPU launcher:
# - Split manifest into odd/even rows
# - Launch two worker processes on GPU 0 and GPU 1
# - Leave processes running in background (nohup)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="${MANIFEST:-datas/GSO/loose_overlap_clean5/manifest.tsv}"
OUT_ROOT="${OUT_ROOT:-datas/GSO/render_same_pose_all_100v_512}"
SPLIT_DIR="$OUT_ROOT/_split"
mkdir -p "$SPLIT_DIR" logs

if [[ ! -f "$MANIFEST" ]]; then
  echo "[ERROR] Manifest not found: $MANIFEST"
  exit 1
fi

WORKER="script/render_gso_same_pose_worker.sh"
if [[ ! -f "$WORKER" ]]; then
  echo "[ERROR] Worker script not found: $WORKER"
  exit 1
fi

GPU0_MANIFEST="$SPLIT_DIR/manifest_gpu0.tsv"
GPU1_MANIFEST="$SPLIT_DIR/manifest_gpu1.tsv"
awk 'NR%2==1' "$MANIFEST" > "$GPU0_MANIFEST"
awk 'NR%2==0' "$MANIFEST" > "$GPU1_MANIFEST"

echo "[INFO] Split complete"
echo "[INFO] gpu0 rows: $(wc -l < "$GPU0_MANIFEST")"
echo "[INFO] gpu1 rows: $(wc -l < "$GPU1_MANIFEST")"

echo "[INFO] Launching GPU0 worker..."
nohup bash "$WORKER" "$GPU0_MANIFEST" 0 > logs/render_gpu0.out 2>&1 &
PID0=$!

echo "[INFO] Launching GPU1 worker..."
nohup bash "$WORKER" "$GPU1_MANIFEST" 1 > logs/render_gpu1.out 2>&1 &
PID1=$!

echo "$PID0" > "$OUT_ROOT/_gpu0.pid"
echo "$PID1" > "$OUT_ROOT/_gpu1.pid"

echo "[OK] Started"
echo "GPU0 PID=$PID0 log=logs/render_gpu0.out"
echo "GPU1 PID=$PID1 log=logs/render_gpu1.out"
echo
echo "Quick monitor commands:"
echo "  tail -n 30 logs/render_gpu0.out"
echo "  tail -n 30 logs/render_gpu1.out"
echo "  nvidia-smi"
