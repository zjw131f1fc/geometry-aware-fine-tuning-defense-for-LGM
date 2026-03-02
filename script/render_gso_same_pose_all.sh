#!/usr/bin/env bash
set -euo pipefail

# Full render for GSO loose_overlap_clean5:
# - 100 views per object
# - same camera poses as OmniObject template transforms
# - resumable (skip if already complete)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BLENDER="/mnt/huangjiaxin/blender-3.2.2-linux-x64/blender"
TEMPLATE="datas/omniobject3d___OmniObject3D-New/raw/blender_renders/pitaya_010/render/transforms.json"
MANIFEST="datas/GSO/loose_overlap_clean5/manifest.tsv"
INPUT_ROOT="datas/GSO/loose_overlap_clean5"
OUT_ROOT="datas/GSO/render_same_pose_all_100v_512"
LOG_ROOT="$OUT_ROOT/_logs"

mkdir -p "$LOG_ROOT"

if [[ ! -x "$BLENDER" ]]; then
  echo "[ERROR] Blender not executable: $BLENDER"
  exit 1
fi
if [[ ! -f "$TEMPLATE" ]]; then
  echo "[ERROR] Missing pose template: $TEMPLATE"
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "[ERROR] Missing manifest: $MANIFEST"
  exit 1
fi

TOTAL=$(wc -l < "$MANIFEST")
COUNT=0

echo "[INFO] Start full rendering"
echo "[INFO] Manifest: $MANIFEST"
echo "[INFO] Output:   $OUT_ROOT"
echo "[INFO] Total objects: $TOTAL"

while IFS=$'\t' read -r MODEL CLS; do
  [[ -z "${MODEL:-}" || -z "${CLS:-}" ]] && continue
  COUNT=$((COUNT + 1))

  OBJ_PATH="$INPUT_ROOT/$CLS/$MODEL/meshes/model.obj"
  OBJ_UID="${CLS}_${MODEL}"
  OBJ_OUT="$OUT_ROOT/$OBJ_UID"
  IMG_DIR="$OBJ_OUT/render/images"
  LOG_FILE="$LOG_ROOT/$OBJ_UID.log"

  if [[ ! -f "$OBJ_PATH" ]]; then
    echo "[$COUNT/$TOTAL][MISSING] $OBJ_UID :: $OBJ_PATH"
    continue
  fi

  IMG_COUNT=0
  if [[ -d "$IMG_DIR" ]]; then
    IMG_COUNT=$(find "$IMG_DIR" -maxdepth 1 -type f -name 'r_*.png' | wc -l)
  fi

  if [[ "$IMG_COUNT" -eq 100 && -f "$OBJ_OUT/render/transforms.json" ]]; then
    echo "[$COUNT/$TOTAL][SKIP] $OBJ_UID already complete"
    continue
  fi

  rm -rf "$OBJ_OUT"
  echo "[$COUNT/$TOTAL][RUN ] $OBJ_UID"

  if xvfb-run -a "$BLENDER" -b -P tools/render_omni_format.py -- \
    --object_path "$OBJ_PATH" \
    --output_dir "$OUT_ROOT" \
    --object_uid "$OBJ_UID" \
    --num_views 100 \
    --resolution 512 \
    --fast \
    --pose_template_json "$TEMPLATE" \
    > "$LOG_FILE" 2>&1; then
    echo "[$COUNT/$TOTAL][DONE] $OBJ_UID"
  else
    echo "[$COUNT/$TOTAL][FAIL] $OBJ_UID (see $LOG_FILE)"
  fi
done < "$MANIFEST"

echo "[INFO] Rendering loop finished."
