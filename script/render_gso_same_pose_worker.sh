#!/usr/bin/env bash
set -euo pipefail

# Single-GPU worker:
# - Reads a manifest.tsv subset (model<TAB>class)
# - Renders 100 views with OmniObject template poses
# - Resumable: skip if already complete

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <manifest_tsv> <gpu_id>"
  exit 1
fi

MANIFEST="$1"
GPU_ID="$2"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BLENDER="${BLENDER_PATH:-$ROOT_DIR/../blender-3.2.2-linux-x64/blender}"
TEMPLATE_ROOT="datas/omniobject3d___OmniObject3D-New/raw/blender_renders"
TEMPLATE_FIXED="${TEMPLATE_JSON:-}"
INPUT_ROOT="${INPUT_ROOT:-datas/GSO/loose_overlap_clean5}"
OUT_ROOT="${OUT_ROOT:-datas/GSO/render_same_pose_all_100v_512}"
NUM_VIEWS="${NUM_VIEWS:-100}"
RESOLUTION="${RESOLUTION:-512}"
NORMALIZE_MAX_DIM="${NORMALIZE_MAX_DIM:-2.7}"
CYCLES_DEVICE="${CYCLES_DEVICE:-AUTO}"
AUTO_ORIENT="${AUTO_ORIENT:-omni5}"
AUTO_YAW="${AUTO_YAW:-camera_right}"
YAW_FLIP_DEFAULT="${YAW_FLIP-}"
YAW_OFFSET_DEG="${YAW_OFFSET_DEG:-0}"
EXTRA_PITCH_DEG="${EXTRA_PITCH_DEG:-0}"
EXTRA_ROLL_DEG_DEFAULT="${EXTRA_ROLL_DEG-}"
CLEAN_PARTIAL="${CLEAN_PARTIAL:-1}"  # 1: remove incomplete object dir and re-render

LOG_ROOT="$OUT_ROOT/_logs"
mkdir -p "$LOG_ROOT"

if [[ ! -x "$BLENDER" ]]; then
  echo "[ERROR] Blender not executable: $BLENDER"
  exit 1
fi
if [[ -n "$TEMPLATE_FIXED" && ! -f "$TEMPLATE_FIXED" ]]; then
  echo "[ERROR] Missing TEMPLATE_JSON: $TEMPLATE_FIXED"
  exit 1
fi
if [[ ! -d "$TEMPLATE_ROOT" ]]; then
  echo "[ERROR] Missing template root dir: $TEMPLATE_ROOT"
  exit 1
fi

# Auto-generate manifest if missing (model<TAB>class). Deterministic order.
if [[ ! -f "$MANIFEST" ]]; then
  echo "[WARN] Missing manifest: $MANIFEST"
  echo "[INFO] Generating manifest from: $INPUT_ROOT"
  mkdir -p "$(dirname "$MANIFEST")"
  : > "$MANIFEST"

  while IFS= read -r -d '' cls_path; do
    cls="$(basename "$cls_path")"
    while IFS= read -r -d '' model_path; do
      model="$(basename "$model_path")"
      obj="$model_path/meshes/model.obj"
      if [[ -f "$obj" ]]; then
        printf "%s\t%s\n" "$model" "$cls" >> "$MANIFEST"
      fi
    done < <(find "$cls_path" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
  done < <(find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

  echo "[OK] Manifest generated: $MANIFEST ($(wc -l < "$MANIFEST") rows)"
fi

TOTAL=$(wc -l < "$MANIFEST")
COUNT=0
DONE=0
SKIP=0
FAIL=0
MISS=0

echo "[INFO][GPU$GPU_ID] Start worker"
echo "[INFO][GPU$GPU_ID] Manifest: $MANIFEST ($TOTAL rows)"
echo "[INFO][GPU$GPU_ID] Output:   $OUT_ROOT"
echo "[INFO][GPU$GPU_ID] Views:    $NUM_VIEWS"
echo "[INFO][GPU$GPU_ID] Res:      $RESOLUTION"
echo "[INFO][GPU$GPU_ID] Normalize max dim: $NORMALIZE_MAX_DIM"
echo "[INFO][GPU$GPU_ID] Auto-orient: $AUTO_ORIENT"
echo "[INFO][GPU$GPU_ID] Cycles device: $CYCLES_DEVICE"
echo "[INFO][GPU$GPU_ID] Auto-yaw: $AUTO_YAW (yaw_flip=${YAW_FLIP_DEFAULT:-<auto>})"
echo "[INFO][GPU$GPU_ID] Yaw offset deg: $YAW_OFFSET_DEG"
echo "[INFO][GPU$GPU_ID] Extra pitch deg: $EXTRA_PITCH_DEG"
echo "[INFO][GPU$GPU_ID] Extra roll deg: ${EXTRA_ROLL_DEG_DEFAULT:-<auto>}"

while IFS=$'\t' read -r MODEL CLS; do
  [[ -z "${MODEL:-}" || -z "${CLS:-}" ]] && continue
  COUNT=$((COUNT + 1))

  OBJ_PATH="$INPUT_ROOT/$CLS/$MODEL/meshes/model.obj"
  SAFE_MODEL="${MODEL// /_}"
  SAFE_CLS="${CLS// /_}"
  OBJ_UID="${SAFE_CLS}_${SAFE_MODEL}"
  OBJ_OUT="$OUT_ROOT/$OBJ_UID"
  IMG_DIR="$OBJ_OUT/render/images"
  LOG_FILE="$LOG_ROOT/$OBJ_UID.gpu${GPU_ID}.log"

  if [[ ! -f "$OBJ_PATH" ]]; then
    echo "[$COUNT/$TOTAL][GPU$GPU_ID][MISS] $OBJ_UID :: $OBJ_PATH"
    MISS=$((MISS + 1))
    continue
  fi

  IMG_COUNT=0
  if [[ -d "$IMG_DIR" ]]; then
    IMG_COUNT=$(find "$IMG_DIR" -maxdepth 1 -type f -name 'r_*.png' | wc -l)
  fi

  if [[ "$IMG_COUNT" -eq "$NUM_VIEWS" && -f "$OBJ_OUT/render/transforms.json" ]]; then
    echo "[$COUNT/$TOTAL][GPU$GPU_ID][SKIP] $OBJ_UID"
    SKIP=$((SKIP + 1))
    continue
  fi

  if [[ "$CLEAN_PARTIAL" == "1" ]]; then
    rm -rf "$OBJ_OUT"
  fi

  echo "[$COUNT/$TOTAL][GPU$GPU_ID][RUN ] $OBJ_UID"

  # Per-category defaults (can be overridden by env vars):
  # - shoe: yaw_flip + a small image-plane roll makes view0 match our preferred "default" orientation.
  local_yaw_flip="${YAW_FLIP_DEFAULT:-}"
  local_extra_roll_deg="${EXTRA_ROLL_DEG_DEFAULT:-}"
  if [[ -z "$local_yaw_flip" ]]; then
    if [[ "$SAFE_CLS" == "shoe" ]]; then
      local_yaw_flip="1"
    else
      local_yaw_flip="0"
    fi
  fi
  if [[ -z "$local_extra_roll_deg" ]]; then
    if [[ "$SAFE_CLS" == "shoe" ]]; then
      local_extra_roll_deg="20"
    else
      local_extra_roll_deg="0"
    fi
  fi

  # Category-aware template:
  # - If TEMPLATE_JSON is set, it overrides all categories.
  # - Else, use OmniObject3D category template: ${CLS}_001/render/transforms.json
  # - Fallback to box_001.
  TEMPLATE="$TEMPLATE_FIXED"
  if [[ -z "$TEMPLATE" ]]; then
    TEMPLATE="$TEMPLATE_ROOT/${SAFE_CLS}_001/render/transforms.json"
    if [[ ! -f "$TEMPLATE" ]]; then
      TEMPLATE="$TEMPLATE_ROOT/box_001/render/transforms.json"
    fi
  fi
  if [[ ! -f "$TEMPLATE" ]]; then
    echo "[$COUNT/$TOTAL][GPU$GPU_ID][FAIL] $OBJ_UID (missing template: $TEMPLATE)"
    FAIL=$((FAIL + 1))
    continue
  fi

  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" xvfb-run -a "$BLENDER" -b -P tools/render_omni_format.py -- \
    --object_path "$OBJ_PATH" \
    --output_dir "$OUT_ROOT" \
    --object_uid "$OBJ_UID" \
    --num_views "$NUM_VIEWS" \
    --resolution "$RESOLUTION" \
    --engine CYCLES \
    --cycles_device "$CYCLES_DEVICE" \
    --auto_orient "$AUTO_ORIENT" \
    --auto_yaw "$AUTO_YAW" \
    --normalize_max_dim "$NORMALIZE_MAX_DIM" \
    --yaw_offset_deg "$YAW_OFFSET_DEG" \
    --extra_pitch_deg "$EXTRA_PITCH_DEG" \
    --extra_roll_deg "$local_extra_roll_deg" \
    $( [[ "$local_yaw_flip" == "1" ]] && echo --yaw_flip ) \
    --pose_template_json "$TEMPLATE" \
    > "$LOG_FILE" 2>&1
  RC=$?
  set -e

  # Blender may return non-zero even when renders are saved; trust on-disk completeness.
  IMG_COUNT=0
  if [[ -d "$IMG_DIR" ]]; then
    IMG_COUNT=$(find "$IMG_DIR" -maxdepth 1 -type f -name 'r_*.png' | wc -l)
  fi
  if [[ "$IMG_COUNT" -eq "$NUM_VIEWS" && -f "$OBJ_OUT/render/transforms.json" ]]; then
    echo "[$COUNT/$TOTAL][GPU$GPU_ID][DONE] $OBJ_UID (rc=$RC)"
    DONE=$((DONE + 1))
  else
    echo "[$COUNT/$TOTAL][GPU$GPU_ID][FAIL] $OBJ_UID (rc=$RC, see $LOG_FILE)"
    FAIL=$((FAIL + 1))
  fi
done < "$MANIFEST"

echo "[INFO][GPU$GPU_ID] Finished. done=$DONE skip=$SKIP fail=$FAIL miss=$MISS total_rows=$TOTAL"
