#!/usr/bin/env bash
# ============================================================================
# Single image + camera intrinsics/extrinsics + prompt -> video
#
#   bash scripts/infer/generate_video.sh --image path/to/first_frame.png \
#        --prompt "a first-person walk down a misty forest trail" \
#        [--synth-frames 256 --forward 0.0049 --yaw 0.15] \
#        [--extrinsics my_c2w.npz] [--intrinsic 0.5 0.89 0.5 0.5] \
#        [--rounds 5]
#
# Notes:
#   * extrinsics: --extrinsics takes your own c2w [N,4,4]; otherwise --synth-* builds one
#   * intrinsics: --intrinsic takes normalized fx fy cx cy; omitted -> placeholder + ViGeo fit
#   * length: --rounds N, 4 latents (~1.3s) per round
#   * weights: set paths.* in configs/infer_i2v_camera.yaml (dmd_resume = the 4-step student)
# ============================================================================
set -euo pipefail
CONFIG_PATH=${CONFIG_PATH:-configs/infer_i2v_camera.yaml}
ROUNDS=""
PREP_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --rounds) ROUNDS="$2"; PREP_ARGS+=(--rounds "$2"); shift 2 ;;
    *) PREP_ARGS+=("$1"); shift ;;
  esac
done

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
OUTDIR=$(python - "$CONFIG_PATH" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(c["validation"]["modes"]["custom_i2v"]["dataset"]["image_dir"].rsplit("/", 1)[0])
PY
)
echo "[generate] preparing inputs in $OUTDIR"
python scripts/infer/prepare_i2v_inputs.py --out "$OUTDIR" "${PREP_ARGS[@]}"

if [ -n "$ROUNDS" ]; then
  python - "$CONFIG_PATH" "$ROUNDS" <<'PY'
import sys, yaml
p, r = sys.argv[1], int(sys.argv[2])
c = yaml.safe_load(open(p, encoding="utf-8"))
c["validation"]["modes"]["custom_i2v"]["rollout_rounds"] = r
yaml.safe_dump(c, open(p, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
print(f"[generate] rollout_rounds = {r}")
PY
fi

echo "[generate] generating (VALIDATE_ONLY=1; the ViGeo spatial path is incompatible with FA3, so ALAYA_USE_FA3=0)"
VALIDATE_ONLY=1 ALAYA_USE_FA3=0 LOG_FILTER=all CONFIG_PATH="$CONFIG_PATH" bash scripts/finetune/train.sh

OUT_ROOT=$(python - "$CONFIG_PATH" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["run"]["output_dir"])
PY
)
echo "[generate] done. Videos:"
find "$OUT_ROOT/validation" -name "*.mp4" -newermt "-30 minutes" 2>/dev/null | sed 's/^/  /' ||   find "$OUT_ROOT/validation" -name "*.mp4" | tail -5 | sed 's/^/  /'
