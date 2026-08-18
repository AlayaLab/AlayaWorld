#!/usr/bin/env bash
# ============================================================================
# stage0: cache prebuild (no training). Recommended before stage1.
#   1) whole-clip VAE latent cache: training encodes the first 17 latents of a window fresh
#      and slices the tail from here. The training path only READS this cache, so without
#      stage0 every step pays a full fresh encode.
#   2) text-embedding cache: training fills this lazily; stage0 just front-loads it.
#
# Usage:
#   bash scripts/finetune/stage0_precache.sh                # everything
#   bash scripts/finetune/stage0_precache.sh --limit 64     # first 64 videos only (smoke test)
#   SKIP_VAE=1 / SKIP_TEXT=1 skip either step
# ============================================================================
set -euo pipefail
CONFIG_PATH=${CONFIG_PATH:-configs/stage0_precache.yaml}
LIMIT_ARGS=()
if [ "${1:-}" = "--limit" ] && [ -n "${2:-}" ]; then LIMIT_ARGS=(--limit "$2"); fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -eq 0 ] && NGPU=1
fi
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export TOKENIZERS_PARALLELISM=false
export TORCH_DISABLE_ADDR2LINE=${TORCH_DISABLE_ADDR2LINE:-1}

echo "=============================================="
echo "stage0 cache prebuild   config=$CONFIG_PATH  gpus=$NGPU  ${LIMIT_ARGS[*]:-(full)}"
echo "=============================================="

# ---- 1) VAE latent cache, sharded across GPUs ----
if [ "${SKIP_VAE:-0}" != "1" ]; then
  echo "[stage0] VAE latent prebuild: $NGPU shards in parallel"
  pids=()
  for r in $(seq 0 $((NGPU - 1))); do
    python scripts/tools/precache_vae_latents.py --config "$CONFIG_PATH" \
        --rank "$r" --world "$NGPU" --device "cuda:$r" "${LIMIT_ARGS[@]}" &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ] && echo "[stage0] VAE latent prebuild done" || { echo "[stage0] VAE latent prebuild: a shard failed"; exit 1; }
else
  echo "[stage0] skipping VAE latent prebuild (SKIP_VAE=1)"
fi

# ---- 2) text embeddings: reuse the trainer setup prebuild (max_steps=0 exits right after) ----
if [ "${SKIP_TEXT:-0}" != "1" ]; then
  echo "[stage0] text-embedding prebuild (max_steps=0, exits after setup)"
  CONFIG_PATH="$CONFIG_PATH" LOG_FILTER=all bash scripts/finetune/train.sh
  echo "[stage0] text-embedding prebuild done"
else
  echo "[stage0] skipping text-embedding prebuild (SKIP_TEXT=1)"
fi

echo "[stage0] all done. Next, run stage1:"
echo "  CONFIG_PATH=configs/stage1_pretrain_bidir.yaml bash scripts/finetune/train.sh"
