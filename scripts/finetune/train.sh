#!/usr/bin/env bash
# ============================================================================
# Generic training launcher: CONFIG_PATH selects what to train.
#   The trainer is dispatched from the config's *.enabled flags (see alaya/train.py):
#     frame_query -> FrameQueryTrainer, dmd -> DmdTrainer, otherwise RolloutTrainer
#
# Usage:
#   CONFIG_PATH=configs/histpretrain_ft20s.yaml bash scripts/finetune/train.sh
#   CONFIG_PATH=configs/pretrain_bidir_ft20s.yaml bash scripts/finetune/train.sh
# Optional environment variables:
#   VALIDATE_ONLY=1  run validation only, no training
#   DESCRIBE=1       print the resolved config summary and exit
#   LOG_FILTER=all   keep all stdout (default keeps only '[Train] step=' lines)
#   NPROC_PER_NODE / MASTER_PORT / LOG_ROOT / LOG_NAME can all be overridden
# ============================================================================
set -euo pipefail

: "${CONFIG_PATH:?CONFIG_PATH=configs/xxx.yaml is required (this script has no built-in default)}"
if [ ! -f "$CONFIG_PATH" ]; then
  echo "CONFIG_PATH does not exist: $CONFIG_PATH" >&2
  exit 1
fi

if [ -n "${KUBERNETES_CONTAINER_RESOURCE_GPU:-}" ]; then
  NPROC_PER_NODE=$KUBERNETES_CONTAINER_RESOURCE_GPU
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
  NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
  [ "$NPROC_PER_NODE" -eq 0 ] && NPROC_PER_NODE=8
fi

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

CONFIG_STEM=$(basename "$CONFIG_PATH")
CONFIG_STEM=${CONFIG_STEM%.*}
# Prefer run.log_dir from the config; otherwise derive ./logs/<config stem>
CONFIG_LOG_ROOT=$(python - "$CONFIG_PATH" "$CONFIG_STEM" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(((cfg.get("run") or {}).get("log_dir")) or f"./logs/{sys.argv[2]}")
PY
)
LOG_ROOT=${LOG_ROOT:-$CONFIG_LOG_ROOT}
LOG_NAME=${LOG_NAME:-$CONFIG_STEM}
LOG_DIR="${LOG_ROOT}/${LOG_NAME}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
LOG_FILTER=${LOG_FILTER:-train}
if [ "$LOG_FILTER" = "all" ]; then
  exec > >(tee -a "$LOG_FILE") 2>&1
else
  exec > >(grep -a --line-buffered '^\[Train\] step=' | tee -a "$LOG_FILE") 2>&1
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# ===== flash-attn-3 (used when a local build is available) =====
# Point FA3_HOPPER at a local flash-attn-3 (hopper) build; loss is bit-identical to FA2.
# ALAYA_USE_FA3=0 disables it; a missing build falls back to FA2 automatically.
FA3_HOPPER=${FA3_HOPPER:-}
if [ "${ALAYA_USE_FA3:-1}" = "1" ] && [ -f "$FA3_HOPPER/build/lib.linux-x86_64-3.10/flash_attn_3/_C.abi3.so" ]; then
  export ALAYA_USE_FA3=1
  export PYTHONPATH="$FA3_HOPPER/build/lib.linux-x86_64-3.10:$FA3_HOPPER:$PYTHONPATH"
  TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
  export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
  echo "[train.sh] FA3 enabled ($FA3_HOPPER)"
else
  export ALAYA_USE_FA3=0
  echo "[train.sh] FA3 disabled -> FA2"
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200000}
export ALAYA_DISTRIBUTED_TIMEOUT_SECONDS=${ALAYA_DISTRIBUTED_TIMEOUT_SECONDS:-36000}
# torch's `Module.cpp:185 symbolizing C++ stack trace` warning floods the log;
# set TORCH_DISABLE_ADDR2LINE=0 when you need the C++ stack.
export TORCH_DISABLE_ADDR2LINE=${TORCH_DISABLE_ADDR2LINE:-1}
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}

# Dataset environment knobs belong in shell/runtime, not in model config.
export LTX_SEKAI_GAME_JSONL=${LTX_SEKAI_GAME_JSONL:-sekai_game_walking_smooth.jsonl}
export LTX_SEKAI_GAME_POSE_SUBDIR=${LTX_SEKAI_GAME_POSE_SUBDIR:-pose_smooth}

echo "=============================================="
echo "Alaya training (generic launcher)"
echo "  config:      $CONFIG_PATH"
echo "  log_file:    $LOG_FILE"
echo "  gpus:        $NNODES x $NPROC_PER_NODE"
echo "  master:      $MASTER_ADDR:$MASTER_PORT"
echo "  start_time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

TRAIN_EXTRA_ARGS=()
if [ "${VALIDATE_ONLY:-0}" = "1" ]; then
  TRAIN_EXTRA_ARGS+=(--validate-only)
fi
if [ "${DESCRIBE:-0}" = "1" ]; then
  TRAIN_EXTRA_ARGS+=(--describe)
fi

python -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m alaya.train \
  --config "$CONFIG_PATH" \
  "${TRAIN_EXTRA_ARGS[@]}"
