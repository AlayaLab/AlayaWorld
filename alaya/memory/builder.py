from __future__ import annotations

from pathlib import Path

import torch

from alaya.config.schema import MemoryConfig
from alaya.memory.history_encoder import VideoHistoryEncoder


def build_history_encoder(
    cfg: MemoryConfig,
    *,
    in_channels: int,
    out_channels: int,
    device: torch.device,
    dtype: torch.dtype,
    checkpoint_path: str | None = None,
    state_dict: dict | None = None,
) -> VideoHistoryEncoder:
    encoder = VideoHistoryEncoder(
        in_channels=in_channels,
        out_channels=out_channels,
        compress_t=cfg.compress_t,
        compress_h=cfg.compress_h,
        compress_w=cfg.compress_w,
        lr_compress_t=cfg.lr_compress_t,
        lr_compress_h=cfg.lr_compress_h,
        lr_compress_w=cfg.lr_compress_w,
        gate_init=cfg.gate_init,
        use_self_attn=cfg.use_self_attn,
        use_lr_branch=cfg.use_lr_branch,
        use_camera_pose=False,
    ).to(device=device, dtype=dtype)

    if state_dict is not None:
        # da3 inference: weights folded into the merged one-file checkpoint
        # (history_encoder.* subset), passed in directly.
        state = _adapt_history_encoder_state(dict(state_dict), encoder)
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        if len(unexpected) != 0:
            print(f"[HistoryEncoder] loaded <merged checkpoint>; missing={len(missing)} unexpected={len(unexpected)}")
    elif checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _adapt_history_encoder_state(state, encoder)
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f"[HistoryEncoder] loaded {checkpoint_path}; missing={len(missing)} unexpected={len(unexpected)}")

    for param in encoder.parameters():
        param.requires_grad_(cfg.train)
    return encoder


def _adapt_history_encoder_state(state: dict[str, torch.Tensor], encoder: VideoHistoryEncoder) -> dict[str, torch.Tensor]:
    """Load old camera-pose history checkpoints into the no-camera clean encoder.

    Old Path-C checkpoints used `in_channels + pose_emb_dim` on hr_stage1. The
    first `in_channels` slice is still the latent branch, so it is safe to keep
    that slice when the clean config disables pose input.
    """
    model_state = encoder.state_dict()
    adapted = {}
    for key, value in state.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if value.shape == target.shape:
            adapted[key] = value
            continue
        if key == "hr_stage1.conv.weight" and value.ndim == target.ndim and value.shape[0] == target.shape[0]:
            if value.shape[1] >= target.shape[1] and value.shape[2:] == target.shape[2:]:
                adapted[key] = value[:, : target.shape[1]].contiguous()
                print(
                    f"[HistoryEncoder] adapted {key}: checkpoint {tuple(value.shape)} -> model {tuple(target.shape)}",
                    flush=True,
                )
                continue
        print(
            f"[HistoryEncoder] skip mismatched {key}: checkpoint {tuple(value.shape)} vs model {tuple(target.shape)}",
            flush=True,
        )
    return adapted
