"""Pure conditioning helpers for FlashAlaya (extracted from RolloutTrainer).

RoPE index grids, local t-offsets for each condition stream, the validation
sigma schedule, and action-control kwargs. All functions are stateless; device
and config values are passed explicitly so this module has no trainer
dependency.

Local RoPE layout convention (matches rollout_trainer.py):
    sink   -> t = 0
    memory -> t starts at 1
    nearby -> tail of history (1 + N - cond_end) when N > 0, else 1
    target -> starts right after history (1 + N) or explicit condition (1 + cond_end)
"""
from __future__ import annotations

import math
from typing import Any

import torch
from ltx2.modules.patchifier import VideoLatentPatchifier, VideoLatentShape
from ltx2.modules.scheduler import LTX2Scheduler, LinearQuadraticScheduler

from alaya.control.action import build_action_vectors


# --------------------------------------------------------------- t offsets
def sink_t_offset(history_latent_frames: int) -> int:
    return 0


def memory_t_offset(history_latent_frames: int, condition_latent_frames: int) -> int:
    return 1


def nearby_t_offset(history_latent_frames: int, condition_latent_frames: int, gap_steps: int = 0) -> int:
    history_latents = max(0, int(history_latent_frames))
    condition_latents = max(0, int(condition_latent_frames))
    if history_latents > 0:
        # With temporal memory enabled, nearby is copied from the tail of
        # history rather than inserted as an extra frame before target.
        return 1 + max(0, history_latents - condition_latents)
    return 1


def target_t_indices(
    frames: int,
    *,
    history_latent_frames: int,
    condition_latent_frames: int,
    gap_steps: int = 0,
    device: torch.device,
) -> torch.Tensor:
    history_latents = max(0, int(history_latent_frames))
    condition_latents = max(0, int(condition_latent_frames))
    start = 1 + history_latents if history_latents > 0 else 1 + condition_latents
    return torch.arange(start, start + int(frames), device=device, dtype=torch.float32)


# --------------------------------------------------------------- index grids
def indices_grid(
    batch: int, frames: int, height: int, width: int, *, t_offset: int, device: torch.device
) -> torch.Tensor:
    patchifier = VideoLatentPatchifier(patch_size=1)
    shape = VideoLatentShape(batch=batch, channels=1, frames=frames, height=height, width=width)
    coords = patchifier.get_patch_grid_bounds(shape, device=device).clone().to(torch.float32)
    coords[:, 0, :, :] += t_offset
    return coords


def indices_grid_for_t_indices(
    batch: int,
    t_indices: list[int | float] | torch.Tensor,
    height: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    frames = int(t_indices.numel()) if torch.is_tensor(t_indices) else len(t_indices)
    coords = indices_grid(batch, frames, height, width, t_offset=0, device=device)
    if torch.is_tensor(t_indices):
        t = t_indices.to(device=device, dtype=coords.dtype)
    else:
        t = torch.tensor(t_indices, device=device, dtype=coords.dtype)
    per_token = t.view(frames, 1, 1).expand(frames, height, width).reshape(-1)
    bounds = torch.stack([per_token, per_token + 1], dim=-1)
    coords[:, 0, :, :] = bounds.unsqueeze(0)
    return coords


# --------------------------------------------------------------- sigma schedule
def denoise_sigmas(
    *,
    steps: int,
    scheduler: str = "uniform",
    device: torch.device,
) -> torch.Tensor:
    """Sigma schedule for the few-step denoise (matches _validation_sigmas
    for the schedulers used at inference; the adaptive-shift training branch
    is intentionally not carried over)."""
    if scheduler == "uniform":
        sigmas = torch.linspace(1.0, 0.0, steps + 1)
    elif scheduler == "linear_quadratic":
        sigmas = LinearQuadraticScheduler().execute(steps=steps)
    else:
        sigmas = LTX2Scheduler().execute(
            steps=steps, latent=None, max_shift=2.05, base_shift=0.95, stretch=True, terminal=0.1
        )
    return sigmas.to(device=device, dtype=torch.float32)


# --------------------------------------------------------------- control kwargs
def _as_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.flatten()[0].item()) if value.numel() else False
    if isinstance(value, (list, tuple)):
        return _as_bool(value[0]) if value else False
    return bool(value)


def build_control_kwargs(
    *,
    metadata: dict[str, Any],
    control_modes: list[str],
    target_t_indices: torch.Tensor,
    condition_t_indices: torch.Tensor | None,
    history_t_indices: torch.Tensor | None,
    action_scale: str,
    temporal_stride: int,
    action_history_memory: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {}
    if "action" in control_modes:
        cam_c2w = metadata.get("cam_c2w")
        if cam_c2w is not None and _as_bool(metadata.get("has_camera", False)):
            kwargs["action_vectors"] = build_action_vectors(
                cam_c2w=cam_c2w,
                target_latent_indices=target_t_indices,
                action_scale=action_scale,
                temporal_stride=temporal_stride,
                device=device,
                dtype=dtype,
            )
            if condition_t_indices is not None and condition_t_indices.numel() > 0:
                kwargs["action_condition_vectors"] = build_action_vectors(
                    cam_c2w=cam_c2w,
                    target_latent_indices=condition_t_indices,
                    action_scale=action_scale,
                    temporal_stride=temporal_stride,
                    device=device,
                    dtype=dtype,
                )
            if action_history_memory and history_t_indices is not None and history_t_indices.numel() > 0:
                kwargs["action_history_vectors"] = build_action_vectors(
                    cam_c2w=cam_c2w,
                    target_latent_indices=history_t_indices,
                    action_scale=action_scale,
                    temporal_stride=temporal_stride,
                    device=device,
                    dtype=dtype,
                )
    return kwargs
