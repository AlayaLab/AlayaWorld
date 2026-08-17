"""Per-rollout cache for the FlashAlaya inference pipeline.

Mirrors FlashDreams' StreamInferencePipelineCache idea: everything computed once
per rollout (one-shot conditions) plus the mutable autoregressive state lives
here, so `generate(ar_index, cache)` is a pure step function over this object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RolloutCache:
    # ---- one-shot conditions (set by initialize_cache, read-only afterwards) ----
    latent_full: torch.Tensor
    """Encoded source video latents [B,C,T,H,W]; provides sink/history seed + GT."""
    video_pixels: torch.Tensor
    """Raw pixel video (used by the spatial bank / DA3 depth path)."""
    metadata: dict[str, Any]
    """Dataset metadata: cam_c2w / intrinsic / action vectors / caption info."""
    context: torch.Tensor
    """Text condition (Gemma encoding of the prompt caption)."""
    negative_context: torch.Tensor | None
    """Negative prompt encoding; only used when cfg_scale > 1."""

    sink_latent: torch.Tensor | None
    sink_indices: torch.Tensor | None
    sigmas: torch.Tensor
    """Denoise sigma schedule (len = steps + 1)."""

    # ---- layout constants ----
    K: int
    N: int
    cond_end: int
    gap_steps: int
    explicit_condition: int
    target_base_start: int

    # ---- mutable autoregressive state (updated by finalize) ----
    history: torch.Tensor | None
    """Sliding window of the last N latent frames (real seed, then own preds)."""
    history_action_t_indices: torch.Tensor | None
    explicit_nearby: torch.Tensor | None
    """Only used when N == 0 (no-memory mode)."""
    spatial_bank: Any | None
    """_RolloutSpatialBank: pixels + frame indices + DA3 depths of past frames."""

    preds: list[torch.Tensor] = field(default_factory=list)
    """Generated latent chunks, one [B,C,K,H,W] per finished AR step."""

    @property
    def H_lat(self) -> int:
        return int(self.latent_full.shape[3])

    @property
    def W_lat(self) -> int:
        return int(self.latent_full.shape[4])

    def target_start(self, ar_index: int) -> int:
        return self.target_base_start + ar_index * self.K
