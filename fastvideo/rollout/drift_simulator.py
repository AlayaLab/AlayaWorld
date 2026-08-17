"""
Helios-style "Easy Anti-Drifting" history latent corruption.

Reference: PKU-YuanGroup/Helios
  - helios/utils/utils_helios_base.py:265 corrupt_history_latents (noise/downsample)
  - helios/utils/utils_helios_base.py:469 add_saturation_to_history_latents
  - call order: corrupt_history first, add_saturation second

Pipeline:
  Stage 1: either noise or downsample (clean_prob skips the whole stage; noise:downsample = 9:1)
  Stage 2: saturation applied independently (saturation_clean_prob skips it)
  is_keep_x0=True: history[:,:,0] always stays the clean ground truth in both stages

Three corruption primitives, all in latent space (no VAE encode):
  - noise: history_lat = σ·randn + (1-σ)·history_lat,  σ ~ U(0, corrupt_ratio)
  - downsample: bilinear down then up (low-resolution artefacts)
  - saturation: (x - μ_c) * sat + μ_c, sat ~ U(0.7, 2.0)
"""
from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn.functional as F


_DRIFT_COUNTERS = {
    'total': 0,
    'noise': 0,
    'downsample': 0,
    'clean_s1': 0,
    'saturation': 0,
    'clean_s2': 0,
}


def get_drift_counters() -> dict:
    """Read the cumulative drift-mode counters (process lifetime)."""
    return dict(_DRIFT_COUNTERS)


def reset_drift_counters() -> None:
    """Reset the counters (debug helper)."""
    for k in _DRIFT_COUNTERS:
        _DRIFT_COUNTERS[k] = 0


def _downsample_corrupt(latent: torch.Tensor, ratio_min: float, ratio_max: float) -> torch.Tensor:
    """Spatial bilinear down-then-up corruption.

    Args:
        latent: [B, C, T, H, W] or [B, C, H, W]
    """
    ratio = random.uniform(ratio_min, ratio_max)
    is_5d = latent.dim() == 5
    if is_5d:
        B, C, T, H, W = latent.shape
        x = latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    else:
        B, C, H, W = latent.shape
        x = latent

    h0, w0 = x.shape[-2:]
    h1 = max(1, int(round(h0 * ratio)))
    w1 = max(1, int(round(w0 * ratio)))

    orig_dtype = x.dtype
    if orig_dtype not in (torch.float32,):
        x = x.float()
    x = F.interpolate(x, size=(h1, w1), mode='bilinear', align_corners=False, antialias=True)
    x = F.interpolate(x, size=(h0, w0), mode='bilinear', align_corners=False, antialias=True)
    x = x.to(orig_dtype)

    if is_5d:
        x = x.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
    return x


def _saturation_corrupt(latent: torch.Tensor,
                        sat_min: float = 0.3, sat_max: float = 1.7) -> torch.Tensor:
    """Latent saturation perturbation (Helios style).

    The channel mean acts as a grey baseline, then:
        x_saturated = (x - mean) * sat_factor + mean
    sat_factor < 1 desaturates (towards the mean), > 1 oversaturates (away from it).
    Desaturation and oversaturation are equally likely.
    """
    import random
    if random.random() < 0.5:
        sat_factor = random.uniform(sat_min, 1.0 - 1e-3)
    else:
        sat_factor = random.uniform(1.0 + 1e-3, sat_max)
    latent_mean = latent.mean(dim=1, keepdim=True)   # [B, 1, T, H, W]
    return (latent - latent_mean) * sat_factor + latent_mean


def _noise_corrupt(latent: torch.Tensor, corrupt_ratio: float,
                   is_frame_independent: bool = False) -> torch.Tensor:
    """Latent gaussian noise mix: x_noisy = σ·noise + (1-σ)·x.

    Args:
        latent: [B, C, T, H, W]
        corrupt_ratio: σ_max, σ ~ U(0, σ_max)
        is_frame_independent: True samples an independent sigma per frame
    """
    B = latent.shape[0]
    if is_frame_independent and latent.dim() == 5:
        T = latent.shape[2]
        sigma = torch.rand(B, 1, T, 1, 1, device=latent.device, dtype=latent.dtype) * corrupt_ratio
    else:
        sigma = torch.rand(B, device=latent.device, dtype=latent.dtype) * corrupt_ratio
        for _ in range(latent.dim() - 1):
            sigma = sigma.unsqueeze(-1)

    noise = torch.randn_like(latent)
    return sigma * noise + (1.0 - sigma) * latent


def corrupt_history_latents_helios(
    history_latent: torch.Tensor,
    noise_mode_prob: float = 0.9,
    corrupt_ratio: float = 1 / 3,
    clean_prob: float = 0.1,
    downsample_min: float = 0.9,
    downsample_max: float = 1.0,
    saturation_clean_prob: float = 0.1,
    saturation_min: float = 0.3,
    saturation_max: float = 1.7,
    is_frame_independent: bool = False,
    is_keep_x0: bool = True,
) -> torch.Tensor:
    """Corrupt history latents (the two Helios stages, applied independently).

    Pipeline (same order as the Helios training code):
      Stage 1 — corrupt_history_latents (noise OR downsample):
          skipped entirely with probability clean_prob, otherwise:
              random < noise_mode_prob (0.9) → noise corrupt
              else                            → downsample corrupt
      Stage 2 - add_saturation_to_history_latents (independent):
          skipped with probability saturation_clean_prob, otherwise saturation is applied

    is_keep_x0=True (the Helios default): both stages skip the first frame, so history[:, :, 0]

    Args:
        history_latent: [B, C, T, H, W]
        noise_mode_prob: probability of choosing noise over downsample in stage 1 (Helios uses 0.9)
        clean_prob: probability of skipping stage 1 entirely (Helios uses 0.1)
        saturation_clean_prob: probability of skipping stage 2 (Helios uses 0.1)
        others: corruption parameters

    Returns:
        the corrupted history latent, same shape
    """
    if is_keep_x0 and history_latent.shape[2] > 1:
        prefix = history_latent[:, :, :1]
        rest = history_latent[:, :, 1:]
    else:
        prefix = None
        rest = history_latent

    _DRIFT_COUNTERS['total'] += 1
    if random.random() >= clean_prob:
        if random.random() < noise_mode_prob:
            rest = _noise_corrupt(rest, corrupt_ratio, is_frame_independent)
            _DRIFT_COUNTERS['noise'] += 1
        else:
            rest = _downsample_corrupt(rest, downsample_min, downsample_max)
            _DRIFT_COUNTERS['downsample'] += 1
    else:
        _DRIFT_COUNTERS['clean_s1'] += 1

    if random.random() >= saturation_clean_prob:
        rest = _saturation_corrupt(rest, saturation_min, saturation_max)
        _DRIFT_COUNTERS['saturation'] += 1
    else:
        _DRIFT_COUNTERS['clean_s2'] += 1

    if prefix is not None:
        return torch.cat([prefix, rest], dim=2)
    return rest
