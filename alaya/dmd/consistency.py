"""Causal consistency distillation: a few-step warm start before DMD.

Cold-starting DMD from a many-step model tends to produce waxy or over-sharpened output, and in
autoregressive rollout the per-step bias compounds (exposure bias). Consistency distillation first
warm-starts the generator LoRA into a few-step-consistent state, then DMD does distribution matching.

Flow-matching convention (same as alaya/dmd/losses.py):
  x_σ = (1-σ)·x0 + σ·noise,  v = noise - x0,  x0 = x_σ - σ·v,  dx/dσ = v
teacher = frozen base (generator LoRA disabled); student = base + generator LoRA (LoRA enabled).
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn.functional as F

from alaya.dmd.losses import flow_pred_to_x0, forward_velocity


def _sigma_tensor(value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((1,), float(value), device=device, dtype=dtype)


def compute_consistency_distill_loss(
    transformer: torch.nn.Module,
    lora_manager,
    *,
    x0: torch.Tensor,
    cond: dict[str, Any],
    sigma_hi: float,
    sigma_lo: float,
    loss_type: str = "huber",
    huber_c: float = 1e-3,
    target_ctx=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Consistency distillation loss on an adjacent sigma pair; at sigma_lo -> 0 it anchors to the teacher.

    target_ctx: zero-argument context-manager factory supplied by the trainer. When given it wraps the
    target forward so the trainer can swap the generator LoRA to its EMA weights;
    None means self-target, i.e. the live weights act as the target.

    - x_hi = (1-σ_hi)·x0 + σ_hi·ε
    - teacher single-step ODE (LoRA off, no_grad): x_lo = x_hi + (sigma_lo - sigma_hi) * v_teacher
      (exact under rectified flow when v is the true velocity)
    - target = student x0 prediction at (x_lo, sigma_lo) (LoRA on, stop-grad)
    - student = student x0 prediction at (x_hi, sigma_hi) (with gradient)
    - loss = d(x0_student, sg[x0_target])
    At sigma_lo = 0, x_lo is the teacher x0 prediction and target = x_lo, anchoring the boundary to the teacher.

    Only the student branch keeps gradients; teacher and target run under no_grad.
    """
    device, dtype = x0.device, x0.dtype
    s_hi = float(sigma_hi)
    s_lo = float(sigma_lo)
    sig_hi = _sigma_tensor(s_hi, device, dtype)
    sig_lo = _sigma_tensor(s_lo, device, dtype)

    eps = torch.randn_like(x0)
    x_hi = (1.0 - s_hi) * x0 + s_hi * eps

    # teacher: frozen base (LoRA off), single ODE step to sigma_lo
    with torch.no_grad(), lora_manager.toggled(False):
        v_teacher = forward_velocity(transformer, x_hi, sig_hi, cond)
        x_lo = x_hi + (s_lo - s_hi) * v_teacher

    # target: x0 prediction of EMA(student) or self-target at the neighbouring sigma, stop-grad.
    # target_ctx swaps to EMA weights and keeps LoRA enabled; None means self-target on live weights.
    if target_ctx is None:
        lora_manager.enable()
        tgt_cm = contextlib.nullcontext()
    else:
        tgt_cm = target_ctx()
    with torch.no_grad(), tgt_cm:
        v_tgt = forward_velocity(transformer, x_lo, sig_lo, cond)
        x0_tgt = flow_pred_to_x0(v_tgt, x_lo, s_lo)

    # student: with gradient (live LoRA; the EMA context has exited and restored live weights)
    lora_manager.enable()
    v_s = forward_velocity(transformer, x_hi, sig_hi, cond)
    x0_s = flow_pred_to_x0(v_s, x_hi, s_hi)

    target = x0_tgt.float().detach()
    pred = x0_s.float()
    if loss_type == "huber":
        diff = pred - target
        loss = torch.sqrt(diff * diff + huber_c * huber_c).mean() - huber_c
    elif loss_type == "mse":
        loss = F.mse_loss(pred, target)
    else:
        raise ValueError(f"unknown consistency loss_type: {loss_type}")

    log = {
        "cd_sigma_hi": s_hi,
        "cd_sigma_lo": s_lo,
        "cd_loss": float(loss.item()),
    }
    return loss, log
