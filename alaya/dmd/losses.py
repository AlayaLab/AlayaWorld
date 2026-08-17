"""DMD2-style distillation losses for alaya-world (pure DMD, no GAN/reward/decouple).

Ported from the Helios post-training utilities and adapted to the flow-matching convention
of this stack.

Convention (see ltx2/modules/scheduler.py):
  x_t = (1-σ)·x0 + σ·noise,  σ∈[0,1]
  velocity target v = noise - x0
  x0 = x_t - sigma * v          <- flow_pred_to_x0, using the raw sigma

These are pure functions: they hold no trainer or distributed state. Quantities that must be
synchronized across ranks (sigma, the gradient step) are sampled by the caller and passed in.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable

import torch
import torch.nn.functional as F


def flow_pred_to_x0(velocity: torch.Tensor, x_t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Convert a rectified-flow velocity prediction to x0: x0 = x_t - sigma * v (raw sigma)."""
    return x_t - sigma * velocity


def forward_velocity(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    sigma: torch.Tensor,
    cond: dict[str, Any],
) -> torch.Tensor:
    """Unified forward: transformer(x=[x_t.squeeze(0)], t=sigma*1000, **cond) -> velocity [B,C,K,H,W].

    sigma: tensor of shape [1]; cond: the conditioning contract (context=[...], seq_len, fps, ...).
    """
    t = (sigma.reshape(1) * 1000.0).to(device=x_t.device, dtype=x_t.dtype)
    return model(x=[x_t.squeeze(0)], t=t, **cond)


def run_generator(
    model: torch.nn.Module,
    noise: torch.Tensor,
    cond: dict[str, Any],
    sigma_list: list[float],
    *,
    grad_step: int | None,
) -> torch.Tensor:
    """Few-step sampling to obtain the fake x0.

    sigma_list: monotonically decreasing sigmas (the first should be 1.0 and the last reaches x0).
    grad_step:  index of the step that keeps the computation graph; the trainer synchronizes it
                across ranks. None keeps no graph at all (used by the critic step).

    Gradient policy: non-selected steps advance under no_grad by re-noising the x0 prediction with
    the initial noise; on the selected step one x0 prediction is made with gradients and the loop
    returns immediately without integrating further.
    """
    x_t = noise
    n = len(sigma_list)
    x0 = x_t
    for i in range(n):
        sigma_i = float(sigma_list[i])
        sigma_t = torch.full((1,), sigma_i, device=x_t.device, dtype=x_t.dtype)
        keep_grad = grad_step is not None and i == int(grad_step)
        ctx = contextlib.nullcontext() if keep_grad else torch.no_grad()
        with ctx:
            v = forward_velocity(model, x_t, sigma_t, cond)
            x0 = flow_pred_to_x0(v, x_t, sigma_i)
        if keep_grad:
            return x0  # selected step: stop integrating
        if i < n - 1:
            next_sigma = float(sigma_list[i + 1])
            x_t = (1.0 - next_sigma) * x0 + next_sigma * noise
    return x0


def compute_kl_grad(
    score_model: torch.nn.Module,
    critic_lora,
    *,
    noisy: torch.Tensor,
    sigma: torch.Tensor,
    x0_fake: torch.Tensor,
    cond: dict[str, Any],
    neg_cond: dict[str, Any] | None,
    real_guidance_scale: float,
    normalization: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Core DMD gradient: grad = pred_fake - pred_real, then normalized per sample.

    This function does not wrap itself in no_grad; the caller must do so, because the score
    networks build no backward graph during the generator step.

    fake score = base + critic LoRA; real score = base only, with classifier-free guidance
    on the real side using the negative conditioning.
    Returns the normalized gradient as a detached constant.
    """
    sigma_f = float(sigma.reshape(-1)[0].item())

    critic_lora.enable()
    v_fake = forward_velocity(score_model, noisy, sigma, cond)
    pred_fake = flow_pred_to_x0(v_fake, noisy, sigma_f)

    with critic_lora.toggled(False):
        v_real_cond = forward_velocity(score_model, noisy, sigma, cond)
        pred_real_cond = flow_pred_to_x0(v_real_cond, noisy, sigma_f)
        if real_guidance_scale != 1.0 and neg_cond is not None:
            v_real_uncond = forward_velocity(score_model, noisy, sigma, neg_cond)
            pred_real_uncond = flow_pred_to_x0(v_real_uncond, noisy, sigma_f)
            pred_real = pred_real_cond + (pred_real_cond - pred_real_uncond) * real_guidance_scale
        else:
            pred_real = pred_real_cond
    critic_lora.enable()  # switch back to the fake score

    grad = pred_fake - pred_real
    log = {}
    if normalization:
        norm_dims = list(range(1, x0_fake.dim()))  # every dimension except batch (per-sample)
        normalizer = (x0_fake - pred_real).abs().mean(dim=norm_dims, keepdim=True)
        grad = grad / normalizer
        log["dmd_normalizer"] = float(normalizer.mean().item())
    grad = torch.nan_to_num(grad)
    log["dmd_grad_abs"] = float(grad.abs().mean().item())
    return grad, log


def compute_distribution_matching_loss(
    score_model: torch.nn.Module,
    critic_lora,
    *,
    x0_fake: torch.Tensor,
    sigma: torch.Tensor,
    cond: dict[str, Any],
    neg_cond: dict[str, Any] | None,
    real_guidance_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Wrap the DMD gradient into an MSE loss that back-propagates into the generator (target is stop-gradient).

    x0_fake: sample produced by run_generator with gradients (holds the generator graph).
    sigma:   tensor of shape [1], the noise level applied to x0_fake (sampled by the trainer across ranks).
    """
    sigma_f = float(sigma.reshape(-1)[0].item())
    with torch.no_grad():  # the score forward and the noising build no backward graph
        noise = torch.randn_like(x0_fake)
        noisy = (1.0 - sigma_f) * x0_fake + sigma_f * noise
        grad, log = compute_kl_grad(
            score_model,
            critic_lora,
            noisy=noisy,
            sigma=sigma,
            x0_fake=x0_fake,
            cond=cond,
            neg_cond=neg_cond,
            real_guidance_scale=real_guidance_scale,
        )

    target = (x0_fake.double() - grad.double()).detach()
    dmd_loss = 0.5 * F.mse_loss(x0_fake.double(), target)
    log["dmd_sigma"] = sigma_f
    return dmd_loss, log


def compute_critic_loss(
    score_model: torch.nn.Module,
    critic_lora,
    *,
    x0_fake: torch.Tensor,
    sigma: torch.Tensor,
    cond: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Critic / fake-score training: rectified-flow denoising MSE on generator samples.

    x0_fake: generated by the trainer under no_grad, so it carries no gradient. Only the critic LoRA is trained.
    """
    x0_fake = x0_fake.detach()
    sigma_f = float(sigma.reshape(-1)[0].item())
    noise = torch.randn_like(x0_fake)
    noisy = (1.0 - sigma_f) * x0_fake + sigma_f * noise

    critic_lora.enable()  # fake state (base + LoRA)
    v_pred = forward_velocity(score_model, noisy, sigma, cond)
    target = (noise - x0_fake).to(dtype=v_pred.dtype)
    denoising_loss = torch.mean((v_pred.float() - target.float()) ** 2)
    return denoising_loss, {"critic_sigma": sigma_f, "denoising_loss": float(denoising_loss.item())}


# =============================================================================
# =============================================================================


def cal_gan_loss(logits, label: int = 1) -> torch.Tensor:
    """GAN loss: mean(softplus(logit * label)). logits may be a single tensor or a list (multiple hook heads).

    Convention: label=+1 pushes the logit down, label=-1 pushes it up.
      - discriminator: real uses label=+1 (push down) and fake uses label=-1 (push up), giving real < 0 < fake;
      - generator: fake uses label=+1, i.e. it wants its own samples to be judged real.
    """
    if logits is None:
        return torch.tensor(0.0)
    if isinstance(logits, (list, tuple)):
        if not logits:
            return torch.tensor(0.0)
        total = 0.0
        for item in logits:
            total = total + torch.mean(F.softplus(item.float() * label))
        return total / len(logits)
    return torch.mean(F.softplus(logits.float() * label))


def _gan_noisy(x0: torch.Tensor, sigma_f: float, noise: torch.Tensor | None = None) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(x0)
    return (1.0 - sigma_f) * x0 + sigma_f * noise


def _gan_logits(
    score_model: torch.nn.Module,
    critic_lora,
    gan_disc,
    *,
    noisy: torch.Tensor,
    sigma: torch.Tensor,
    cond: dict[str, Any],
    K: int,
    H: int,
    W: int,
) -> list[torch.Tensor]:
    """Run one critic forward (fake state) and turn the hooked features into discriminator logits."""
    critic_lora.enable()  # the discriminator reads fake-score features
    with gan_disc.capturing():
        forward_velocity(score_model, noisy, sigma, cond)  # only the hooked features are needed; velocity is discarded
    return gan_disc.compute_logits(K, H, W)


def compute_gan_generator_loss(
    score_model: torch.nn.Module,
    critic_lora,
    gan_disc,
    *,
    x0_fake: torch.Tensor,
    sigma: torch.Tensor,
    cond: dict[str, Any],
    K: int,
    H: int,
    W: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GAN term of the generator step: noise the gradient-carrying x0_fake, run the discriminator and

    ask to be judged real. Gradients flow back through the discriminator head and the critic features into
    the generator; the discriminator parameters are not updated here.
    """
    sigma_f = float(sigma.reshape(-1)[0].item())
    noisy = _gan_noisy(x0_fake, sigma_f)
    logits = _gan_logits(score_model, critic_lora, gan_disc, noisy=noisy, sigma=sigma, cond=cond, K=K, H=H, W=W)
    gan_g_loss = cal_gan_loss(logits, label=1)
    return gan_g_loss, {"gan_G_loss": float(gan_g_loss.item())}


def compute_gan_critic_loss(
    score_model: torch.nn.Module,
    critic_lora,
    gan_disc,
    *,
    x0_fake: torch.Tensor,
    x0_real: torch.Tensor,
    sigma: torch.Tensor,
    cond: dict[str, Any],
    K: int,
    H: int,
    W: int,
    gan_d_weight: float = 1.0,
    r1_weight: float = 0.0,
    r2_weight: float = 0.0,
    r1_sigma: float = 0.1,
    r2_sigma: float = 0.1,
    split_backward: bool = False,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """GAN term of the critic step: discriminate real (ground-truth target latents) from fake (generator samples).

    Both x0_fake and x0_real are detached, so nothing flows back into the generator; this updates the
    discriminator head and the critic LoRA. The returned scalar already includes the loss weights.
    R1/R2 use the approximate form (MSE of the logits after perturbing the input) and are skipped when the weight is 0.
    """
    x0_fake = x0_fake.detach()
    x0_real = x0_real.detach()
    sigma_f = float(sigma.reshape(-1)[0].item())

    noise_f = torch.randn_like(x0_fake)
    noise_r = torch.randn_like(x0_real)
    noisy_fake = _gan_noisy(x0_fake, sigma_f, noise_f)
    noisy_real = _gan_noisy(x0_real, sigma_f, noise_r)

    if split_backward:
        log = {}
        fake_logits = _gan_logits(score_model, critic_lora, gan_disc, noisy=noisy_fake, sigma=sigma, cond=cond, K=K, H=H, W=W)
        gan_d_fake = cal_gan_loss(fake_logits, label=-1) * gan_d_weight
        anchor_fake = [l.detach() for l in fake_logits] if r2_weight > 0.0 else None
        gan_d_fake.backward()
        fv = float(gan_d_fake.item()); del fake_logits, gan_d_fake
        if r2_weight > 0.0:
            noisy_fake_p = noisy_fake + r2_sigma * torch.randn_like(noisy_fake)
            fake_logits_p = _gan_logits(score_model, critic_lora, gan_disc, noisy=noisy_fake_p, sigma=sigma, cond=cond, K=K, H=H, W=W)
            r2 = r2_weight * sum(F.mse_loss(a.float(), b.float()) for a, b in zip(anchor_fake, fake_logits_p)) / len(fake_logits_p)
            r2.backward()
            log["r2_loss"] = float(r2.item()); del fake_logits_p, r2, noisy_fake_p
        real_logits = _gan_logits(score_model, critic_lora, gan_disc, noisy=noisy_real, sigma=sigma, cond=cond, K=K, H=H, W=W)
        gan_d_real = cal_gan_loss(real_logits, label=1) * gan_d_weight
        anchor_real = [l.detach() for l in real_logits] if r1_weight > 0.0 else None
        gan_d_real.backward()
        rv = float(gan_d_real.item()); del real_logits, gan_d_real
        if r1_weight > 0.0:
            noisy_real_p = noisy_real + r1_sigma * torch.randn_like(noisy_real)
            real_logits_p = _gan_logits(score_model, critic_lora, gan_disc, noisy=noisy_real_p, sigma=sigma, cond=cond, K=K, H=H, W=W)
            r1 = r1_weight * sum(F.mse_loss(a.float(), b.float()) for a, b in zip(anchor_real, real_logits_p)) / len(real_logits_p)
            r1.backward()
            log["r1_loss"] = float(r1.item()); del real_logits_p, r1, noisy_real_p
        log.update({"gan_D_loss": fv + rv, "gan_D_fake": fv, "gan_D_real": rv})
        return None, log

    fake_logits = _gan_logits(score_model, critic_lora, gan_disc, noisy=noisy_fake, sigma=sigma, cond=cond, K=K, H=H, W=W)
    real_logits = _gan_logits(score_model, critic_lora, gan_disc, noisy=noisy_real, sigma=sigma, cond=cond, K=K, H=H, W=W)

    gan_d_fake = cal_gan_loss(fake_logits, label=-1) * gan_d_weight
    gan_d_real = cal_gan_loss(real_logits, label=1) * gan_d_weight
    gan_d_loss = gan_d_fake + gan_d_real

    log = {
        "gan_D_loss": float(gan_d_loss.item()),
        "gan_D_fake": float(gan_d_fake.item()),
        "gan_D_real": float(gan_d_real.item()),
    }

    reg = None
    if r1_weight > 0.0:  # R1: perturb real inputs and penalize the logit change (approximate gradient penalty)
        noisy_real_p = noisy_real + r1_sigma * torch.randn_like(noisy_real)
        real_logits_p = _gan_logits(score_model, critic_lora, gan_disc, noisy=noisy_real_p, sigma=sigma, cond=cond, K=K, H=H, W=W)
        r1 = r1_weight * sum(F.mse_loss(a.float(), b.float()) for a, b in zip(real_logits, real_logits_p)) / len(real_logits)
        reg = r1 if reg is None else reg + r1
        log["r1_loss"] = float(r1.item())
    if r2_weight > 0.0:  # R2: perturb fake inputs
        noisy_fake_p = noisy_fake + r2_sigma * torch.randn_like(noisy_fake)
        fake_logits_p = _gan_logits(score_model, critic_lora, gan_disc, noisy=noisy_fake_p, sigma=sigma, cond=cond, K=K, H=H, W=W)
        r2 = r2_weight * sum(F.mse_loss(a.float(), b.float()) for a, b in zip(fake_logits, fake_logits_p)) / len(fake_logits)
        reg = r2 if reg is None else reg + r2
        log["r2_loss"] = float(r2.item())

    total = gan_d_loss if reg is None else gan_d_loss + reg
    return total, log
