from __future__ import annotations

import random
from typing import Any

import torch
import torch.distributed as dist

from alaya.dmd import (
    compute_critic_loss,
    compute_distribution_matching_loss,
    compute_gan_critic_loss,
    compute_gan_generator_loss,
    run_generator,
)
from alaya.trainer.dmd_self_rollout import DmdSelfRolloutMixin
from alaya.trainer.rollout_trainer import RolloutTrainer
from alaya.utils.distributed import broadcast_tensor, rank0_print


class DmdTrainer(DmdSelfRolloutMixin, RolloutTrainer):
    """DMD2-style few-step distillation trainer.

    This intentionally reuses RolloutTrainer's data, conditioning, spatial warp,
    history memory, checkpoint, and validation paths. Only the training step is
    replaced by generator/critic DMD updates.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.critic_optimizer = None

    def setup(self) -> None:
        if not self.cfg.dmd.enabled:
            raise ValueError("DmdTrainer requires dmd.enabled=true")
        super().setup()
        self._setup_dmd()

    def _setup_dmd(self) -> None:
        if self.components is None:
            raise RuntimeError("components are not initialized")
        if self.components.score_model is None or self.components.critic_lora is None:
            raise RuntimeError("dmd.enabled but score_model/critic_lora were not built")
        critic_params = self.components.critic_lora.get_trainable_parameters()
        if not critic_params:
            raise RuntimeError("dmd critic LoRA has no trainable parameters")

        gan_params = []
        if self.cfg.dmd.is_use_gan:
            if self.components.gan_discriminator is None:
                raise RuntimeError("dmd.is_use_gan=true but gan_discriminator was not built")
            gan_params = self.components.gan_discriminator.trainable_parameters()
            if not gan_params:
                raise RuntimeError("dmd.is_use_gan=true but gan_discriminator has no trainable parameters")

        self.critic_optimizer = torch.optim.AdamW(
            critic_params + gan_params,
            lr=float(self.cfg.dmd.critic_lr),
            weight_decay=float(self.cfg.optimizer.weight_decay),
        )
        rank0_print(
            self.dist,
            "[DMD]",
            f"critic optimizer built: critic_params={len(critic_params)} gan_params={len(gan_params)} "
            f"lr={self.cfg.dmd.critic_lr} sigma_list={self.cfg.dmd.dmd_sigma_list} "
            f"ratio={self.cfg.dmd.dfake_gen_update_ratio} real_cfg={self.cfg.dmd.real_guidance_scale} "
            f"gan={self.cfg.dmd.is_use_gan} gan_start_step={self.cfg.dmd.gan_start_step}",
        )
        # Consistency regularization (CM + DMD jointly): initialize the EMA shadow when enabled.
        cmr = self.cfg.dmd.cm_reg
        self._cmreg_ema = None
        if cmr.enabled:
            if self.components.lora_manager is None:
                raise RuntimeError("dmd.cm_reg.enabled requires a generator LoRA (training.mode=lora)")
            self._cmreg_init_ema()
            rank0_print(
                self.dist,
                "[DMD]",
                f"cm_reg on: weight={cmr.weight} every={cmr.every} num_scales={cmr.num_scales} "
                f"ema={cmr.use_ema}(w={cmr.ema_weight},start={cmr.ema_start_step}) "
                f"dmd_warmup={cmr.dmd_warmup_steps}",
            )

    def train_one_step(self, batch: Any) -> tuple[float, float, dict[str, Any]]:
        return self.train_one_step_dmd(batch)

    def _build_conditioning(self, batch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._uses_vigeo_prefix_last_frame():
            # DMD cond assembly mirrors the non-vigeo rollout layout (single-stream split),
            # the vigeo prefix packing is not wired here. The standard stage3 setup skips spatial during
            # training (it enters no loss), so the classic layout is used; vigeo applies to the validation bank only.
            _sr = self.cfg.dmd.self_rollout
            _spatial_skipped = bool(
                _sr.enabled
                and not _sr.reuse_spatial_bank
                and (bool(getattr(_sr, "score_context_free", False)) or bool(getattr(_sr, "score_gt_context", False)))
            )
            # With reuse_spatial_bank=true the rollout conditions take vigeo spatial latents from the bank
            # (wired inside dmd_self_rollout; GT and generated latents are mixed per spatial_bank_gt_prob).
            _vigeo_bank_wired = bool(_sr.enabled and _sr.reuse_spatial_bank)
            if not (_spatial_skipped or _vigeo_bank_wired):
                raise ValueError(
                    "vigeo mode in DMD training requires spatial to be skipped "
                    "(self_rollout + gt/cf-score + reuse_spatial_bank=false) "
                    "or the bank path (reuse_spatial_bank=true)"
                )
        video_pixels, caption, metadata = self._unpack_batch(batch)
        prompt_caption = self._caption_with_prefix(caption, metadata)
        latent_full = self._encode_video(video_pixels, metadata=metadata)
        context = self._encode_caption(prompt_caption, sync=False)

        K, gap_steps, cond_mode, cond_end = self._sample_layout(metadata)
        control_modes = self._sample_control_modes()
        # vigeo_seed aligns the round-0 seed condition with the teacher: sink/history/nearby are built
        # from GT pixels as three separate small encodes (sink = one random prefix frame, history = 25-frame
        # prefix, nearby = 9-frame motion latent) and the target starts at pixel 25, matching the
        # teacher round-0 condition distribution. Deeper rounds still continue in latent space.
        # latent_full remains a whole-clip encode. Default off = no behaviour change.
        _sr = self.cfg.dmd.self_rollout if self.cfg.dmd.enabled else None
        vigeo_seed = bool(
            _sr is not None and _sr.enabled and bool(getattr(_sr, "vigeo_seed", False))
        )
        if vigeo_seed:
            if cond_mode != "i2v" or cond_end != 1:
                raise ValueError("dmd.self_rollout.vigeo_seed requires an i2v layout with cond_end=1")
            _N = int(self.cfg.layout.history_latent_frames)
            if _N <= 0:
                raise ValueError("dmd.self_rollout.vigeo_seed requires history_latent_frames > 0")
            _prefix_px = self._vigeo_target_prefix_pixel_frames()
            _sink_idx = random.randrange(_prefix_px)
            sink_latent = self._encode_video(
                self._slice_video_pixel_frames(video_pixels, _sink_idx, _sink_idx + 1),
                needed_latents=1,
            )
            history_latent = self._encode_video(
                self._slice_video_pixel_frames(video_pixels, 0, _prefix_px),
                needed_latents=_N,
            )
            if int(history_latent.shape[2]) != _N:
                raise RuntimeError(
                    f"vigeo_seed history VAE length mismatch: got {history_latent.shape[2]}, expected {_N}"
                )
            _motion_px = self._vigeo_motion_pixel_frames()
            _anchor, nearby_latent = self._encode_vigeo_motion_window(
                self._slice_video_pixel_frames(video_pixels, _prefix_px - _motion_px, _prefix_px)
            )
            target_start = _N
            target_clean = latent_full[:, :, _N:_N + K].contiguous()
            if int(target_clean.shape[2]) != K:
                raise RuntimeError("vigeo_seed: latent_full too short for the first target chunk")
        else:
            sink_latent, history_latent, nearby_latent, target_clean, target_start = self._split_segments(
                latent_full,
                K=K,
                gap_steps=gap_steps,
                cond_end=cond_end,
            )

        # round0_no_memory matches the inference protocol memory_start_round=1: round 0 injects no memory
        # tokens (the HistoryEncoder is not called and the sequence is genuinely shorter); the history latent
        # buffer is still carried into round 1. sink/nearby are unaffected. Default off = no behaviour change.
        round0_no_memory = bool(
            _sr is not None and _sr.enabled and bool(getattr(_sr, "round0_no_memory", False))
        )
        mem_tokens = None
        mem_indices = None
        if history_latent is not None:
            if not round0_no_memory:
                if self.history_encoder is None:
                    raise RuntimeError("history_latent present but history_encoder is None")
                mem_tokens, mem_indices = self.history_encoder(history_latent)
                mem_indices = mem_indices.clone()
                mem_indices[:, 0, :, :] += self._local_memory_t_offset(
                    self.cfg.layout.history_latent_frames,
                    cond_end,
                )
            if cond_end > 0 and not vigeo_seed:
                nearby_latent = history_latent[:, :, -cond_end:].contiguous()

        B, C, _, H_lat, W_lat = target_clean.shape
        sink_indices = (
            self._indices_grid(
                B,
                self.cfg.layout.sink_latent_frames,
                H_lat,
                W_lat,
                t_offset=self._local_sink_t_offset(self.cfg.layout.history_latent_frames),
            )
            if sink_latent is not None
            else None
        )
        nearby_indices = (
            self._indices_grid(
                B,
                cond_end,
                H_lat,
                W_lat,
                t_offset=self._local_nearby_t_offset(
                    self.cfg.layout.history_latent_frames,
                    cond_end,
                    gap_steps=gap_steps,
                ),
            )
            if cond_end > 0
            else None
        )
        target_rope_t_indices = self._local_target_t_indices(
            K,
            history_latent_frames=self.cfg.layout.history_latent_frames,
            condition_latent_frames=cond_end,
            gap_steps=gap_steps,
        )
        # SF++ optimization (only when self_rollout is enabled): with cf-/gt-score and the spatial
        # bank off, the round-0 spatial build (DA3+warp+VAE encode) is never consumed by any loss
        # forward (scoring is text-only or GT-context; rollout/gen conds get spatial from the bank or
        # None) → skip it to save 1-3s/step. Guarded on _sr.enabled so the disabled path is
        # untouched; dropout=0 means the skipped branch draws no RNG either (no reordering).
        _sr = self.cfg.dmd.self_rollout if self.cfg.dmd.enabled else None
        _skip_spatial = bool(
            _sr is not None
            and _sr.enabled
            and (
                (
                    not _sr.reuse_spatial_bank
                    and (bool(getattr(_sr, "score_context_free", False)) or bool(getattr(_sr, "score_gt_context", False)))
                )
                # With reuse + vigeo the round-0 base condition skips the depth-warp builder
                # (vigeo spatial latents come from the self-rollout bank).
                or (_sr.reuse_spatial_bank and self._uses_vigeo_prefix_last_frame())
            )
        )
        spatial_context = (
            None
            if _skip_spatial
            else self._build_spatial_context(
                latent_full=latent_full,
                video_pixels=video_pixels,
                metadata=metadata,
                target_start=target_start,
                K=K,
                cond_end=cond_end,
                target_rope_t_indices=target_rope_t_indices,
            )
        )
        spatial_latent = spatial_context["latent"] if spatial_context is not None else None
        spatial_mask_patch = spatial_context.get("mask_patch") if spatial_context is not None else None
        spatial_indices = (
            self._indices_grid_for_t_indices(
                B,
                spatial_context.get("rope_t_indices", spatial_context["target_indices"]),
                H_lat,
                W_lat,
            )
            if spatial_context is not None
            else None
        )

        target_action_t_indices = torch.arange(
            target_start,
            target_start + K,
            device=self.dist.device,
            dtype=torch.float32,
        )
        condition_action_t_indices = (
            torch.arange(target_start - cond_end, target_start, device=self.dist.device, dtype=torch.float32)
            if cond_end > 0
            else None
        )
        history_action_t_indices = (
            torch.arange(
                target_start - self.cfg.layout.history_latent_frames,
                target_start,
                device=self.dist.device,
                dtype=torch.float32,
            )
            if (
                bool(getattr(self.cfg.control, "action_history_memory", False))
                and history_latent is not None
            )
            else None
        )
        control_kwargs = self._build_control_kwargs(
            metadata=metadata,
            control_modes=control_modes,
            target_t_indices=target_action_t_indices,
            condition_t_indices=condition_action_t_indices,
            history_t_indices=history_action_t_indices,
            dtype=target_clean.dtype,
        )

        cond = {
            "context": [context],
            "seq_len": K * H_lat * W_lat,
            "fps": self.cfg.sample.fps,
            "history_kv_tokens": mem_tokens,
            "history_indices_grid": mem_indices,
            "gen_t_indices_override": target_rope_t_indices,
            "sink_latent": sink_latent,
            "sink_indices_grid": sink_indices,
            "spatial_latent": spatial_latent,
            "spatial_mask_patch": spatial_mask_patch,
            "spatial_indices_grid": spatial_indices,
            "nearby_latent": nearby_latent,
            "nearby_indices_grid": nearby_indices,
            **control_kwargs,
        }

        target_next_clean = None
        next_indices_grid = None
        if self.cfg.next_forcing.enabled:
            next_start = target_start + K
            next_end = next_start + K
            if latent_full.shape[2] < next_end:
                raise ValueError(
                    f"next_forcing.enabled needs latent T>={next_end} "
                    f"(target_start={target_start}, K={K}), got {latent_full.shape[2]}"
                )
            target_next_clean = latent_full[:, :, next_start:next_end].contiguous()
            next_indices_grid = self._indices_grid_for_t_indices(B, target_rope_t_indices + K, H_lat, W_lat)

        return cond, {
            "B": B,
            "C": C,
            "K": K,
            "H_lat": H_lat,
            "W_lat": W_lat,
            "dtype": target_clean.dtype,
            "target_clean": target_clean,
            "target_next_clean": target_next_clean,
            "next_indices_grid": next_indices_grid,
            "gap_steps": gap_steps,
            "cond_mode": cond_mode,
            "cond_end": cond_end,
            "control_modes": control_modes,
            "spatial_context": 0 if spatial_latent is None else int(spatial_latent.shape[2]),
            "source": str(metadata.get("source", "")),
            "video_id": str(metadata.get("video_id", "unknown")),
            "frame_start": int(self._first_scalar(metadata.get("frame_start", -1))),
            "frame_end": int(self._first_scalar(metadata.get("frame_end", -1))),
            # SF++ self-rollout reuse (avoids a second _encode_video / _unpack_batch). Internal only,
            # additive — the disabled train step never reads these, so no behavior change. Do not
            # json.dumps(meta) (these hold tensors / raw batch metadata).
            "latent_full": latent_full,
            "metadata": metadata,
            "video_pixels": video_pixels,
            "target_start": target_start,
            # vigeo_seed: the three-part seed condition reused by self-rollout; None when the flag is off.
            "vigeo_seed_conds": (
                {"sink": sink_latent, "history": history_latent, "nearby": nearby_latent}
                if vigeo_seed
                else None
            ),
        }

    def _dmd_sample_inner_sigma(self, dtype: torch.dtype, *, latent_frames: int) -> torch.Tensor:
        # Affine-map the sampled sigma into [min_inner_sigma, max_inner_sigma] (was a hard clamp to
        # [0.02, 0.98]). Affine (rather than clamp) keeps the in-range distribution uniform when the
        # interval narrows — e.g. max_inner_sigma<0.98 to cut the high-sigma teacher over-saturation
        # push (SF++ color diagnostic). Same RNG draw as before (no new/extra draw). Defaults
        # [0.02, 0.98] reproduce the old *range* but with affine (not clamped) sampling.
        sigma = self._sample_sigma(dtype, latent_frames=int(latent_frames))
        lo, hi = float(self.cfg.dmd.min_inner_sigma), float(self.cfg.dmd.max_inner_sigma)
        return lo + (hi - lo) * sigma

    def _dmd_sample_mcp_sigma(self, dtype: torch.dtype, *, latent_frames: int) -> torch.Tensor:
        sigma = self._dmd_sample_inner_sigma(dtype, latent_frames=int(latent_frames))
        shift = float(self.cfg.next_forcing.sigma_shift)
        if shift != 1.0:
            sigma = (shift * sigma) / (1.0 + (shift - 1.0) * sigma)
        return sigma.clamp_(0.02, 0.98)

    def _dmd_sample_grad_step(self, num_steps: int) -> int:
        if bool(self.cfg.dmd.last_step_only):
            return int(num_steps) - 1
        if self.dist.is_main:
            step = torch.randint(low=0, high=int(num_steps), size=(1,), device=self.dist.device, dtype=torch.long)
        else:
            step = torch.zeros(1, device=self.dist.device, dtype=torch.long)
        broadcast_tensor(step)
        return int(step.item())

    @staticmethod
    def _first_scalar(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.reshape(-1)[0].item()
        if isinstance(value, (list, tuple)):
            return value[0] if value else -1
        return value

    def train_one_step_dmd(self, batch: Any) -> tuple[float, float, dict[str, Any]]:
        # Self-Forcing++ (the SF++ port notes (not shipped)): when self_rollout is enabled, the generator
        # sample is replaced by a no_grad autoregressive rollout + supervised window. When it is
        # disabled the rest of this method is the original teacher-forcing DMD step, unchanged —
        # the gate draws no RNG and does not perturb the disabled path (hard zero-regression req).
        if bool(self.cfg.dmd.self_rollout.enabled):
            return self._train_one_step_dmd_self_rollout(batch)

        if self.components is None:
            raise RuntimeError("components are not initialized")
        if self.optimizer is None or self.critic_optimizer is None:
            raise RuntimeError("DMD optimizers are not initialized")

        transformer = self.components.transformer
        score_model = self.components.score_model
        critic_lora = self.components.critic_lora
        if score_model is None or critic_lora is None:
            raise RuntimeError("DMD score_model/critic_lora are not initialized")

        transformer.train()
        score_model.train()
        if self.history_encoder is not None:
            self.history_encoder.train(self.cfg.memory.train)

        cond, meta = self._build_conditioning(batch)
        B = meta["B"]
        C = meta["C"]
        K = meta["K"]
        H_lat = meta["H_lat"]
        W_lat = meta["W_lat"]
        dtype = meta["dtype"]
        sigma_list = [float(s) for s in self.cfg.dmd.dmd_sigma_list]
        real_cfg = float(self.cfg.dmd.real_guidance_scale)

        neg_cond = None
        if real_cfg != 1.0:
            neg_caption = self.cfg.dmd.negative_prompt or self.cfg.validation.negative_prompt
            neg_context = self._encode_caption(neg_caption, sync=False)
            neg_cond = {**cond, "context": [neg_context]}

        def _noise() -> torch.Tensor:
            return torch.randn(B, C, K, H_lat, W_lat, device=self.dist.device, dtype=dtype)

        train_generator = self.global_step % int(self.cfg.dmd.dfake_gen_update_ratio) == 0
        gan_on = bool(self.cfg.dmd.is_use_gan) and self.global_step >= int(self.cfg.dmd.gan_start_step)
        gan_disc = self.components.gan_discriminator
        mcp = self.components.next_forcing_head

        gen_loss_val: float | None = None
        gen_grad_val: float | None = None
        gan_g_val: float | None = None
        gan_d_val: float | None = None
        mcp_loss_val: float | None = None
        gen_log: dict[str, Any] = {}

        if train_generator:
            self.optimizer.zero_grad(set_to_none=True)
            grad_step = self._dmd_sample_grad_step(len(sigma_list))
            if mcp is not None:
                with mcp.capturing():
                    x0_fake = run_generator(transformer, _noise(), cond, sigma_list, grad_step=grad_step)
            else:
                x0_fake = run_generator(transformer, _noise(), cond, sigma_list, grad_step=grad_step)

            dmd_sigma = self._dmd_sample_inner_sigma(dtype, latent_frames=K)
            dmd_loss, gen_log = compute_distribution_matching_loss(
                score_model,
                critic_lora,
                x0_fake=x0_fake,
                sigma=dmd_sigma,
                cond=cond,
                neg_cond=neg_cond,
                real_guidance_scale=real_cfg,
            )
            gen_total = dmd_loss

            if gan_on:
                if gan_disc is None:
                    raise RuntimeError("GAN is enabled but gan_discriminator is None")
                gan_g_sigma = self._dmd_sample_inner_sigma(dtype, latent_frames=K)
                gan_g_loss, gan_g_log = compute_gan_generator_loss(
                    score_model,
                    critic_lora,
                    gan_disc,
                    x0_fake=x0_fake,
                    sigma=gan_g_sigma,
                    cond=cond,
                    K=K,
                    H=H_lat,
                    W=W_lat,
                )
                gen_total = gen_total + float(self.cfg.dmd.gan_g_weight) * gan_g_loss
                gen_log.update(gan_g_log)
                gan_g_val = float(gan_g_loss.item())

            if mcp is not None:
                x0_next = meta["target_next_clean"]
                next_indices_grid = meta["next_indices_grid"]
                if x0_next is None or next_indices_grid is None:
                    raise RuntimeError("next_forcing enabled but next target conditioning is missing")
                mcp_sigma = self._dmd_sample_mcp_sigma(dtype, latent_frames=K)
                mcp_loss, mcp_log = mcp.compute_loss(
                    x0_next=x0_next,
                    noise=torch.randn_like(x0_next),
                    sigma=mcp_sigma,
                    next_indices_grid=next_indices_grid,
                    fps=self.cfg.sample.fps,
                    K=K,
                    H=H_lat,
                    W=W_lat,
                )
                _nf_w = float(self.cfg.next_forcing.loss_weight)
                _nf_ws = int(getattr(self.cfg.next_forcing, "loss_weight_warmup_steps", 0) or 0)
                if _nf_ws > 0:
                    _nf_w = _nf_w * min(1.0, float(self.global_step) / float(_nf_ws))
                gen_total = gen_total + _nf_w * mcp_loss
                gen_log.update(mcp_log)
                mcp_loss_val = float(mcp_loss.item())

            gen_total.backward()
            self._sync_grads_outside_fsdp()   # HistoryEncoder + LoRA (idempotent) + trainable transformer params when FSDP is off
            gen_trainable = [p for p in transformer.parameters() if p.requires_grad]
            if self.history_encoder is not None:
                gen_trainable += [p for p in self.history_encoder.parameters() if p.requires_grad]
            if self.components.lora_manager is not None:
                lora_params = self.components.lora_manager.get_trainable_parameters()
                gen_trainable += lora_params
                self._allreduce_grads(lora_params)
            if mcp is not None:
                mcp_params = mcp.trainable_parameters()
                gen_trainable += mcp_params
                self._allreduce_grads(mcp_params)

            gen_grad = torch.nn.utils.clip_grad_norm_(gen_trainable, self.cfg.optimizer.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            gen_loss_val = float(dmd_loss.item())
            gen_grad_val = float(gen_grad.item())

        self.critic_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            x0_fake_c = run_generator(transformer, _noise(), cond, sigma_list, grad_step=None)
        critic_sigma = self._dmd_sample_inner_sigma(dtype, latent_frames=K)
        critic_loss, critic_log = compute_critic_loss(
            score_model,
            critic_lora,
            x0_fake=x0_fake_c,
            sigma=critic_sigma,
            cond=cond,
        )
        critic_loss.backward()

        clip_params = list(critic_lora.get_trainable_parameters())
        if gan_on:
            if gan_disc is None:
                raise RuntimeError("GAN is enabled but gan_discriminator is None")
            gan_d_sigma = self._dmd_sample_inner_sigma(dtype, latent_frames=K)
            gan_d_loss, gan_d_log = compute_gan_critic_loss(
                score_model,
                critic_lora,
                gan_disc,
                x0_fake=x0_fake_c,
                x0_real=meta["target_clean"],
                sigma=gan_d_sigma,
                cond=cond,
                K=K,
                H=H_lat,
                W=W_lat,
                gan_d_weight=float(self.cfg.dmd.gan_d_weight),
                r1_weight=float(self.cfg.dmd.r1_weight),
                r2_weight=float(self.cfg.dmd.r2_weight),
                r1_sigma=float(self.cfg.dmd.r1_sigma),
                r2_sigma=float(self.cfg.dmd.r2_sigma),
                split_backward=True,
            )
            if gan_d_loss is not None:
                gan_d_loss.backward()
            clip_params += gan_disc.trainable_parameters()
            critic_log.update(gan_d_log)
            gan_d_val = float(gan_d_log["gan_D_loss"])

        self._allreduce_grads(clip_params)
        critic_grad = torch.nn.utils.clip_grad_norm_(clip_params, self.cfg.optimizer.max_grad_norm)
        self.critic_optimizer.step()
        critic_lora.enable()

        info = {
            "K": K,
            "gap_steps": meta["gap_steps"],
            "cond_mode": meta["cond_mode"],
            "cond_end": meta["cond_end"],
            "control_modes": meta["control_modes"],
            "spatial_context": meta["spatial_context"],
            "sigma": float(critic_sigma.item()),
            "memory_dropped": False,
            "source": meta["source"],
            "video_id": meta["video_id"],
            "frame_start": meta["frame_start"],
            "frame_end": meta["frame_end"],
            "anti_drift": "dmd",
            "inject_latent": False,
            "inject_history": False,
            "condition_prefix_fixed": False,
            "train_generator": train_generator,
            "dmd_loss": gen_loss_val,
            "dmd_grad": gen_grad_val,
            "gan_g_loss": gan_g_val,
            "gan_d_loss": gan_d_val,
            "gan_on": gan_on,
            "mcp_loss": mcp_loss_val,
            "critic_loss": float(critic_loss.item()),
            "critic_grad": float(critic_grad.item()),
            **{f"dmd/{k}": v for k, v in gen_log.items()},
            **{f"critic/{k}": v for k, v in critic_log.items()},
        }
        return float(critic_loss.item()), float(critic_grad.item()), info
