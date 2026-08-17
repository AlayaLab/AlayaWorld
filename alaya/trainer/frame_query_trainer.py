from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from alaya.trainer.rollout_trainer import RolloutTrainer, _first_scalar


class FrameQueryTrainer(RolloutTrainer):
    """History pretraining (frame-query) trainer.

    Objective: reconstruct Omega latent frames inside a masked history window (masked autoencoding)
    in order to pretrain the HistoryEncoder (plus an optional LoRA).

    Fundamental difference from RolloutTrainer.train_one_step:
      - normal rollout: history is a past segment before the target, and target is a future chunk
      - here: H_full is the first T_history latents of the clip and Omega is a random contiguous run
        of K frames inside H_full; non-Omega frames are noised and Omega frames stay clean for the
        HistoryEncoder, then only the Omega frames are reconstructed with a velocity MSE.

    Setup, model loading, HistoryEncoder construction, dataloader, optimizer, checkpointing and
    validation are reused from RolloutTrainer; only the training step is overridden. No camera,
    spatial memory or next-forcing is involved.
    """

    def setup(self) -> None:
        if not self.cfg.frame_query.enabled:
            raise ValueError("FrameQueryTrainer requires frame_query.enabled=true")
        RolloutTrainer.setup(self)
        if self.history_encoder is None:
            raise RuntimeError(
                "frame_query requires a HistoryEncoder (layout.history_latent_frames > 0), "
                "otherwise there is no trainable memory component"
            )

    def train_one_step(self, batch: Any) -> tuple[float, float, dict[str, Any]]:
        assert self.components is not None
        assert self.optimizer is not None
        assert self.history_encoder is not None

        self.components.transformer.train()
        self.history_encoder.train(self.cfg.memory.train)
        self.optimizer.zero_grad(set_to_none=True)

        video_pixels, caption, metadata = self._unpack_batch(batch)
        prompt_caption = self._caption_with_prefix(caption, metadata)
        context = self._encode_caption(prompt_caption, sync=False)
        latent_full = self._encode_video(video_pixels, metadata=metadata)         # [B, C, T, H, W]

        B, _C_lat, T_lat_full, H_lat, W_lat = latent_full.shape
        T_history = min(int(self.cfg.layout.history_latent_frames), int(T_lat_full))
        H_full = latent_full[:, :, :T_history].contiguous()     # [B, C, T_history, H, W]

        fq = self.cfg.frame_query
        omega_sizes = [int(k) for k in fq.omega_sizes]
        omega_probs = torch.tensor([float(p) for p in fq.omega_probs], dtype=torch.double)
        K = min(int(omega_sizes[int(torch.multinomial(omega_probs, 1).item())]), T_history)
        if K >= T_history:
            start, K = 0, T_history
        else:
            start = random.randint(0, T_history - K)
        Omega = list(range(start, start + K))

        non_omega = sorted(set(range(T_history)) - set(Omega))
        mask_sigma_per_frame = torch.zeros(T_history, device=self.dist.device, dtype=H_full.dtype)
        if non_omega:
            lo, hi = float(fq.mask_sigma_min), float(fq.mask_sigma_max)
            samp = torch.rand(len(non_omega), device=self.dist.device, dtype=H_full.dtype) * (hi - lo) + lo
            mask_sigma_per_frame[torch.tensor(non_omega, device=self.dist.device, dtype=torch.long)] = samp
        mask_sigma = mask_sigma_per_frame.view(1, 1, T_history, 1, 1)
        mask_noise = torch.randn_like(H_full)
        H_masked = (1.0 - mask_sigma) * H_full + mask_sigma * mask_noise

        # ===== 4. HistoryEncoder forward =====
        mem_tokens, mem_indices = self.history_encoder(H_masked)

        X_omega = H_full[:, :, Omega].contiguous()              # [B, C, K, H, W]
        sigma = torch.rand(1, device=self.dist.device, dtype=X_omega.dtype)
        diff_noise = torch.randn_like(X_omega)
        sigma_view = sigma.view(1, 1, 1, 1, 1)
        X_omega_noisy = (1.0 - sigma_view) * X_omega + sigma_view * diff_noise

        Omega_tensor = torch.tensor(Omega, device=self.dist.device, dtype=torch.float32)
        pred_velocity = self.components.transformer(
            x=[X_omega_noisy.squeeze(0)],
            t=sigma * 1000.0,
            context=[context],
            seq_len=K * H_lat * W_lat,
            fps=self.cfg.sample.fps,
            history_kv_tokens=mem_tokens,
            history_indices_grid=mem_indices,
            gen_t_indices_override=Omega_tensor,
        )

        # ===== 7. Velocity loss + backward =====
        target_velocity = (diff_noise - X_omega).to(dtype=pred_velocity.dtype)
        loss = F.mse_loss(pred_velocity, target_velocity)
        loss.backward()

        self._sync_grads_outside_fsdp()   # all-reduce the HistoryEncoder and LoRA gradients across ranks
        trainable: list[torch.nn.Parameter] = [
            p for p in self.history_encoder.parameters() if p.requires_grad
        ]
        trainable += [p for p in self.components.transformer.parameters() if p.requires_grad]
        if self.components.lora_manager is not None:
            trainable += self.components.lora_manager.get_trainable_parameters()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.cfg.optimizer.max_grad_norm)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # (gap_steps/cond_mode/cond_end/control_modes/spatial_context/memory_dropped/
        return float(loss.item()), float(grad_norm.item()), {
            "K": K,
            "gap_steps": 0,
            "cond_mode": "fq",
            "cond_end": 0,
            "control_modes": [],
            "spatial_context": 0,
            "memory_dropped": False,
            "anti_drift": "clean",
            "sigma": float(sigma.item()),
            "source": str(metadata.get("source", "")),
            "video_id": str(metadata.get("video_id", "unknown")),
            "frame_start": int(_first_scalar(metadata.get("frame_start", -1))),
            "frame_end": int(_first_scalar(metadata.get("frame_end", -1))),
            "omega_start": start,
            "T_history": T_history,
        }

    @torch.no_grad()
    def validate(self, step: int) -> None:
        """Frame-query reconstruction eval.

        For each validation sample: masked history -> HistoryEncoder -> multi-step flow matching from
        pure noise -> paste the predicted Omega back into H_full -> decode both ground truth and
        prediction and concatenate them side by side. Uses a fixed Omega (middle of the history) and
        """
        assert self.components is not None
        if not self.cfg.validation.enabled:
            return
        from ltx2.modules.scheduler import LTX2Scheduler

        comp = self.components
        comp.transformer.eval()
        assert self.history_encoder is not None
        self.history_encoder.eval()
        dev = self.dist.device
        dtype = self.dtype
        fq = self.cfg.frame_query
        steps = int(self.cfg.validation.sampling_steps)
        n_samples = max(1, int(self.cfg.validation.max_samples))
        val_K = int(max(fq.omega_sizes))
        out_dir = Path(self.cfg.run.output_dir) / "validation" / f"step-{step:06d}" / "fq_recon"
        out_dir.mkdir(parents=True, exist_ok=True)

        sigmas = LTX2Scheduler().execute(
            steps=steps, latent=None, max_shift=2.05, base_shift=0.95, stretch=True, terminal=0.1
        ).to(device=dev, dtype=torch.float32)

        if getattr(self, "_fq_eval_loader", None) is None:
            from alaya.data import build_train_dataloader
            self._fq_eval_loader = build_train_dataloader(self.cfg, self.dist)
        loader_iter = iter(self._fq_eval_loader)

        for i in range(n_samples):
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            video_pixels, caption, metadata = self._unpack_batch(batch)
            context = self._encode_caption(self._caption_with_prefix(caption, metadata), sync=False)
            latent_full = self._encode_video(video_pixels, metadata=metadata)
            B, C, T, H_lat, W_lat = latent_full.shape
            T_hist = min(int(self.cfg.layout.history_latent_frames), int(T))
            H_full = latent_full[:, :, :T_hist].contiguous()
            K = min(val_K, T_hist)
            start = (T_hist - K) // 2                       # always take the middle of the history
            Omega = list(range(start, start + K))

            non_omega = sorted(set(range(T_hist)) - set(Omega))
            msig = torch.zeros(T_hist, device=dev, dtype=H_full.dtype)
            if non_omega:
                lo, hi = float(fq.mask_sigma_min), float(fq.mask_sigma_max)
                s = torch.rand(len(non_omega), device=dev, dtype=H_full.dtype) * (hi - lo) + lo
                msig[torch.tensor(non_omega, device=dev, dtype=torch.long)] = s
            mask = msig.view(1, 1, T_hist, 1, 1)
            H_masked = (1.0 - mask) * H_full + mask * torch.randn_like(H_full)
            mem_tokens, mem_indices = self.history_encoder(H_masked)

            Ot = torch.tensor(Omega, device=dev, dtype=torch.float32)

            def _vel(x: torch.Tensor, sigma_val: float) -> torch.Tensor:
                return comp.transformer(
                    x=[x.squeeze(0)],
                    t=torch.tensor([sigma_val], device=dev, dtype=dtype) * 1000.0,
                    context=[context],
                    seq_len=K * H_lat * W_lat,
                    fps=self.cfg.sample.fps,
                    history_kv_tokens=mem_tokens,
                    history_indices_grid=mem_indices,
                    gen_t_indices_override=Ot,
                )

            X = torch.randn(B, C, K, H_lat, W_lat, device=dev, dtype=dtype)
            for j in range(len(sigmas) - 1):
                s_cur = float(sigmas[j].item())
                s_next = float(sigmas[j + 1].item())
                v = _vel(X, s_cur)
                X = X + (s_next - s_cur) * v
            s_last = float(sigmas[-1].item())
            x0 = X - s_last * _vel(X, s_last)               # remove the terminal residual noise to get x0

            H_pred = H_full.clone()
            H_pred[:, :, start:start + K] = x0.to(H_pred.dtype)
            gt_frames = self._decode_latent_to_video_frames(H_full)      # [Tpix, H, W, C] uint8
            pred_frames = self._decode_latent_to_video_frames(H_pred)
            stitched = torch.cat([gt_frames, pred_frames], dim=2)        # concatenate along width: GT | prediction
            self._write_video(
                out_dir / f"rank-{self.dist.rank:03d}_sample-{i:03d}_gt_L_recon_R.mp4", stitched
            )
            if self.dist.is_main:
                print(
                    f"[FQ-Val] step={step} sample={i} K={K} Omega=[{start}..{start + K - 1}] "
                    f"src={metadata.get('source', '')} video={metadata.get('video_id', '?')}",
                    flush=True,
                )
            del latent_full, H_full, H_masked, H_pred, mem_tokens, X, x0, gt_frames, pred_frames, stitched
            self._cleanup_cuda_cache()

        comp.transformer.train()
        self.history_encoder.train(self.cfg.memory.train)
