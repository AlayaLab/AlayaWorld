"""Self-Forcing++ Extended DMD (long-video self-rollout) for :class:`DmdTrainer`.

Ported from the reference SF++ implementation's inlined SF++ logic in ``alaya/trainer/rollout_trainer.py``
(methods ``_assemble_cond`` / ``_noise_like`` / ``_dmd_sample_rollout_depth`` /
``_dmd_self_rollout`` and the self-rollout branch of ``train_one_step_dmd``).

Structural adaptation: the reference implementation has no separate ``dmd_trainer.py`` — it inlined DMD+SF++ into
``rollout_trainer.py`` and refactored ``_build_conditioning`` to route through a shared
``_assemble_cond`` helper. this repo instead has a dedicated ``DmdTrainer(RolloutTrainer)``
whose ``_build_conditioning`` inlines the cond assembly. To keep the ``self_rollout.enabled=False``
path byte-for-byte identical to today (hard zero-regression requirement), we do NOT refactor
``_build_conditioning``; instead this mixin re-implements the equivalent cond assembly as
``_assemble_cond`` (mirroring this repo's inline block, including its ``action_history_memory``
support) and uses it only inside the self-rollout path. At rollout depth 0 with window_chunks==1,
``_assemble_cond`` reproduces this repo's ``_build_conditioning`` cond exactly.

Everything the SF++ step needs beyond the base cond (``latent_full`` / ``metadata`` /
``video_pixels`` / ``target_start``) is stashed into ``meta`` by ``_build_conditioning``; those are
additive keys that the disabled path never reads (no new RNG, no behavior change).

NOT ported from the reference implementation (see the SF++ port notes (not shipped)): the ``SFPP_DEBUG_*`` offline diagnostic dumps
(probes E1-E7), and the dataloader pixel/pose window extension required for ``score_gt_context`` to
reach deep rollout depths. ``score_gt_context`` here still works up to whatever depth the GT latent
already covers and otherwise raises the same guard the reference implementation raises.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Any

import torch

from alaya.dmd import (
    compute_critic_loss,
    compute_distribution_matching_loss,
    compute_gan_critic_loss,
    compute_gan_generator_loss,
    run_generator,
)
from alaya.dmd.consistency import compute_consistency_distill_loss
from alaya.dmd.losses import forward_velocity
from alaya.utils.distributed import broadcast_tensor


class DmdSelfRolloutMixin:
    """SF++ self-rollout methods mixed into :class:`DmdTrainer`.

    Relies on helpers provided by ``RolloutTrainer`` (``_assemble_cond`` uses
    ``history_encoder`` / ``_indices_grid`` / ``_local_*_t_offset`` /
    ``_local_target_t_indices`` / ``_indices_grid_for_t_indices`` / ``_build_control_kwargs``;
    the rollout uses ``_split_segments`` and the validation spatial-bank helpers) and on
    ``DmdTrainer``'s own ``_dmd_sample_inner_sigma`` / ``_dmd_sample_mcp_sigma`` /
    ``_dmd_sample_grad_step`` / ``_allreduce_grads``.
    """

    # ──────────────────────────────────────────────────────────────────────
    # dCM consistency regularizer (rCM-style CM+DMD joint, discrete, no JVP)
    # ──────────────────────────────────────────────────────────────────────
    def _cmreg_init_ema(self) -> None:
        """Initialize the EMA shadow of the consistency regularizer (generator LoRA only)."""
        self._cmreg_ema = None
        cmr = self.cfg.dmd.cm_reg
        if not cmr.enabled or not cmr.use_ema:
            return
        lm = self.components.lora_manager
        if lm is None:
            raise RuntimeError("cm_reg.use_ema requires a generator LoRA (training.mode=lora)")
        self._cmreg_ema = {
            name: (a.detach().clone(), b.detach().clone())
            for name, (a, b) in lm.lora_dict.items()
        }

    @contextmanager
    def _cmreg_ema_swapped(self):
        """Temporarily swap the generator LoRA for its EMA shadow during the target forward."""
        lm = self.components.lora_manager
        saved = {}
        for name, (a, b) in lm.lora_dict.items():
            saved[name] = (a.data.clone(), b.data.clone())
            ea, eb = self._cmreg_ema[name]
            a.data.copy_(ea)
            b.data.copy_(eb)
        lm.enable()
        try:
            yield
        finally:
            for name, (a, b) in lm.lora_dict.items():
                sa, sb = saved[name]
                a.data.copy_(sa)
                b.data.copy_(sb)

    def _cmreg_update_ema(self) -> None:
        cmr = self.cfg.dmd.cm_reg
        if getattr(self, "_cmreg_ema", None) is None:
            return
        lm = self.components.lora_manager
        w = float(cmr.ema_weight)
        warmup = self.global_step < int(cmr.ema_start_step)
        for name, (a, b) in lm.lora_dict.items():
            ea, eb = self._cmreg_ema[name]
            if warmup:
                ea.copy_(a.data)
                eb.copy_(b.data)
            else:
                ea.mul_(w).add_(a.data, alpha=1.0 - w)
                eb.mul_(w).add_(b.data, alpha=1.0 - w)

    def _cmreg_sample_pair(self) -> tuple[float, float]:
        """Sample an adjacent sigma pair (sigma_hi, sigma_lo), synchronized across ranks."""
        cmr = self.cfg.dmd.cm_reg
        n_scales = max(1, int(cmr.num_scales))
        if self.dist.is_main:
            n = torch.randint(low=1, high=n_scales + 1, size=(1,), device=self.dist.device, dtype=torch.long)
        else:
            n = torch.zeros(1, device=self.dist.device, dtype=torch.long)
        broadcast_tensor(n)
        n_i = int(n.item())
        smin, smax = float(cmr.sigma_min), float(cmr.sigma_max)
        span = smax - smin
        return smin + span * (n_i / n_scales), smin + span * ((n_i - 1) / n_scales)

    def _cmreg_loss(self, cond: dict[str, Any], meta: dict[str, Any]):
        """Teacher-forced consistency loss (teacher = frozen base with LoRA off, student = base + LoRA).
        x0 is the ground-truth target latent and the context comes from _build_conditioning."""
        cmr = self.cfg.dmd.cm_reg
        lm = self.components.lora_manager
        sigma_hi, sigma_lo = self._cmreg_sample_pair()
        target_ctx = self._cmreg_ema_swapped if (cmr.use_ema and self._cmreg_ema is not None) else None
        return compute_consistency_distill_loss(
            self.components.transformer,
            lm,
            x0=meta["target_clean"],
            cond=cond,
            sigma_hi=sigma_hi,
            sigma_lo=sigma_lo,
            loss_type=cmr.loss_type,
            huber_c=float(cmr.huber_c),
            target_ctx=target_ctx,
        )

    # ──────────────────────────────────────────────────────────────────────
    # cond assembly (mirrors this repo DmdTrainer._build_conditioning inline block)
    # ──────────────────────────────────────────────────────────────────────
    def _assemble_cond(
        self,
        *,
        context: torch.Tensor,
        B: int,
        H_lat: int,
        W_lat: int,
        dtype: torch.dtype,
        sink_latent: torch.Tensor | None,
        history_latent: torch.Tensor | None,
        nearby_latent: torch.Tensor | None,
        target_action_start: int,
        K: int,
        cond_end: int,
        gap_steps: int,
        spatial_context: dict[str, Any] | None,
        control_modes: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the transformer forward cond (everything except x/t).

        Local RoPE (sink@0 / mem@1.. / target@1+N..) is independent of rollout depth; only the
        *action* indices advance with the absolute ``target_action_start`` (out-of-range camera is
        clamped inside ``build_action_vectors``). Shapes come from the caller (B/H_lat/W_lat/dtype)
        rather than being inferred from optional latents (history/nearby can both be None). ``K`` is
        the number of *target* frames for this cond (== window_chunks*K for a whole-window cond).
        """
        N = int(self.cfg.layout.history_latent_frames)
        mem_tokens = None
        mem_indices = None
        if history_latent is not None:
            if self.history_encoder is None:
                raise RuntimeError("history_latent present but history_encoder is None")
            mem_tokens, mem_indices = self.history_encoder(history_latent)
            mem_indices = mem_indices.clone()
            mem_indices[:, 0, :, :] += self._local_memory_t_offset(N, cond_end)
        sink_indices = (
            self._indices_grid(
                B,
                self.cfg.layout.sink_latent_frames,
                H_lat,
                W_lat,
                t_offset=self._local_sink_t_offset(N),
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
                t_offset=self._local_nearby_t_offset(N, cond_end, gap_steps=gap_steps),
            )
            if cond_end > 0
            else None
        )
        target_rope_t_indices = self._local_target_t_indices(
            K, history_latent_frames=N, condition_latent_frames=cond_end, gap_steps=gap_steps
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
            target_action_start, target_action_start + K, device=self.dist.device, dtype=torch.float32
        )
        condition_action_t_indices = (
            torch.arange(
                target_action_start - cond_end, target_action_start, device=self.dist.device, dtype=torch.float32
            )
            if cond_end > 0
            else None
        )
        history_action_t_indices = (
            torch.arange(
                target_action_start - N,
                target_action_start,
                device=self.dist.device,
                dtype=torch.float32,
            )
            if (bool(getattr(self.cfg.control, "action_history_memory", False)) and history_latent is not None)
            else None
        )
        control_kwargs = self._build_control_kwargs(
            metadata=metadata,
            control_modes=control_modes,
            target_t_indices=target_action_t_indices,
            condition_t_indices=condition_action_t_indices,
            history_t_indices=history_action_t_indices,
            dtype=dtype,
        )
        return dict(
            context=[context],
            seq_len=K * H_lat * W_lat,
            fps=self.cfg.sample.fps,
            history_kv_tokens=mem_tokens,
            history_indices_grid=mem_indices,
            gen_t_indices_override=target_rope_t_indices,
            sink_latent=sink_latent,
            sink_indices_grid=sink_indices,
            spatial_latent=spatial_latent,
            spatial_mask_patch=spatial_mask_patch,
            spatial_indices_grid=spatial_indices,
            nearby_latent=nearby_latent,
            nearby_indices_grid=nearby_indices,
            **control_kwargs,
        )

    # ──────────────────────────────────────────────────────────────────────
    # rollout helpers
    # ──────────────────────────────────────────────────────────────────────
    def _noise_like(self, meta: dict[str, Any]) -> torch.Tensor:
        """Fresh Gaussian noise shaped like one target chunk [B,C,K,H_lat,W_lat]."""
        return torch.randn(
            int(meta["B"]),
            int(meta["C"]),
            int(meta["K"]),
            int(meta["H_lat"]),
            int(meta["W_lat"]),
            device=self.dist.device,
            dtype=meta["dtype"],
        )

    def _dmd_sample_rollout_depth(self) -> int:
        """Sample this step's rollout depth r ~ Unif{max(min_depth, window_chunks-1)..max_chunks}.

        Lower bound is raised to window_chunks-1 so the supervised window [r-w+1..r] is entirely
        student-produced. Uniform r ⇒ uniform-random window start (== SF++'s uniform sliding window).
        Broadcast from rank0 for cross-rank consistency (same pattern as ``_dmd_sample_grad_step``).
        Called ONLY inside the self_rollout.enabled branch → disabled path draws no RNG (zero
        regression). Deterministic early-out (no RNG) when lo==hi.
        """
        sr = self.cfg.dmd.self_rollout
        lo = max(int(sr.min_depth), int(sr.window_chunks) - 1)
        hi = int(sr.max_chunks)
        if hi <= lo:
            return lo
        if self.dist.is_main:
            d = torch.randint(low=lo, high=hi + 1, size=(1,), device=self.dist.device, dtype=torch.long)
        else:
            d = torch.zeros(1, device=self.dist.device, dtype=torch.long)
        broadcast_tensor(d)
        return int(d.item())

    def _dmd_self_rollout(
        self, *, base_cond: dict[str, Any], meta: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], list[torch.Tensor], dict[str, Any]]:
        """SF++ Extended DMD core: from the GT seed, roll the student forward r chunks under
        no_grad feeding its OWN outputs back into history/nearby; the supervised window is the
        ``window_chunks`` (=w) consecutive chunks ending at the (r+1)-th chunk.

        Returns (gen_cond, window_cond, win_prefix, window_info):
          - gen_cond:    chunk-level cond (K latent) for the final, grad-carrying chunk (context
                         rolled forward r rounds);
          - window_cond: scoring cond over the whole window (W=w*K latent). Its causal context is the
                         *window-start snapshot* (history/nearby just before chunk r-w+1 was made),
                         with target RoPE / action / seq_len covering W. Variants: score_context_free
                         (strip all causal context, re-index window RoPE to 0..W-1, keep noised window
                         + prompt only) and score_gt_context (slice GT latent context by window
                         position); mutually exclusive; default = drifted-context snapshot;
          - win_prefix:  the first w-1 window chunks as no_grad rollout outputs (time order); combined
                         with the grad-carrying final chunk into the window. w==1 ⇒ empty list and
                         window_cond IS gen_cond (byte-identical to the old single-chunk behavior).

        Reuses run_generator (consistency sampling) + _assemble_cond + the validation spatial bank.
        Per-round local RoPE is unchanged; only target_action_start advances absolutely. The whole
        rollout is under no_grad. window_info["window_gen_cond"] is the full-context, target=W
        generation cond used by window_full_grad (None otherwise).
        """
        sr = self.cfg.dmd.self_rollout
        K, cond_end, gap_steps = int(meta["K"]), int(meta["cond_end"]), int(meta["gap_steps"])
        B, H_lat, W_lat, dtype = int(meta["B"]), int(meta["H_lat"]), int(meta["W_lat"]), meta["dtype"]
        N = int(self.cfg.layout.history_latent_frames)
        sigma_list = [float(s) for s in self.cfg.dmd.dmd_sigma_list]
        latent_full = meta["latent_full"]
        batch_meta = meta["metadata"]
        control_modes = meta["control_modes"]
        context = base_cond["context"][0]
        transformer = self.components.transformer

        _r0nm = bool(getattr(sr, "round0_no_memory", False))
        _seed = meta.get("vigeo_seed_conds")
        if _seed is not None:
            sink_latent = _seed["sink"]
            history_latent = _seed["history"]
            nearby_latent = _seed["nearby"]
            target_start = int(meta["target_start"])
        else:
            sink_latent, history_latent, nearby_latent, _tgt, target_start = self._split_segments(
                latent_full, K=K, gap_steps=gap_steps, cond_end=cond_end
            )
        use_bank = bool(
            sr.reuse_spatial_bank
            and self.cfg.spatial_memory.enabled
        )
        spatial_bank = (
            self._init_validation_rollout_spatial_bank(
                video_pixels=meta["video_pixels"], metadata=batch_meta, target_start=target_start,
                history_latent_frames=N,
            )
            if use_bank
            else None
        )
        spatial_dropped = 0
        _bank_vigeo = bool(spatial_bank is not None and self._uses_vigeo_prefix_last_frame())
        if _bank_vigeo:
            _bank_gt_prob = float(getattr(sr, "spatial_bank_gt_prob", 0.5))
            _bank_gt_per_rollout = str(getattr(sr, "spatial_bank_gt_mode", "per_chunk")) == "per_sample"
            _bank_rollout_use_gt = random.random() < _bank_gt_prob if _bank_gt_per_rollout else None
            _bank_stride = int(self.cfg.sample.temporal_stride)
            _bank_px_off = int(getattr(spatial_bank, "vigeo_pixel_offset", 0))
            _bank_gt_px = self._video_pixels_to_bcfhw(meta["video_pixels"]).to(
                device=self.dist.device, dtype=dtype
            )
            _prefix_px = self._vigeo_target_prefix_pixel_frames()
            _motion_px = self._vigeo_motion_pixel_frames()
            _bank_anchor, _bank_motion = self._encode_vigeo_motion_window(
                self._slice_video_pixel_frames(
                    meta["video_pixels"], _prefix_px - _motion_px, _prefix_px
                )
            )

        def _round_spatial(cur_start: int, frames: int = K) -> dict[str, Any] | None:
            nonlocal spatial_dropped
            if spatial_bank is None:
                return None
            sc = self._build_validation_rollout_bank_spatial_context(
                bank=spatial_bank,
                metadata=batch_meta,
                target_start=cur_start,
                K=frames,
                target_rope_t_indices=self._local_target_t_indices(
                    frames, history_latent_frames=N, condition_latent_frames=cond_end, gap_steps=gap_steps
                ),
            )
            if sc is None:
                spatial_dropped += 1
            return sc

        w = int(sr.window_chunks)
        r = self._dmd_sample_rollout_depth()  # r >= w-1 (lower bound already raised)
        win_chunk_idx = r - w + 1  # rollout index of the window's first chunk (uniform r ⇒ uniform start)
        _ar_grad = bool(getattr(sr, "window_ar_grad", False)) and (
            self.global_step % int(self.cfg.dmd.dfake_gen_update_ratio) == 0
        )
        # Window-start context snapshot; win_chunk_idx==0 ⇒ GT seed state (snapshot taken before loop).
        win_history, win_nearby = history_latent, nearby_latent
        win_prefix: list[torch.Tensor] = []  # the w-1 window chunks before the final one (time order)
        with torch.enable_grad() if _ar_grad else torch.no_grad():
            for j in range(r):
                _grad_this = _ar_grad and j >= win_chunk_idx
                _ctx = torch.enable_grad() if _grad_this else torch.no_grad()
                if w > 1 and j == win_chunk_idx:
                    win_history, win_nearby = history_latent, nearby_latent
                cur_start = target_start + j * K
                with _ctx:
                    cond_j = self._assemble_cond(
                        context=context,
                        B=B,
                        H_lat=H_lat,
                        W_lat=W_lat,
                        dtype=dtype,
                        sink_latent=sink_latent,
                        history_latent=(None if (_r0nm and cur_start == target_start) else history_latent),
                        nearby_latent=nearby_latent,
                        target_action_start=cur_start,
                        K=K,
                        cond_end=cond_end,
                        gap_steps=gap_steps,
                        spatial_context=_round_spatial(cur_start),
                        control_modes=control_modes,
                        metadata=batch_meta,
                    )
                    x0_j = run_generator(
                        transformer, self._noise_like(meta), cond_j, sigma_list,
                        grad_step=(len(sigma_list) - 1 if _grad_this else None),
                    )
                if j >= win_chunk_idx:
                    win_prefix.append(x0_j)
                _x0_ng = x0_j.detach() if _grad_this else x0_j
                if spatial_bank is not None:
                    if _bank_vigeo:
                        _ps = cur_start * _bank_stride + _bank_px_off
                        _gt_ok = (
                            int(latent_full.shape[2]) >= cur_start + K
                            and int(_bank_gt_px.shape[2]) >= _ps + K * _bank_stride
                        )
                        _use_gt_this = (
                            _bank_rollout_use_gt
                            if _bank_gt_per_rollout
                            else random.random() < _bank_gt_prob
                        )
                        if _gt_ok and _use_gt_this:
                            self._append_validation_rollout_spatial_bank_prediction(
                                bank=spatial_bank,
                                pred_latent=latent_full[:, :, cur_start:cur_start + K].contiguous(),
                                decoded_pixels=_bank_gt_px[:, :, _ps:_ps + K * _bank_stride].contiguous(),
                                metadata=batch_meta, target_start=cur_start,
                            )
                        else:
                            with torch.no_grad():
                                _tgt_px, _, _ = self._decode_and_reencode_vigeo_motion_chunk(
                                    anchor_latent=_bank_anchor.to(dtype),
                                    motion_latent=_bank_motion.to(dtype),
                                    target_latent=_x0_ng.to(dtype),
                                )
                            self._append_validation_rollout_spatial_bank_prediction(
                                bank=spatial_bank, pred_latent=_x0_ng,
                                decoded_pixels=_tgt_px,
                                metadata=batch_meta, target_start=cur_start,
                            )
                        _bank_anchor = _x0_ng[:, :, -2:-1].contiguous()
                        _bank_motion = _x0_ng[:, :, -1:].contiguous()
                    else:
                        self._append_validation_rollout_spatial_bank_prediction(
                            bank=spatial_bank, pred_latent=_x0_ng, decoded_pixels=None,
                            metadata=batch_meta, target_start=cur_start
                        )
                if history_latent is not None:
                    history_latent = torch.cat(
                        [history_latent, _x0_ng.to(history_latent.dtype)], dim=2
                    )[:, :, -N:].contiguous()
                    nearby_latent = history_latent[:, :, -cond_end:].contiguous() if cond_end > 0 else None
                elif cond_end > 0:
                    nearby_latent = _x0_ng[:, :, -cond_end:].to(dtype).contiguous()
                if _grad_this and cond_end > 0:
                    nearby_latent = x0_j[:, :, -cond_end:].to(dtype)
        gen_start = target_start + r * K
        win_start = target_start + win_chunk_idx * K  # w==1 ⇒ == gen_start
        gen_cond = self._assemble_cond(
            context=context,
            B=B,
            H_lat=H_lat,
            W_lat=W_lat,
            dtype=dtype,
            sink_latent=sink_latent,
            history_latent=(None if (_r0nm and gen_start == target_start) else history_latent),
            nearby_latent=nearby_latent,
            target_action_start=gen_start,
            K=K,
            cond_end=cond_end,
            gap_steps=gap_steps,
            spatial_context=_round_spatial(gen_start),
            control_modes=control_modes,
            metadata=batch_meta,
        )

        def _window_snapshot_cond() -> dict[str, Any]:
            """Full-causal-context window cond (target=W latent, context=window-start snapshot).

            The default w>1 scoring cond; also the *generation* cond for whole-window regeneration
            under window_full_grad. w==1 ⇒ window == final chunk, byte-identical to gen_cond
            (reuse directly → zero regression).
            """
            if w == 1:
                return gen_cond
            return self._assemble_cond(
                context=context,
                B=B,
                H_lat=H_lat,
                W_lat=W_lat,
                dtype=dtype,
                sink_latent=sink_latent,
                history_latent=(None if (_r0nm and win_start == target_start) else win_history),
                nearby_latent=win_nearby,
                target_action_start=win_start,
                K=w * K,
                cond_end=cond_end,
                gap_steps=gap_steps,
                spatial_context=_round_spatial(win_start, frames=w * K),
                control_modes=control_modes,
                metadata=batch_meta,
            )

        # window_full_grad (stage 2): whole-window regeneration needs a "full causal context +
        # target=W" generation cond.
        window_gen_cond = _window_snapshot_cond() if bool(getattr(sr, "window_full_grad", False)) else None

        if bool(getattr(sr, "score_context_free", False)):
            # Marginal-distribution scoring (aligns with the SF++ paper): scoring cond = noised window
            # + prompt, with NO causal context; window RoPE re-indexed to a standalone 0..W-1 clip.
            # Generation still uses gen_cond (full context); critic/teacher score with this same cond.
            window_cond = dict(
                context=[context],
                seq_len=(w * K) * H_lat * W_lat,
                fps=self.cfg.sample.fps,
                gen_t_indices_override=torch.arange(0, w * K, device=self.dist.device, dtype=torch.float32),
                history_kv_tokens=None,
                history_indices_grid=None,
                sink_latent=None,
                sink_indices_grid=None,
                spatial_latent=None,
                spatial_mask_patch=None,
                spatial_indices_grid=None,
                nearby_latent=None,
                nearby_indices_grid=None,
            )
        elif bool(getattr(sr, "score_gt_context", False)):
            # GT-context scoring (stage-2.5): history/nearby sliced from the ground-truth latent by
            # window position (sink stays GT; spatial not used in scoring); action indices match the
            # generation side (absolute win_start). Requires the GT latent to cover the window start;
            # the dataloader pixel-window extension that guarantees this at deep depth is NOT ported
            # (see the SF++ port notes (not shipped)) — beyond GT coverage this raises, same as the reference implementation.
            if int(latent_full.shape[2]) < win_start:
                raise ValueError(
                    f"score_gt_context needs GT latent covering the window start (T>={win_start}), "
                    f"got T={int(latent_full.shape[2])}; the dataloader pixel-window extension "
                    f"(_pixel_extra_frames_for_self_rollout, max_chunks={int(sr.max_chunks)}) is NOT "
                    f"ported in this repo — use score_gt_context only at depths the GT already covers."
                )
            if _seed is not None and win_start == target_start:
                gt_history = _seed["history"]
                gt_nearby = _seed["nearby"]
            elif N > 0:
                gt_history = latent_full[:, :, win_start - N:win_start].contiguous()
                gt_nearby = gt_history[:, :, -cond_end:].contiguous() if cond_end > 0 else None
            else:
                gt_history = None
                gt_nearby = (
                    latent_full[:, :, win_start - cond_end:win_start].contiguous() if cond_end > 0 else None
                )
            _gt_score_sp = None
            if (
                bool(getattr(sr, "score_gt_spatial", False))
                and self.cfg.spatial_memory.enabled
            ):
                try:
                    with torch.no_grad():
                        _gtb = self._init_validation_rollout_spatial_bank(
                            video_pixels=meta["video_pixels"],
                            metadata=batch_meta,
                            target_start=target_start,
                            history_latent_frames=N,
                        )
                        if _gtb is not None:
                            _st = int(self.cfg.sample.temporal_stride)
                            _off = int(getattr(_gtb, "vigeo_pixel_offset", 0))
                            _px = self._video_pixels_to_bcfhw(meta["video_pixels"]).to(
                                device=self.dist.device, dtype=dtype
                            )
                            _n_app = 0
                            for _j in range(win_chunk_idx):
                                _cs = target_start + _j * K
                                _ps = _cs * _st + _off
                                if int(latent_full.shape[2]) < _cs + K or int(_px.shape[2]) < _ps + K * _st:
                                    break
                                self._append_validation_rollout_spatial_bank_prediction(
                                    bank=_gtb,
                                    pred_latent=latent_full[:, :, _cs:_cs + K].contiguous(),
                                    decoded_pixels=_px[:, :, _ps:_ps + K * _st].contiguous(),
                                    metadata=batch_meta,
                                    target_start=_cs,
                                )
                                _n_app += 1
                            if _n_app == win_chunk_idx:
                                _gt_score_sp = self._build_validation_rollout_bank_spatial_context(
                                    bank=_gtb,
                                    metadata=batch_meta,
                                    target_start=win_start,
                                    K=w * K,
                                    target_rope_t_indices=self._local_target_t_indices(
                                        w * K,
                                        history_latent_frames=N,
                                        condition_latent_frames=cond_end,
                                        gap_steps=gap_steps,
                                    ),
                                )
                except Exception as _e:
                    _gt_score_sp = None
                    print(f"[score_gt_spatial] skip (fail-open): {_e}", flush=True)
            window_cond = self._assemble_cond(
                context=context,
                B=B,
                H_lat=H_lat,
                W_lat=W_lat,
                dtype=dtype,
                sink_latent=sink_latent,
                history_latent=(None if (_r0nm and win_start == target_start) else gt_history),
                nearby_latent=gt_nearby,
                target_action_start=win_start,
                K=w * K,
                cond_end=cond_end,
                gap_steps=gap_steps,
                spatial_context=_gt_score_sp,
                control_modes=control_modes,
                metadata=batch_meta,
            )
        else:
            window_cond = window_gen_cond if window_gen_cond is not None else _window_snapshot_cond()

        info = {
            "depth": r,
            "win_start": win_start,
            "window_chunks": w,
            "ar_grad": bool(_ar_grad),
            "spatial_dropped": spatial_dropped,
            "bank_mode": (
                ("gt" if _bank_rollout_use_gt else "pred")
                if (_bank_vigeo and _bank_gt_per_rollout)
                else ("mix" if _bank_vigeo else "off")
            ),
            "window_gen_cond": window_gen_cond,
            "rollout_history_latent": history_latent,
            "rollout_sink_latent": sink_latent,
            "gen_start": gen_start,
            "context": context,
        }
        return gen_cond, window_cond, win_prefix, info

    # ──────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────
    def _nf_teacher_next_target(
        self, *, meta: dict[str, Any], window_info: dict[str, Any], current_chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        K, cond_end, gap_steps = int(meta["K"]), int(meta["cond_end"]), int(meta["gap_steps"])
        B, H_lat, W_lat, dtype = int(meta["B"]), int(meta["H_lat"]), int(meta["W_lat"]), meta["dtype"]
        N = int(self.cfg.layout.history_latent_frames)
        context = window_info["context"]
        transformer = self.components.transformer
        lora_manager = self.components.lora_manager
        gen_start = int(window_info["gen_start"])
        roll_hist = window_info.get("rollout_history_latent")
        sink_latent = window_info.get("rollout_sink_latent")
        cur = current_chunk.to(dtype)

        if N > 0 and roll_hist is not None:
            teacher_hist = torch.cat([roll_hist, cur.to(roll_hist.dtype)], dim=2)[:, :, -N:].contiguous()
        else:
            teacher_hist = None
        if cond_end > 0:
            src = teacher_hist if teacher_hist is not None else cur
            teacher_nearby = src[:, :, -cond_end:].contiguous()
        else:
            teacher_nearby = None

        next_cond = self._assemble_cond(
            context=context, B=B, H_lat=H_lat, W_lat=W_lat, dtype=dtype,
            sink_latent=sink_latent, history_latent=teacher_hist, nearby_latent=teacher_nearby,
            target_action_start=gen_start + K, K=K, cond_end=cond_end, gap_steps=gap_steps,
            spatial_context=None, control_modes=meta["control_modes"], metadata=meta["metadata"],
        )
        steps = max(1, int(self.cfg.next_forcing.teacher_target_steps))
        teacher_sigmas = [1.0 - i / steps for i in range(steps)]  # decreasing, the last step reaches x0
        C = int(cur.shape[1])
        noise = torch.randn(B, C, K, H_lat, W_lat, device=self.dist.device, dtype=dtype)
        with torch.no_grad(), lora_manager.toggled(False):
            x0_next = run_generator(transformer, noise, next_cond, teacher_sigmas, grad_step=None)
        local_target = self._local_target_t_indices(
            K, history_latent_frames=N, condition_latent_frames=cond_end, gap_steps=gap_steps
        )
        next_indices_grid = self._indices_grid_for_t_indices(B, local_target + K, H_lat, W_lat)
        return x0_next.detach(), next_indices_grid

    # ──────────────────────────────────────────────────────────────────────
    # self-rollout DMD train step (adapted from the reference implementation train_one_step_dmd, SF++ branch)
    # ──────────────────────────────────────────────────────────────────────
    def _train_one_step_dmd_self_rollout(self, batch: Any) -> tuple[float, float, dict[str, Any]]:
        """One DMD step under SF++ self-rollout. Only reached when self_rollout.enabled.

        GAN is intentionally never active here (GAN and self_rollout are mutually exclusive — a
        discriminator inside rollout learns "context type" rather than real/fake; matches the reference implementation's
        ``gan_on = ... and not sr.enabled``). Next-Forcing MCP is applied only when the supervised
        window is round-0 (depth==0) and window_full_grad is off (no aligned next-chunk GT / hook
        otherwise).
        """
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

        sr = self.cfg.dmd.self_rollout
        gen_cond, window_cond, win_prefix, window_info = self._dmd_self_rollout(base_cond=cond, meta=meta)
        win_latents = len(win_prefix) * K + K  # total window latent count W (w==1 ⇒ K, old behavior)

        def _win(x0_chunk: torch.Tensor) -> torch.Tensor:
            """Window assembly: [prefix chunks, final chunk] (time dim). The prefix is detached in
            the plain path (grad only through the final chunk), but carries grad when
            window_ar_grad is active — see _win_ar_grad for the loss-scaling consequence."""
            if not win_prefix:
                return x0_chunk
            return torch.cat([*win_prefix, x0_chunk.to(win_prefix[0].dtype)], dim=2)

        neg_cond = None
        if real_cfg != 1.0:
            neg_caption = self.cfg.dmd.negative_prompt or self.cfg.validation.negative_prompt
            neg_context = self._encode_caption(neg_caption, sync=False)
            neg_cond = {**window_cond, "context": [neg_context]}

        def _noise() -> torch.Tensor:
            return torch.randn(B, C, K, H_lat, W_lat, device=self.dist.device, dtype=dtype)

        sr_full_grad = bool(getattr(sr, "window_full_grad", False))
        _win_ar_grad = bool(window_info.get("ar_grad", False))
        window_gen_cond = window_info.get("window_gen_cond")

        def _regen_window(keep_grad: bool, capture: bool = False) -> torch.Tensor:
            """Whole-window regeneration (SF++ paper Eq.2 gradient path): sample the final chunk
            fresh (no_grad, same distribution as the rollout), assemble the whole clean window,
            pick one student sigma step and re-noise the whole window (backward noise init), then let
            the student denoise the whole window in ONE forward pass. keep_grad=True builds the graph
            so the DMD gradient covers every window frame (incl. the drifted prefix); False draws the
            critic's same-distribution fake sample.

            capture=True (only with keep_grad, teacher_target on): wrap the grad-carrying forward in
            mcp.capturing() so the MCP hook grabs the window's gen-token features → next_forcing can
            coexist with window_full_grad (hook captures the last-chunk tokens = the chunk at gen_start)."""
            if window_gen_cond is None:
                raise RuntimeError("window_full_grad set but window_gen_cond is None")
            with torch.no_grad():
                x0_win = _win(run_generator(transformer, _noise(), gen_cond, sigma_list, grad_step=None))
                g = self._dmd_sample_grad_step(len(sigma_list))
                sigma_g = float(sigma_list[g])
                x_t_win = (1.0 - sigma_g) * x0_win + sigma_g * torch.randn_like(x0_win)
            sigma_t = torch.full((1,), sigma_g, device=self.dist.device, dtype=dtype)
            if keep_grad:
                if capture:
                    with mcp.capturing():
                        v = forward_velocity(transformer, x_t_win, sigma_t, window_gen_cond)
                else:
                    v = forward_velocity(transformer, x_t_win, sigma_t, window_gen_cond)
                return x_t_win - sigma_g * v
            with torch.no_grad():
                v = forward_velocity(transformer, x_t_win, sigma_t, window_gen_cond)
                return x_t_win - sigma_g * v

        train_generator = self.global_step % int(self.cfg.dmd.dfake_gen_update_ratio) == 0
        gan_on = bool(self.cfg.dmd.is_use_gan) and self.global_step >= int(self.cfg.dmd.gan_start_step)
        gan_g_start = max(int(self.cfg.dmd.gan_start_step), int(getattr(self.cfg.dmd, "gan_g_start_step", 0)))
        gan_g_on = gan_on and self.global_step >= gan_g_start
        gan_disc = self.components.gan_discriminator
        gan_g_val: float | None = None
        gan_d_val: float | None = None
        mcp = self.components.next_forcing_head  # Next-Forcing depth=1 MCP head (None = disabled)

        gen_loss_val: float | None = None
        gen_grad_val: float | None = None
        mcp_loss_val: float | None = None
        cm_loss_val: float | None = None
        seam_loss_val: float | None = None
        gen_log: dict[str, Any] = {}

        # ── GENERATOR (every dfake_gen_update_ratio steps) ──────────────────
        if train_generator:
            self.optimizer.zero_grad(set_to_none=True)
            nf_teacher = mcp is not None and bool(self.cfg.next_forcing.teacher_target)
            mcp_active = mcp is not None and (int(window_info["depth"]) == 0 or nf_teacher) and (
                not sr_full_grad or nf_teacher
            )
            if sr_full_grad:
                x0_fake = _regen_window(keep_grad=True, capture=mcp_active)
            else:
                grad_step = self._dmd_sample_grad_step(len(sigma_list))
                if mcp_active:
                    with mcp.capturing():
                        x0_fake_chunk = run_generator(
                            transformer, _noise(), gen_cond, sigma_list, grad_step=grad_step
                        )
                else:
                    x0_fake_chunk = run_generator(
                        transformer, _noise(), gen_cond, sigma_list, grad_step=grad_step
                    )
                x0_fake = _win(x0_fake_chunk)
            dmd_sigma = self._dmd_sample_inner_sigma(dtype, latent_frames=win_latents)
            dmd_loss, gen_log = compute_distribution_matching_loss(
                score_model,
                critic_lora,
                x0_fake=x0_fake,
                sigma=dmd_sigma,
                cond=window_cond,
                neg_cond=neg_cond,
                real_guidance_scale=real_cfg,
            )
            if win_latents != K and not sr_full_grad and not _win_ar_grad:
                # Window MSE mean's denominator includes the detached prefix (grad only flows through
                # the final chunk) → each final-chunk element's grad is diluted ×K/W; multiply back
                # W/K to keep the effective LR identical to w==1. window_full_grad / window_ar_grad
                # both put grad on all window frames (no dilution → no compensation); compensating
                # there would silently scale DMD ×W/K and break the DMD:cm:gan weight ratio.
                dmd_loss = dmd_loss * (float(win_latents) / float(K))
            gen_total = dmd_loss
            if gan_g_on:
                assert gan_disc is not None
                gan_g_sigma = self._dmd_sample_inner_sigma(dtype, latent_frames=win_latents)
                gan_g_loss, gan_g_log = compute_gan_generator_loss(
                    score_model,
                    critic_lora,
                    gan_disc,
                    x0_fake=x0_fake,
                    sigma=gan_g_sigma,
                    cond=window_cond,
                    K=win_latents,
                    H=H_lat,
                    W=W_lat,
                )
                gen_total = gen_total + float(self.cfg.dmd.gan_g_weight) * gan_g_loss
                gen_log.update(gan_g_log)
                gan_g_val = float(gan_g_loss.item())

            if mcp_active:
                mcp_sigma = self._dmd_sample_mcp_sigma(dtype, latent_frames=K)
                if nf_teacher:
                    x0_next, next_indices_grid = self._nf_teacher_next_target(
                        meta=meta, window_info=window_info,
                        current_chunk=x0_fake[:, :, -K:].detach(),
                    )
                else:
                    x0_next = meta["target_next_clean"]
                    next_indices_grid = meta["next_indices_grid"]
                if x0_next is None or next_indices_grid is None:
                    raise RuntimeError("next_forcing enabled but next target conditioning is missing")
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
                gen_log["nf_weight"] = _nf_w
                mcp_loss_val = float(mcp_loss.item())

            cmr = self.cfg.dmd.cm_reg
            if cmr.enabled and (self.global_step % max(1, int(cmr.every)) == 0):
                cm_loss, cm_log = self._cmreg_loss(cond, meta)
                if self.global_step < int(cmr.dmd_warmup_steps):
                    gen_total = float(cmr.weight) * cm_loss  # warmup: consistency term only
                else:
                    gen_total = gen_total + float(cmr.weight) * cm_loss
                gen_log.update({f"cm_{k}": v for k, v in cm_log.items()})
                cm_loss_val = float(cm_loss.item())

            seam_w = float(getattr(sr, "seam_loss_weight", 0.0) or 0.0)
            if seam_w > 0.0 and win_latents > K:
                xf = x0_fake.float()
                seam_terms = []
                for _b in range(K, win_latents, K):
                    prev_dc = xf[:, :, _b - 1:_b].mean(dim=(3, 4), keepdim=True)  # last frame of the previous chunk
                    cur_dc = xf[:, :, _b:_b + 1].mean(dim=(3, 4), keepdim=True)   # first frame of this chunk
                    seam_terms.append((cur_dc - prev_dc).pow(2).mean())
                seam_loss = torch.stack(seam_terms).mean()
                gen_total = gen_total + seam_w * seam_loss
                gen_log["seam_loss"] = float(seam_loss.item())
                seam_loss_val = float(seam_loss.item())

            gen_total.backward()
            self._sync_grads_outside_fsdp()   # HistoryEncoder + LoRA (idempotent) and, with FSDP off, the transformer
            gen_trainable = [p for p in transformer.parameters() if p.requires_grad]
            if self.history_encoder is not None:
                gen_trainable += [p for p in self.history_encoder.parameters() if p.requires_grad]
            if self.components.lora_manager is not None:
                lora_params = self.components.lora_manager.get_trainable_parameters()
                gen_trainable += lora_params
                self._allreduce_grads(lora_params)
            if mcp_active:
                # MCP head has grad only at depth==0 → include in clip/all-reduce only then.
                mcp_params = mcp.trainable_parameters()
                gen_trainable += mcp_params
                self._allreduce_grads(mcp_params)

            gen_grad = torch.nn.utils.clip_grad_norm_(gen_trainable, self.cfg.optimizer.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.cfg.dmd.cm_reg.enabled and self.cfg.dmd.cm_reg.use_ema:
                self._cmreg_update_ema()  # update the EMA shadow after the optimizer step
            gen_loss_val = float(dmd_loss.item())
            gen_grad_val = float(gen_grad.item())

        # ── CRITIC / FAKE SCORE (every step) ────────────────────────────────
        # Final chunk resampled with fresh noise under no_grad, assembled into the full window.
        # window_full_grad uses the same re-noise→denoise process → critic models exactly the
        # generator's current output distribution.
        self.critic_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            if sr_full_grad:
                x0_fake_c = _regen_window(keep_grad=False)
            else:
                x0_fake_c = _win(run_generator(transformer, _noise(), gen_cond, sigma_list, grad_step=None))
        critic_sigma = self._dmd_sample_inner_sigma(dtype, latent_frames=win_latents)
        critic_loss, critic_log = compute_critic_loss(
            score_model,
            critic_lora,
            x0_fake=x0_fake_c,
            sigma=critic_sigma,
            cond=window_cond,
        )
        critic_loss.backward()

        clip_params = list(critic_lora.get_trainable_parameters())
        if gan_on:
            assert gan_disc is not None
            _W = int(win_latents)
            _ws = int(window_info["win_start"])
            latent_full = meta["latent_full"]
            if int(latent_full.shape[2]) >= _ws + _W:
                x0_real_win = latent_full[:, :, _ws:_ws + _W].to(
                    device=x0_fake_c.device, dtype=x0_fake_c.dtype
                )
                gan_d_sigma = self._dmd_sample_inner_sigma(dtype, latent_frames=_W)
                gan_d_loss, gan_d_log = compute_gan_critic_loss(
                    score_model,
                    critic_lora,
                    gan_disc,
                    x0_fake=x0_fake_c,
                    x0_real=x0_real_win,
                    sigma=gan_d_sigma,
                    cond=window_cond,
                    K=_W,
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
                clip_params = clip_params + gan_disc.trainable_parameters()
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
            "self_rollout_depth": int(window_info["depth"]),
            "self_rollout_window": int(window_info.get("window_chunks", 1)),
            "self_rollout_spatial_dropped": int(window_info.get("spatial_dropped", 0)),
            "bank_mode": window_info.get("bank_mode", "off"),
            "dmd_loss": gen_loss_val,
            "dmd_grad": gen_grad_val,
            "cm_loss": cm_loss_val,
            "seam_loss": seam_loss_val,
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
