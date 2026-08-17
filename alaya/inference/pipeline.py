"""FlashAlaya: standalone streaming inference pipeline for the alaya world model.

FlashDreams-style execution contract on top of the self-contained
InferenceEngine (no alaya.trainer dependency):

    engine = InferenceEngine(cfg); engine.setup()
    pipe   = FlashAlayaPipeline(engine)
    cache  = pipe.initialize_cache(video_pixels, caption, metadata, rounds=5)
    for i in range(rounds):
        pred = pipe.generate(i, cache)     # hot path: conditions -> 4-step denoise
        pipe.finalize(i, cache, pred)      # off hot path: bank append (decode+DA3), history roll
    frames = pipe.decode(cache)            # latent prefix + preds -> pixel frames

Denoise math/conditioning replicates the trainer's validation rollout
(_validate_rollout_sample) for the inference paths: no-CFG / CFG / action-CFG.
STG perturbations are not carried over (the few-step student runs without
them); set validation.stg_blocks=[] in the config.

Scope: spatial conditioning is the bank path only (camera required); without a
camera the spatial stream is simply absent.

Optimization hooks (planned, marked with `# OPT:`):
  - CUDA-graph / torch.compile of the DiT forward    -> run.py COMPILE env
  - lightweight VAE (taehv) for the bank decode      -> spatial.SpatialMemory
  - streaming pixel decode for interactive use       -> docs/streaming_decoder_plan.md
"""
from __future__ import annotations

import os
import time
from typing import Any

import torch

import alaya.inference.conditioning as C
from alaya.inference.cache import RolloutCache
from alaya.inference.engine import InferenceEngine


class FlashAlayaPipeline:
    def __init__(
        self,
        engine: InferenceEngine,
        *,
        control_modes: list[str] | None = None,
        use_memory: bool = True,
        action_cfg_scale: float = 1.0,
        flex_attn: bool = False,
        seed: int | None = None,
        ttc: bool = False,
        ttc_levels: tuple[int, ...] = (500, 250),
        ttc_strength: float = 1.0,
        ttc_ref_action: bool = True,
    ) -> None:
        """
        seed: if set, the initial noise of chunk i uses generator seed
            (seed*1000 + i) — makes runs comparable across settings.
        ttc: enable Pathwise Test-Time Correction (arXiv:2602.05871). At the
            correction noise levels it re-anchors the per-chunk denoise to the
            initial frame (the sink) to curb appearance drift over long rollouts.
            OFF by default; when off the denoise path is byte-for-byte the
            baseline (no extra RNG draws, no extra forwards).
        ttc_levels: the denoise target noise levels (on the 0..1000 scale, i.e.
            sigma*1000) at which the reference correction is injected. Paper
            default {500, 250} = sigma {0.5, 0.25}.
        """
        self.e = engine
        self.cfg = engine.cfg
        self.flex_attn = bool(flex_attn)  # exact: masked self-attn via fused flex_attention (use with COMPILE)
        self.seed = seed
        self.ttc = bool(ttc)
        self.ttc_levels = frozenset(int(round(x)) for x in ttc_levels)
        self.ttc_strength = float(ttc_strength)
        # if False, the TTC reference pass drops the camera action so it anchors a
        # clean initial-frame APPEARANCE (style) instead of a forward-extrapolated
        # (and thus equally-drifted) target; geometry is restored by the resume step.
        self.ttc_ref_action = bool(ttc_ref_action)
        if control_modes is None:
            modes = getattr(engine.cfg.validation, "modes", {}) or {}
            first = next(iter(modes.values()), None)
            control_modes = list(first.control) if first is not None else ["action"]
        self.control_modes = control_modes
        self.use_memory = bool(use_memory)
        self.action_cfg_scale = float(action_cfg_scale)
        if list(getattr(self.cfg.validation, "stg_blocks", []) or []):
            raise NotImplementedError("FlashAlaya does not carry over STG perturbations; set stg_blocks=[]")
        self.cfg_scale = float(self.cfg.validation.cfg_scale)
        self.timings: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ utils
    def _now(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _t(self, name: str, t0: float) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings.setdefault(name, []).append(time.perf_counter() - t0)

    # --------------------------------------------------------- initialize_cache
    @torch.no_grad()
    def initialize_cache(
        self,
        video_pixels: torch.Tensor,
        caption: str,
        metadata: dict[str, Any],
        *,
        rounds: int,
        K: int | None = None,
        cond_end: int | None = None,
        needed_latents: int | None = None,
    ) -> RolloutCache:
        """One-shot conditioning: encode video + text, slice layout, init spatial bank."""
        e, cfg = self.e, self.cfg
        K = int(K if K is not None else max(cfg.layout.output.latent_frames))
        N = int(cfg.layout.history_latent_frames)
        gap_steps = int(float(cfg.layout.max_gap_sec or 0.0) * cfg.sample.fps / cfg.sample.temporal_stride)
        if cond_end is None:
            cond_end = 1 if cfg.layout.condition.i2v_prob > 0 else 0
        cond_end = int(cond_end)
        explicit_condition = cond_end if N == 0 else 0
        if needed_latents is None:
            needed_latents = cfg.layout.sink_latent_frames + gap_steps + N + explicit_condition + rounds * K + 1

        t0 = self._now()
        latent_full = e.encode_video(video_pixels, needed_latents=needed_latents)
        self._t("init/vae_encode", t0)
        t0 = self._now()
        # caption is encoded as given; dataset-specific caption prefixes are the
        # caller's responsibility (the demo run.py applies them).
        context = e.encode_caption(caption)
        negative_context = (
            e.encode_caption(cfg.validation.negative_prompt) if self.cfg_scale > 1.0 else None
        )
        self._t("init/text_encode", t0)

        B, _, _, H_lat, W_lat = latent_full.shape
        sink_count = cfg.layout.sink_latent_frames
        sink_latent = latent_full[:, :, :sink_count].contiguous() if sink_count > 0 else None
        hist_start = sink_count + gap_steps
        hist_end = hist_start + N
        condition_start = hist_end
        target_base_start = condition_start + explicit_condition
        history = latent_full[:, :, hist_start:hist_end].clone().contiguous() if N > 0 else None
        history_action_t_indices = (
            torch.arange(hist_start, hist_end, device=e.device, dtype=torch.float32)
            if bool(getattr(cfg.control, "action_history_memory", False)) and N > 0
            else None
        )
        explicit_nearby = (
            latent_full[:, :, condition_start:target_base_start].contiguous() if explicit_condition > 0 else None
        )
        sink_indices = (
            C.indices_grid(B, sink_count, H_lat, W_lat, t_offset=C.sink_t_offset(N), device=e.device)
            if sink_count > 0
            else None
        )
        sigmas = C.denoise_sigmas(
            steps=int(cfg.validation.sampling_steps),
            scheduler=str(cfg.validation.scheduler),
            device=e.device,
        )

        t0 = self._now()
        spatial_bank = e.spatial.init_bank(
            video_pixels=video_pixels, metadata=metadata, target_start=target_base_start
        )
        self._t("init/spatial_bank", t0)

        return RolloutCache(
            latent_full=latent_full,
            video_pixels=video_pixels,
            metadata=metadata,
            context=context,
            negative_context=negative_context,
            sink_latent=sink_latent,
            sink_indices=sink_indices,
            sigmas=sigmas,
            K=K,
            N=N,
            cond_end=cond_end,
            gap_steps=gap_steps,
            explicit_condition=explicit_condition,
            target_base_start=target_base_start,
            history=history,
            history_action_t_indices=history_action_t_indices,
            explicit_nearby=explicit_nearby,
            spatial_bank=spatial_bank,
        )

    # ------------------------------------------------------------------ generate
    @torch.no_grad()
    def generate(self, ar_index: int, cache: RolloutCache) -> torch.Tensor:
        """Hot path: build per-step conditions and run the few-step denoise.

        Returns the clean latent chunk [B,C,K,H,W]. Does NOT mutate cache —
        call finalize() to commit it into history / spatial bank.
        """
        e, cfg, c = self.e, self.cfg, cache
        B = c.latent_full.shape[0]
        K, N, cond_end = c.K, c.N, c.cond_end
        current_target_start = c.target_start(ar_index)
        target_rope_t_indices = C.target_t_indices(
            K,
            history_latent_frames=N,
            condition_latent_frames=cond_end,
            gap_steps=c.gap_steps,
            device=e.device,
        )
        target_action_t_indices = torch.arange(
            current_target_start, current_target_start + K, device=e.device, dtype=torch.float32
        )

        # -- memory tokens --
        t0 = self._now()
        mem_tokens = mem_indices = None
        if c.history is not None:
            mem_tokens, mem_indices = e.history_encoder(c.history)
            mem_indices = mem_indices.clone()
            mem_indices[:, 0, :, :] += C.memory_t_offset(N, cond_end)
            if not self.use_memory:
                mem_tokens = mem_tokens * 0.0
        self._t("gen/memory", t0)

        # -- nearby condition --
        if cond_end <= 0:
            nearby_latent = None
        elif c.history is not None:
            nearby_latent = c.history[:, :, -cond_end:].contiguous()
        elif ar_index == 0:
            nearby_latent = c.explicit_nearby
        else:
            nearby_latent = c.preds[-1][:, :, -cond_end:].to(c.latent_full.dtype).contiguous()
        nearby_indices = (
            C.indices_grid(
                B, cond_end, c.H_lat, c.W_lat,
                t_offset=C.nearby_t_offset(N, cond_end, gap_steps=c.gap_steps),
                device=e.device,
            )
            if cond_end > 0
            else None
        )

        # -- spatial (geometric) condition: bank path only --
        t0 = self._now()
        spatial_context = None
        if c.spatial_bank is not None:
            spatial_context = e.spatial.build_context(
                bank=c.spatial_bank,
                metadata=c.metadata,
                target_start=current_target_start,
                K=K,
                target_rope_t_indices=target_rope_t_indices,
            )
        spatial_latent = spatial_context["latent"] if spatial_context is not None else None
        spatial_mask_patch = spatial_context.get("mask_patch") if spatial_context is not None else None
        spatial_indices = (
            C.indices_grid_for_t_indices(
                B,
                spatial_context.get("rope_t_indices", spatial_context["target_indices"]),
                c.H_lat,
                c.W_lat,
                device=e.device,
            )
            if spatial_context is not None
            else None
        )
        self._t("gen/spatial_ctx", t0)

        # -- control (action) --
        control_kwargs = C.build_control_kwargs(
            metadata=c.metadata,
            control_modes=list(self.control_modes),
            target_t_indices=target_action_t_indices,
            condition_t_indices=(
                torch.arange(
                    current_target_start - cond_end, current_target_start,
                    device=e.device, dtype=torch.float32,
                )
                if cond_end > 0
                else None
            ),
            history_t_indices=(
                c.history_action_t_indices if mem_tokens is not None and self.use_memory else None
            ),
            action_scale=cfg.control.action_scale,
            temporal_stride=int(cfg.sample.temporal_stride),
            action_history_memory=bool(getattr(cfg.control, "action_history_memory", False)),
            device=e.device,
            dtype=e.dtype,
        )

        # -- few-step denoise --
        t0 = self._now()
        pred = self._denoise_chunk(
            ar_index=ar_index,
            cache=c,
            mem_tokens=mem_tokens,
            mem_indices=mem_indices,
            nearby_latent=nearby_latent,
            nearby_indices=nearby_indices,
            spatial_latent=spatial_latent,
            spatial_mask_patch=spatial_mask_patch,
            spatial_indices=spatial_indices,
            control_kwargs=control_kwargs,
            target_rope_t_indices=target_rope_t_indices,
        )
        self._t("gen/denoise", t0)
        return pred

    def _denoise_chunk(
        self,
        *,
        ar_index: int,
        cache: RolloutCache,
        mem_tokens,
        mem_indices,
        nearby_latent,
        nearby_indices,
        spatial_latent,
        spatial_mask_patch,
        spatial_indices,
        control_kwargs,
        target_rope_t_indices,
    ) -> torch.Tensor:
        """Few-step flow-matching denoise of one K-frame chunk."""
        e, cfg, c = self.e, self.cfg, cache
        B = c.latent_full.shape[0]
        K = c.K
        sigmas = c.sigmas

        if self.seed is not None:
            gen = torch.Generator(device=e.device).manual_seed(int(self.seed) * 1000 + int(ar_index))
            x_t = torch.randn(
                B, c.latent_full.shape[1], K, c.H_lat, c.W_lat,
                device=e.device, dtype=e.dtype, generator=gen,
            )
        else:
            gen = None
            x_t = torch.randn(B, c.latent_full.shape[1], K, c.H_lat, c.W_lat, device=e.device, dtype=e.dtype)

        def _forward_velocity(x_in, sigma, context_tensor, *, control, ref):
            # ref=True is the TTC reference pass: condition ONLY on the sink
            # (initial frame) and drop the evolving memory/nearby/spatial context.
            # Under compile=reduce-overhead each invocation replays a CUDA graph;
            # mark a new step so outputs of the previous call are not treated as
            # still-live graph buffers (they get cloned instead of overwritten).
            torch.compiler.cudagraph_mark_step_begin()
            return e.transformer(
                x=[x_in.squeeze(0)],
                t=(sigma * 1000.0).view(1).to(device=e.device, dtype=x_in.dtype),
                context=[context_tensor],
                seq_len=K * c.H_lat * c.W_lat,
                fps=cfg.sample.fps,
                perturbations=None,
                history_kv_tokens=None if ref else mem_tokens,
                history_indices_grid=None if ref else mem_indices,
                gen_t_indices_override=target_rope_t_indices,
                sink_latent=c.sink_latent,
                sink_indices_grid=c.sink_indices,
                spatial_latent=None if ref else spatial_latent,
                spatial_mask_patch=None if ref else spatial_mask_patch,
                spatial_indices_grid=None if ref else spatial_indices,
                nearby_latent=None if ref else nearby_latent,
                nearby_indices_grid=None if ref else nearby_indices,
                flex_masked_attention=self.flex_attn,
                **control,
            )

        def _compose_velocity(x_in, sigma, *, ref=False):
            """Full few-step velocity (pos + CFG + action-CFG + rescale)."""
            # style-anchored reference: drop the camera action so the anchor is the
            # clean initial-frame appearance, not a forward-extrapolated target.
            ctl = {} if (ref and not self.ttc_ref_action) else control_kwargs
            pos_v = _forward_velocity(x_in, sigma, c.context, control=ctl, ref=ref)
            pred_v = pos_v

            if self.cfg_scale > 1.0 and c.negative_context is not None:
                neg_v = _forward_velocity(x_in, sigma, c.negative_context, control=ctl, ref=ref)
                pred_v = pred_v + (self.cfg_scale - 1.0) * (pos_v - neg_v)

            if self.action_cfg_scale > 1.0 and "action_vectors" in ctl:
                no_action: dict[str, torch.Tensor] = {}
                na_pos = _forward_velocity(x_in, sigma, c.context, control=no_action, ref=ref)
                na_v = na_pos
                if self.cfg_scale > 1.0 and c.negative_context is not None:
                    na_neg = _forward_velocity(x_in, sigma, c.negative_context, control=no_action, ref=ref)
                    na_v = na_v + (self.cfg_scale - 1.0) * (na_pos - na_neg)
                pred_v = na_v + self.action_cfg_scale * (pred_v - na_v)

            if cfg.validation.rescale_scale > 0.0 and pred_v is not pos_v:
                factor = pos_v.float().std() / (pred_v.float().std() + 1e-8)
                factor = float(cfg.validation.rescale_scale) * factor + (1.0 - float(cfg.validation.rescale_scale))
                pred_v = pred_v * factor
            return pred_v

        def _renoise(x0, sigma):
            # Flow-matching forward process Ψ on the linear path:
            # x_sigma = (1 - sigma) * x0 + sigma * eps,  eps ~ N(0, I) fresh.
            shape = (B, c.latent_full.shape[1], K, c.H_lat, c.W_lat)
            eps = torch.randn(shape, device=e.device, dtype=e.dtype, generator=gen)
            s = sigma.to(dtype=x0.dtype)
            return (1.0 - s) * x0 + s * eps

        n_steps = len(sigmas) - 1
        _dbg_steps = os.environ.get("FLASH_DEBUG_STEPS") == "1"
        # TTC correction steps: transitions whose TARGET level (sigma_next*1000)
        # is a requested correction level. Empty when --ttc is off, so the loop
        # below is byte-for-byte the baseline (no extra RNG / forwards).
        ttc_steps = (
            frozenset(
                step
                for step in range(n_steps)
                if int(round(sigmas[step + 1].item() * 1000.0)) in self.ttc_levels
            )
            if self.ttc
            else frozenset()
        )

        for step in range(n_steps):
            if _dbg_steps:
                torch.cuda.synchronize()
                _dbg_t0 = time.perf_counter()
            sigma_now = sigmas[step]
            sigma_next = sigmas[step + 1]

            if _dbg_steps and step == 0:
                def _sig(v):
                    if v is None:
                        return "None"
                    if isinstance(v, torch.Tensor):
                        return f"{tuple(v.shape)}/s{tuple(v.stride())}/c{int(v.is_contiguous())}"
                    return type(v).__name__
                sig = {
                    "x": _sig(x_t), "mem": _sig(mem_tokens), "mem_idx": _sig(mem_indices),
                    "sink": _sig(c.sink_latent), "sink_idx": _sig(c.sink_indices),
                    "sp": _sig(spatial_latent), "sp_mask": _sig(spatial_mask_patch), "sp_idx": _sig(spatial_indices),
                    "nb": _sig(nearby_latent), "nb_idx": _sig(nearby_indices),
                    "rope_t": _sig(target_rope_t_indices),
                    **{f"ctl:{k}": _sig(v) for k, v in control_kwargs.items()},
                }
                print(f"[SIG ar={ar_index}] " + " | ".join(f"{k}={v}" for k, v in sig.items()), flush=True)

            pred_v = _compose_velocity(x_t, sigma_now, ref=False)

            if step in ttc_steps:
                # Pathwise Test-Time Correction (arXiv:2602.05871), Eq. 8-10.
                # Eq.8 clean estimate with evolving context S_t at level sigma_now.
                x0_hat = (x_t.float() - pred_v.float() * sigma_now.float()).to(x_t.dtype)
                # Eq.9 re-noise to sigma_next, denoise with reference S_0 (sink only).
                x_ref_in = _renoise(x0_hat, sigma_next)
                v_ref = _compose_velocity(x_ref_in, sigma_next, ref=True)
                x0_corr = (x_ref_in.float() - v_ref.float() * sigma_next.float()).to(x_t.dtype)
                if _dbg_steps:
                    _rel = (x0_corr.float() - x0_hat.float()).norm() / (x0_hat.float().norm() + 1e-8)
                    print(f"[TTC ar={ar_index} step{step} sigma->{sigma_next.item():.2f}] "
                          f"|corr-x0|/|x0|={_rel.item():.4f}", flush=True)
                # ttc_strength>1 over-corrects along the (corrected - evolving)
                # direction = pull harder toward the initial-frame anchor; ==1.0 is
                # the plain TTC (x0_eff == x0_corr); 0 disables the correction.
                if self.ttc_strength != 1.0:
                    x0_corr = (x0_hat.float() + self.ttc_strength * (x0_corr.float() - x0_hat.float())).to(x_t.dtype)
                # Eq.10 re-noise the corrected estimate to sigma_next and resume the
                # normal trajectory (next iteration denoises it with S_t).
                x_t = _renoise(x0_corr, sigma_next)
            elif sigma_next.item() > 1e-5:
                dt = (sigma_now - sigma_next).to(dtype=x_t.dtype)
                x_t = x_t - dt * pred_v
            else:
                x_t = (x_t.float() - pred_v.float() * sigma_now.float()).to(x_t.dtype)
            if _dbg_steps:
                torch.cuda.synchronize()
                print(f"[FlashAlaya:step] step {step}: {time.perf_counter() - _dbg_t0:.3f}s", flush=True)

        return x_t

    # ------------------------------------------------------------------ finalize
    @torch.no_grad()
    def finalize(self, ar_index: int, cache: RolloutCache, pred: torch.Tensor) -> None:
        """Commit a generated chunk: spatial bank append (decode + DA3 depth) and
        history sliding-window roll."""
        e, c = self.e, cache
        pred = pred.detach()
        c.preds.append(pred)
        current_target_start = c.target_start(ar_index)

        t0 = self._now()
        if c.spatial_bank is not None:
            e.spatial.append_prediction(
                bank=c.spatial_bank,
                pred_latent=pred,
                metadata=c.metadata,
                target_start=current_target_start,
            )
        self._t("fin/bank_append", t0)

        if c.history is not None:
            c.history = torch.cat([c.history, pred.to(c.history.dtype)], dim=2)[:, :, -c.N:].contiguous()
            if c.history.shape[2] != c.N:
                raise RuntimeError(f"history length changed: {c.history.shape[2]} != {c.N}")
            if c.history_action_t_indices is not None:
                target_action_t = torch.arange(
                    current_target_start, current_target_start + c.K,
                    device=e.device, dtype=torch.float32,
                )
                c.history_action_t_indices = torch.cat(
                    [c.history_action_t_indices, target_action_t], dim=0
                )[-c.N:].contiguous()

    # -------------------------------------------------------------------- decode
    @torch.no_grad()
    def decode(self, cache: RolloutCache, *, keep_history_prefix: bool = False) -> torch.Tensor:
        """Decode the rollout to uint8 pixel frames [T,H,W,C].

        The real history prefix (video_history_latent_frames latents) is always
        prepended BEFORE decoding so the first generated frames get correct VAE
        left-context (seamless). By default it is then STRIPPED from the output so
        the saved video starts at the first generated frame (matches the trainer's
        `final/` pred). Set keep_history_prefix=True to keep the real history as a
        visible lead-in."""
        e, cfg, c = self.e, self.cfg, cache
        parts: list[torch.Tensor] = []
        prefix_pixel_len = 0
        if c.N > 0:
            keep = cfg.validation.video_history_latent_frames or c.N
            keep = max(1, min(int(keep), c.N))
            hist_end = cfg.layout.sink_latent_frames + c.gap_steps + c.N
            parts.append(c.latent_full[:, :, hist_end - keep:hist_end].contiguous())
            prefix_pixel_len = (keep - 1) * int(cfg.sample.temporal_stride) + 1
        parts.extend(p.to(c.latent_full.dtype) for p in c.preds)
        pred_latent = torch.cat(parts, dim=2).contiguous()
        t0 = self._now()
        frames = e.decode_latent_to_video_frames(pred_latent)
        self._t("decode/vae", t0)
        if not keep_history_prefix and 0 < prefix_pixel_len < int(frames.shape[0]):
            frames = frames[prefix_pixel_len:]   # drop the real-history lead-in (pure generation)
        return frames

    # ------------------------------------------------------------------- report
    def print_timings(self) -> None:
        print("\n========== FlashAlaya pipeline timings ==========", flush=True)
        for name, vals in self.timings.items():
            print(
                f"  {name:<22}: n={len(vals):>3}  sum={sum(vals):7.2f}s  mean={sum(vals)/len(vals)*1000:7.1f}ms",
                flush=True,
            )
        print("==================================================", flush=True)
