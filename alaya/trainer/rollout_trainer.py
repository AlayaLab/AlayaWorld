from __future__ import annotations

import json
import math
import os
import random
import time
import gc
import ctypes
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ltx2.modules.patchifier import VideoLatentPatchifier, VideoLatentShape
from ltx2.modules.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx2.modules.scheduler import LTX2Scheduler, LinearQuadraticScheduler

from alaya.checkpoint import load_checkpoint_weights, save_checkpoint
from alaya.config.schema import TrainConfig, ValidationModeConfig
from alaya.control.action import (
    build_action_vectors,
    build_action_vectors_from_pixel_indices,
)
from alaya.data import build_train_dataloader, build_validation_dataset
from alaya.memory.builder import build_history_encoder
from alaya.memory.da3_depth import DA3DepthEstimator
from alaya.memory.vigeo_geometry import ViGeoGeometryEstimator
from alaya.memory.spatial_cache import (
    Sparse3DCache,
    build_retrieved_latent_context,
    forward_warp_indexed_pixel_sources_to_pixel_targets,
    forward_warp_pixel_sources_to_pixel_targets,
    forward_warp_video_to_targets,
    pixel_intrinsics,
    render_colored_pointmaps_to_camera_targets,
)
from alaya.model import build_model_components
from alaya.model.fsdp import maybe_wrap_fsdp
from alaya.utils.distributed import broadcast_tensor, init_distributed, rank0_print
from alaya.utils.dtype import resolve_dtype
from alaya.utils.seed import seed_everything
from fastvideo.dataset.t2v_datasets import MultiSourceVideoDataset, _strip_camera_blocks
from alaya.data.wbench_shard import shard_indices_with_padding
from fastvideo.rollout.drift_simulator import corrupt_history_latents_helios, get_drift_counters
from fastvideo.rollout.error_bank import ErrorBank


@dataclass
class _RolloutSpatialBank:
    pixels: list[torch.Tensor]
    frame_indices: list[int]
    depths: list[torch.Tensor | None]
    vigeo_pointmaps: list[torch.Tensor] = field(default_factory=list)
    vigeo_valid_masks: list[torch.Tensor] = field(default_factory=list)
    vigeo_predicted_poses: list[torch.Tensor] = field(default_factory=list)
    vigeo_intrinsics: list[torch.Tensor] = field(default_factory=list)
    vigeo_kv_caches: Any = None
    vigeo_scale: float = 1.0
    vigeo_pairwise_scales: tuple[float, ...] = ()
    vigeo_generated_chunks: int = 0
    vigeo_pixel_offset: int = 0
    causal_prefix_pixel: torch.Tensor | None = None
    causal_prefix_frame_index: int | None = None
    subject_anchor_pixel: torch.Tensor | None = None
    subject_anchor_mask: torch.Tensor | None = None
    subject_anchor_paste: bool = True  # refresh the subject anchor on every ingest
    subject_exclusion_mask: torch.Tensor | None = None


class RolloutTrainer:
    """Readable rollout trainer for sink + memory + nearby condition + output."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.dist = init_distributed()
        seed_everything(cfg.run.seed + self.dist.rank)
        self.dtype = resolve_dtype(cfg.runtime.dtype)
        self.global_step = 0

        self.components = None
        self.history_encoder = None
        self.optimizer = None
        self.scheduler = None
        self.dataloader = None
        self.error_bank = None
        self.validation_datasets = {}
        self.action_param_ids: set[int] = set()
        self.action_param_count = 0
        self.da3_depth: DA3DepthEstimator | None = None
        self.vigeo_geometry: ViGeoGeometryEstimator | None = None
        self._text_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._text_cache_hits = 0
        self._text_cache_misses = 0
        self._perf_text_s = 0.0
        self._perf_vae_s = 0.0
        self._perf_calls = 0
        self._vae_cache_hits = 0
        self._vae_cache_misses = 0

    def describe(self) -> None:
        max_k = max(self._active_output_latent_frames())
        max_gap = int(self.cfg.layout.max_gap_sec * self.cfg.sample.fps / self.cfg.sample.temporal_stride)
        next_extra = max_k if self.cfg.next_forcing.enabled else 0
        if self._uses_vigeo_prefix_last_frame():
            target_latents = max_k + next_extra
            target_pixels = target_latents * int(self.cfg.sample.temporal_stride)
            scale_pixels = self._vigeo_prefix_pixel_frames()
            target_prefix_pixels = self._vigeo_target_prefix_pixel_frames()
            pixel_frames = target_prefix_pixels + target_pixels
            rank0_print(
                self.dist,
                "[Describe]",
                f"vigeo_scale_pixels={scale_pixels} "
                f"target_prefix_pixels={target_prefix_pixels} "
                f"sink=random-history-frame nearby=last-{self._vigeo_motion_pixel_frames()}-pixel-motion "
                f"history={self.cfg.layout.history_latent_frames} "
                f"retrieval_views={self.cfg.spatial_memory.num_context_frames} "
                f"handoff={self.cfg.validation.vigeo_handoff_mode} "
                f"pred_decode={self.cfg.validation.vigeo_pred_decode_mode} "
                f"target={max_k} next_extra={next_extra} required_pixel_frames={pixel_frames}",
            )
            return
        _sr = self.cfg.dmd.self_rollout if self.cfg.dmd.enabled else None
        gt_extra = (
            int(_sr.max_chunks) * max_k
            if _sr is not None and _sr.enabled and bool(getattr(_sr, "score_gt_context", False))
            else 0
        )
        if self._uses_fixed_no_history_condition_window():
            max_condition = self._max_explicit_condition_latents()
            total_latents = self.cfg.layout.sink_latent_frames + max_gap + max_k + 1 + next_extra + gt_extra
        else:
            max_condition = self._max_explicit_condition_latents()
            total_latents = (
                self.cfg.layout.sink_latent_frames
                + max_gap
                + self.cfg.layout.history_latent_frames
                + max_condition
                + max_k
                + next_extra
                + gt_extra
            )
        pixel_frames = (total_latents - 1) * self.cfg.sample.temporal_stride + 1
        rank0_print(
            self.dist,
            "[Describe]",
            f"sink={self.cfg.layout.sink_latent_frames} gap<= {max_gap} "
            f"history={self.cfg.layout.history_latent_frames} explicit_condition={max_condition} "
            f"target<= {max_k} next_extra={next_extra} total_latents={total_latents} "
            f"required_pixel_frames={pixel_frames}",
        )

    @staticmethod
    def _release_cpu_load_memory() -> None:
        gc.collect()
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError):
            pass

    def _run_rank_serialized_load(self, label: str, load_fn):
        """Optionally serialize host-heavy checkpoint loading across local ranks."""
        enabled = os.environ.get("ALAYA_SERIAL_MODEL_LOAD", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled or not dist.is_initialized() or dist.get_world_size() <= 1:
            return load_fn()

        result = None
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for load_rank in range(world_size):
            if rank == load_rank:
                print(
                    f"[Setup] rank {rank}/{world_size} loading {label} serially",
                    flush=True,
                )
                result = load_fn()
                self._release_cpu_load_memory()
                print(
                    f"[Setup] rank {rank}/{world_size} finished {label}",
                    flush=True,
                )
            dist.barrier()
        if result is None:
            raise RuntimeError(f"rank {rank} did not execute serialized load for {label}")
        return result

    def setup(self) -> None:
        Path(self.cfg.run.output_dir).mkdir(parents=True, exist_ok=True)
        self.components = self._run_rank_serialized_load(
            "model components",
            lambda: build_model_components(self.cfg, self.dist.device, self.dtype),
        )

        patchify_proj = self.components.transformer.patchify_proj
        if self.cfg.layout.history_latent_frames > 0:
            self.history_encoder = build_history_encoder(
                self.cfg.memory,
                in_channels=patchify_proj.in_features,
                out_channels=patchify_proj.out_features,
                device=self.dist.device,
                dtype=self.dtype,
                checkpoint_path=self.cfg.paths.history_encoder,
            )
            if self.cfg.memory.use_lr_branch:
                self.history_encoder.setup_lr_proj_from_patchify(patchify_proj)
        else:
            self.history_encoder = None
            rank0_print(self.dist, "[Memory]", "disabled because layout.history_latent_frames=0")

        self.global_step = self._run_rank_serialized_load(
            "resume checkpoint",
            lambda: load_checkpoint_weights(
                cfg=self.cfg,
                dist_state=self.dist,
                transformer=self.components.transformer,
                history_encoder=self.history_encoder,
                lora_manager=self.components.lora_manager,
                critic_lora=self.components.critic_lora,
                score_model=self.components.score_model,
                gan_discriminator=self.components.gan_discriminator,
                next_forcing_head=self.components.next_forcing_head,
            ),
        )

        self._broadcast_module_from_rank0(self.components.next_forcing_head)
        self._broadcast_module_from_rank0(self.components.gan_discriminator)

        self.action_param_ids = {
            id(param)
            for name, param in self.components.transformer.named_parameters()
            if _is_action_adaln_param(name)
        }
        self.action_param_count = sum(
            param.numel()
            for name, param in self.components.transformer.named_parameters()
            if _is_action_adaln_param(name)
        )
        self.components.transformer = maybe_wrap_fsdp(
            self.components.transformer,
            self.cfg.runtime,
            self.dist.device,
        )
        if (
            self.cfg.dmd.enabled
            and bool(getattr(self.cfg.dmd, "shard_score_model", False))
            and self.components.score_model is not None
        ):
            self.components.score_model = maybe_wrap_fsdp(
                self.components.score_model,
                self.cfg.runtime,
                self.dist.device,
            )
            rank0_print(self.dist, "[DMD]", "score_model FSDP sharding enabled")

        trainable, param_groups = self._optimizer_parameters()
        if self.components.lora_manager is not None:
            lora_params = self.components.lora_manager.get_trainable_parameters()
            trainable += lora_params
            if lora_params:
                if param_groups:
                    param_groups[0]["params"].extend(lora_params)
                else:
                    param_groups.append({"params": lora_params, "lr": self.cfg.optimizer.lr})
        if self.components.next_forcing_head is not None:
            nf_params = self.components.next_forcing_head.trainable_parameters()
            trainable += nf_params
            if nf_params:
                if param_groups:
                    param_groups[0]["params"].extend(nf_params)
                else:
                    param_groups.append({"params": nf_params, "lr": self.cfg.optimizer.lr})
        if not trainable:
            raise RuntimeError("no trainable parameters; enable memory.train or lora.train")

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        scheduler_start = self.global_step - 1 if self.global_step > 0 else -1
        if scheduler_start >= 0:
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, float(step + 1) / max(1, self.cfg.optimizer.warmup_steps)),
            last_epoch=scheduler_start,
        )
        self.dataloader = build_train_dataloader(self.cfg, self.dist)
        self.error_bank = self._build_error_bank()
        if getattr(self.cfg.runtime, "precache_text_embeds", False) and self.cfg.runtime.text_embed_cache_dir:
            from alaya.data.text_embed_cache import precache_text_embeds
            _ds = self.dataloader.dataset
            _subsets = [_ds] if hasattr(_ds, "samples") else list(getattr(_ds, "datasets", []))
            for _sub in _subsets:
                precache_text_embeds(self.cfg, self.components, self.dist, _sub)

    def train(self) -> None:
        self.setup()
        assert self.dataloader is not None

        if self.cfg.validation.enabled and self.cfg.validation.before_train:
            self.validate(self.global_step)
        if self.cfg.optimizer.max_steps is not None and self.global_step >= self.cfg.optimizer.max_steps:
            return

        for epoch in range(self.cfg.optimizer.epochs):
            if hasattr(self.dataloader, "set_epoch"):
                self.dataloader.set_epoch(epoch)
            else:
                dataset = getattr(self.dataloader, "dataset", None)
                sampler = getattr(self.dataloader, "sampler", None)
                if hasattr(dataset, "set_epoch"):
                    dataset.set_epoch(epoch)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

            for batch in self.dataloader:
                step_start = time.time()
                loss, grad_norm, info = self.train_one_step(batch)
                self.global_step += 1

                if self.dist.is_main and self.global_step % self.cfg.optimizer.log_steps == 0:
                    lr = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.cfg.optimizer.lr
                    mcp_text = f"mcp={info['mcp_loss']:.4f} " if info.get("mcp_loss") is not None else ""
                    dmd_text = ""
                    if info.get("critic_loss") is not None:
                        dmd_loss = info.get("dmd_loss")
                        dmd_str = f"{dmd_loss:.6f}" if dmd_loss is not None else "-"
                        dmd_grad = info.get("dmd_grad")
                        dmd_grad_str = f"{dmd_grad:.4f}" if dmd_grad is not None else "-"
                        dmd_text = (
                            f"dmd_loss={dmd_str} dmd_grad={dmd_grad_str} "
                            f"critic={info['critic_loss']:.6f} "
                        )
                        if info.get("gan_on"):
                            _gg = info.get("gan_g_loss")
                            _gd = info.get("gan_d_loss")
                            if _gg is not None:
                                dmd_text += f"gan_g={_gg:.4f} "
                            if _gd is not None:
                                dmd_text += f"gan_d={_gd:.4f} "
                            _r1 = info.get("critic/r1_loss")
                            _r2 = info.get("critic/r2_loss")
                            if _r1 is not None:
                                dmd_text += f"r1={_r1:.4f} "
                            if _r2 is not None:
                                dmd_text += f"r2={_r2:.4f} "
                    cm_text = ""
                    if info.get("cm_loss") is not None:
                        cm_text = f"cm={info['cm_loss']:.6f} "
                    seam_text = ""
                    if info.get("seam_loss") is not None:
                        seam_text = f"seam={info['seam_loss']:.6f} "

                    print(
                        f"[Train] step={self.global_step} epoch={epoch} "
                        f"source={info['source']} video={info['video_id']} "
                        f"fs={info['frame_start']} fe={info['frame_end']} "
                        f"K={info['K']} gap={info['gap_steps']} cond={info['cond_mode']}:{info['cond_end']} "
                        f"control={info['control_modes']} "
                        f"spatial={info['spatial_context']} "
                        f"mem_drop={int(info['memory_dropped'])} "
                        f"hist={info['anti_drift']} "
                        f"prefix_fix={int(info.get('condition_prefix_fixed', False))} "
                        + (f"sr_depth={info.get('self_rollout_depth')} " if self.cfg.dmd.self_rollout.enabled else "")
                        + (f"bank={info['bank_mode']} " if info.get("bank_mode") not in (None, "off") else "")
                        + f"{mcp_text}"
                        f"{dmd_text}"
                        f"{cm_text}"
                        f"{seam_text}"
                        f"sigma={info['sigma']:.3f} loss={loss:.6f} grad={grad_norm:.4f} "
                        f"lr={lr:.2e} time={time.time() - step_start:.2f}s",
                        flush=True,
                    )
                    self._perf_calls += 1
                    if self._perf_calls % 10 == 0:
                        _tc_total = self._text_cache_hits + self._text_cache_misses
                        _vc_total = self._vae_cache_hits + self._vae_cache_misses
                        print(
                            f"[Perf] last10: text_encode={self._perf_text_s:.2f}s "
                            f"vae_encode={self._perf_vae_s:.2f}s "
                            f"text_cache_hit={self._text_cache_hits}/{_tc_total} "
                            f"vae_cache_hit={self._vae_cache_hits}/{_vc_total} "
                            f"cache_size={len(self._text_cache)}",
                            flush=True,
                        )
                        self._perf_text_s = 0.0
                        self._perf_vae_s = 0.0
                        self._text_cache_hits = 0
                        self._text_cache_misses = 0
                        self._vae_cache_hits = 0
                        self._vae_cache_misses = 0

                if self.global_step % self.cfg.optimizer.checkpoint_steps == 0:
                    try:
                        self._cleanup_cuda_cache()
                        save_checkpoint(
                            cfg=self.cfg,
                            dist_state=self.dist,
                            step=self.global_step,
                            transformer=self.components.transformer,
                            history_encoder=self.history_encoder,
                            lora_manager=self.components.lora_manager,
                            critic_lora=self.components.critic_lora,
                            gan_discriminator=self.components.gan_discriminator,
                            next_forcing_head=self.components.next_forcing_head,
                        )
                        if dist.is_initialized():
                            dist.barrier()
                    except Exception as exc:
                        print(
                            f"[Rank {self.dist.rank}] [ERROR] Checkpoint save failed at "
                            f"step {self.global_step}: {exc}",
                            flush=True,
                        )
                        traceback.print_exc()

                if self._should_validate(self.global_step):
                    self.validate(self.global_step)

                if self.cfg.optimizer.max_steps is not None and self.global_step >= self.cfg.optimizer.max_steps:
                    return

    def train_one_step(self, batch: Any) -> tuple[float, float, dict[str, Any]]:
        assert self.components is not None
        assert self.optimizer is not None

        if self.cfg.layout.condition.type == "inline":
            return self._train_one_step_inline(batch)

        self.components.transformer.train()
        if self.history_encoder is not None:
            self.history_encoder.train(self.cfg.memory.train)
        self.optimizer.zero_grad(set_to_none=True)

        video_pixels, caption, metadata = self._unpack_batch(batch)
        prompt_caption = self._caption_with_prefix(caption, metadata)
        # Each data-parallel rank sees a different sample. Do not broadcast
        # rank-0 text here, or most ranks train on mismatched video/caption pairs.
        context = self._encode_caption(prompt_caption, sync=False)

        K, gap_steps, cond_mode, cond_end = self._sample_layout(metadata)
        control_modes = self._sample_control_modes()
        vigeo_prefix_mode = self._uses_vigeo_prefix_last_frame()
        sink_prefix_pixel_index: int | None = None
        prefix_pixel_frames = 0
        if vigeo_prefix_mode:
            if cond_mode != "i2v" or cond_end != 1:
                raise ValueError(
                    "vigeo_prefix_last_frame requires an i2v layout with one nearby latent"
                )
            prefix_pixel_frames = self._vigeo_target_prefix_pixel_frames()
            motion_pixel_frames = self._vigeo_motion_pixel_frames()
            target_latent_frames = int(K) * (2 if self.cfg.next_forcing.enabled else 1)
            target_pixel_frames = target_latent_frames * int(self.cfg.sample.temporal_stride)
            required_pixel_frames = prefix_pixel_frames + target_pixel_frames
            available_pixel_frames = self._video_pixel_frame_count(video_pixels)
            if available_pixel_frames < required_pixel_frames:
                raise ValueError(
                    "vigeo prefix training needs "
                    f"{required_pixel_frames} pixel frames (prefix={prefix_pixel_frames}, "
                    f"target_latents={target_latent_frames}), got {available_pixel_frames}"
                )

            continuation_video_pixels = self._slice_video_pixel_frames(
                video_pixels,
                prefix_pixel_frames - motion_pixel_frames,
                prefix_pixel_frames + target_pixel_frames,
            )
            continuation_latent = self._encode_video(
                continuation_video_pixels,
                needed_latents=target_latent_frames + 2,
            )
            if int(continuation_latent.shape[2]) != target_latent_frames + 2:
                raise RuntimeError(
                    "ViGeo continuation VAE length mismatch: "
                    f"got {continuation_latent.shape[2]}, expected {target_latent_frames + 2}"
                )
            sink_prefix_pixel_index = random.randrange(prefix_pixel_frames)
            sink_latent = self._encode_video(
                self._slice_video_pixel_frames(
                    video_pixels,
                    sink_prefix_pixel_index,
                    sink_prefix_pixel_index + 1,
                ),
                needed_latents=1,
            )
            history_latent_frames = int(self.cfg.layout.history_latent_frames)
            if history_latent_frames > 0:
                history_latent = self._encode_video(
                    self._slice_video_pixel_frames(
                        video_pixels,
                        0,
                        prefix_pixel_frames,
                    ),
                    needed_latents=history_latent_frames,
                )
                if int(history_latent.shape[2]) != history_latent_frames:
                    raise RuntimeError(
                        "ViGeo history VAE length mismatch: "
                        f"got {history_latent.shape[2]}, expected {history_latent_frames}"
                    )
            else:
                history_latent = None
            # [anchor, motion, target...]: the DiT sees the temporal motion
            # latent as nearby. The anchor is used only by the chunk-local decoder.
            nearby_latent = continuation_latent[:, :, 1:2].contiguous()
            latent_full = continuation_latent[:, :, 2:].contiguous()
            target_clean = latent_full[:, :, :K].contiguous()
            target_start = 0
            condition_video_pixels = video_pixels
            condition_latent_full = latent_full
            condition_prefix_fixed = False
        else:
            target_start_for_aug = self._segment_target_start(gap_steps=gap_steps, cond_end=cond_end)
            condition_video_pixels, condition_prefix_fixed = self._maybe_fix_condition_prefix_pixels(
                video_pixels,
                target_start=target_start_for_aug,
            )
            latent_full = self._encode_video(video_pixels, metadata=metadata)
            condition_latent_full = (
                self._encode_video(condition_video_pixels)
                if condition_prefix_fixed
                else latent_full
            )
            sink_latent, history_latent, nearby_latent, target_clean, target_start = self._split_segments(
                latent_full,
                K=K,
                gap_steps=gap_steps,
                cond_end=cond_end,
            )
            if condition_prefix_fixed:
                sink_latent, history_latent, nearby_latent, _, _ = self._split_segments(
                    condition_latent_full,
                    K=K,
                    gap_steps=gap_steps,
                    cond_end=cond_end,
                )

        _freeze_skip_drift = bool(condition_prefix_fixed) and bool(
            getattr(self.cfg.anti_drift.drift, "freeze_prefix_skip_drift", False)
        )

        sigma = self._sample_sigma(target_clean.dtype, latent_frames=int(K))
        sigma_view = sigma.view(1, 1, 1, 1, 1)
        mem_tokens = None
        mem_indices = None
        memory_dropped = False
        anti_drift_info = {"anti_drift": "clean", "inject_latent": False, "inject_history": False}
        if history_latent is not None:
            assert self.history_encoder is not None
            history_degrade_mode = (
                None if _freeze_skip_drift else (
                    self._sample_history_degradation_mode()
                    if self._uses_parallel_history_degradation()
                    else None
                )
            )
            history_input, anti_drift_info = self._prepare_history(
                history_latent,
                sigma=float(sigma.item()),
                degradation_mode=history_degrade_mode,
                force_clean=_freeze_skip_drift,
            )
            mem_tokens, mem_indices = self.history_encoder(history_input)
            mem_indices = mem_indices.clone()
            mem_indices[:, 0, :, :] += self._local_memory_t_offset(self.cfg.layout.history_latent_frames, cond_end)
            if history_degrade_mode is not None:
                drop_flag = torch.tensor(
                    [int(history_degrade_mode == "drop")],
                    device=self.dist.device,
                    dtype=torch.long,
                )
            else:
                drop_prob = float(self.cfg.memory.drop_prob)
                if drop_prob > 0.0:
                    if self.dist.is_main:
                        drop_flag = torch.tensor([int(random.random() < drop_prob)], device=self.dist.device, dtype=torch.long)
                    else:
                        drop_flag = torch.zeros(1, device=self.dist.device, dtype=torch.long)
                    broadcast_tensor(drop_flag)
                else:
                    drop_flag = torch.zeros(1, device=self.dist.device, dtype=torch.long)
            if bool(drop_flag.item()):
                # Hard-drop the whole temporal-memory prefix. Keeping zero-valued
                # slots would still consume self-attention probability and let the
                # slots become scratch tokens in later transformer blocks.
                mem_tokens = None
                mem_indices = None
                memory_dropped = True
            if cond_end > 0 and not vigeo_prefix_mode:
                nearby_latent = history_input[:, :, -cond_end:].contiguous()

        B, _, _, H_lat, W_lat = target_clean.shape
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
                    self.cfg.layout.history_latent_frames, cond_end, gap_steps=gap_steps
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
        if vigeo_prefix_mode:
            spatial_context = self._build_vigeo_prefix_last_frame_spatial_context(
                video_pixels=video_pixels,
                metadata=metadata,
                K=K,
                target_rope_t_indices=target_rope_t_indices,
            )
        else:
            spatial_context = self._build_spatial_context(
                latent_full=latent_full,
                video_pixels=video_pixels,
                metadata=metadata,
                target_start=target_start,
                K=K,
                cond_end=cond_end,
                target_rope_t_indices=target_rope_t_indices,
            )
        spatial_latent = spatial_context["latent"] if spatial_context is not None else None
        spatial_mask_patch = spatial_context.get("mask_patch") if spatial_context is not None else None
        _drift_applied = False
        _drift_nearby_applied = False
        _bank_sp_applied = False
        _bank_nb_applied = False
        _step_uses_bank = (
            self.error_bank is not None
            and self.error_bank.is_warm()
            and random.random() >= self.cfg.anti_drift.error_bank.bank_skip_prob
            and not _freeze_skip_drift  # a frozen static prefix also skips error-bank injection
        )
        _bank_flags = (
            self.error_bank.roll_inject_flags(ignore_clean=True)
            if _step_uses_bank else None
        )
        if _step_uses_bank:
            # ErrorBank inject on spatial (K-matched single sample)
            if (
                _bank_flags is not None and bool(_bank_flags.get('spatial', False))
                and spatial_latent is not None
            ):
                _delta_sp = self.error_bank.sample_one(
                    K=int(spatial_latent.shape[2]),
                    timestep=float(sigma.item()),
                    target_shape=tuple(spatial_latent.shape),
                    device=spatial_latent.device,
                    dtype=spatial_latent.dtype,
                )
                if _delta_sp is not None:
                    spatial_latent = spatial_latent + self.error_bank.gamma * _delta_sp
                    _bank_sp_applied = True
            # ErrorBank inject on nearby (cross-K X-frame sample, X=cond_end=1 for i2v)
            if (
                _bank_flags is not None and bool(_bank_flags.get('nearby', False))
                and nearby_latent is not None and nearby_latent.numel() > 0
                and int(nearby_latent.shape[2]) > 0
            ):
                _delta_nb = self.error_bank.sample_x_frames(
                    X=int(nearby_latent.shape[2]),
                    target_shape=tuple(nearby_latent.shape),
                    device=nearby_latent.device,
                    timestep=float(sigma.item()),
                    dtype=nearby_latent.dtype,
                )
                if _delta_nb is not None:
                    nearby_latent = nearby_latent + self.error_bank.gamma * _delta_nb
                    _bank_nb_applied = True
        elif bool(self.cfg.anti_drift.drift.enabled) and not _freeze_skip_drift:
            _drift_cfg = self.cfg.anti_drift.drift
            if spatial_latent is not None:
                _drift_applied = True
                spatial_latent = corrupt_history_latents_helios(
                    spatial_latent,
                    noise_mode_prob=float(_drift_cfg.noise_mode_prob),
                    corrupt_ratio=float(_drift_cfg.corrupt_ratio),
                    clean_prob=float(_drift_cfg.clean_prob),
                    downsample_min=float(_drift_cfg.downsample_min),
                    downsample_max=float(_drift_cfg.downsample_max),
                    saturation_clean_prob=float(_drift_cfg.saturation_clean_prob),
                    is_keep_x0=bool(_drift_cfg.keep_x0),  # spatial has no clean first frame by default
                    is_frame_independent=True,  # one sigma per latent frame
                )
            if nearby_latent is not None and nearby_latent.numel() > 0 and int(nearby_latent.shape[2]) > 0:
                _drift_nearby_applied = True
                nearby_latent = corrupt_history_latents_helios(
                    nearby_latent,
                    noise_mode_prob=float(_drift_cfg.noise_mode_prob),
                    corrupt_ratio=float(_drift_cfg.corrupt_ratio),
                    clean_prob=float(_drift_cfg.clean_prob),
                    downsample_min=float(_drift_cfg.downsample_min),
                    downsample_max=float(_drift_cfg.downsample_max),
                    saturation_clean_prob=float(_drift_cfg.saturation_clean_prob),
                    is_keep_x0=False,  # the nearby segment comes entirely from predictions, so there is no anchor
                    is_frame_independent=True,
                )
        _A_triggered = False
        _A_ratio = 0.0
        _A_invalid_before = 0
        _A_invalid_after = 0
        if spatial_mask_patch is not None and random.random() < 0.5:
            _A_triggered = True
            _A_ratio = random.uniform(0.0, 0.5)
            valid_now = spatial_mask_patch[..., 0] > 0.5  # [B, N]
            _A_invalid_before = int((~valid_now).sum().item())
            rand = torch.rand_like(spatial_mask_patch[..., 0])
            force_invalid = (rand < _A_ratio) & valid_now  # [B, N]
            if bool(force_invalid.any()):
                spatial_mask_patch = spatial_mask_patch.clone()
                spatial_mask_patch = torch.where(
                    force_invalid.unsqueeze(-1),
                    torch.zeros_like(spatial_mask_patch),
                    spatial_mask_patch,
                )
                _A_invalid_after = int((spatial_mask_patch[..., 0] <= 0.5).sum().item())
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
        target_next_clean = None
        next_indices_grid = None
        if self.cfg.next_forcing.enabled:
            next_start = K if vigeo_prefix_mode else target_start + K
            next_end = next_start + K
            if latent_full.shape[2] < next_end:
                raise ValueError(
                    f"next_forcing.enabled needs latent T>={next_end} "
                    f"(target_start={target_start}, K={K}), got {latent_full.shape[2]}"
                )
            target_next_clean = latent_full[:, :, next_start:next_end].contiguous()
            next_indices_grid = self._indices_grid_for_t_indices(B, target_rope_t_indices + K, H_lat, W_lat)

        noise = torch.randn_like(target_clean)
        if dist.is_initialized():
            dist.broadcast(noise, src=0)

        target_clean_for_input = target_clean
        if anti_drift_info["inject_latent"]:
            target_clean_for_input = self._inject_target_error(target_clean, K, float(sigma.item()))
        target_noisy = (1.0 - sigma_view) * target_clean_for_input + sigma_view * noise

        if vigeo_prefix_mode:
            control_kwargs = self._build_pixel_prefix_control_kwargs(
                metadata=metadata,
                control_modes=control_modes,
                prefix_pixel_frames=prefix_pixel_frames,
                target_latent_frames=K,
                dtype=target_noisy.dtype,
            )
        else:
            target_action_t_indices = torch.arange(target_start, target_start + K, device=self.dist.device, dtype=torch.float32)
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
                    and not memory_dropped
                )
                else None
            )
            control_kwargs = self._build_control_kwargs(
                metadata=metadata,
                control_modes=control_modes,
                target_t_indices=target_action_t_indices,
                condition_t_indices=condition_action_t_indices,
                history_t_indices=history_action_t_indices,
                dtype=target_noisy.dtype,
            )
        mcp = self.components.next_forcing_head

        def _forward_target_velocity() -> torch.Tensor:
            return self.components.transformer(
                x=[target_noisy.squeeze(0)],
                t=sigma * 1000.0,
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

        if mcp is not None:
            with mcp.capturing():
                pred_velocity = _forward_target_velocity()
        else:
            pred_velocity = _forward_target_velocity()

        target_velocity = (noise - target_clean).to(dtype=pred_velocity.dtype)
        _B_invalid_ratio = 0.0
        _B_loss_valid = 0.0
        _B_loss_invalid = 0.0
        if spatial_mask_patch is not None and spatial_latent is not None:
            # mask_patch [B, N=K*H_lat*W_lat, C_patch] → [B, 1, K, H_lat, W_lat]
            mask_2d = spatial_mask_patch[..., 0]  # [B, N]
            mask_3d = mask_2d.reshape(B, K, H_lat, W_lat).unsqueeze(1)  # [B, 1, K, H_lat, W_lat]
            invalid_mask = (mask_3d < 0.5).to(dtype=pred_velocity.dtype)  # [B, 1, K, H_lat, W_lat]
            weight = 1.0 + invalid_mask
            per_elem = (pred_velocity - target_velocity).pow(2)
            _B_invalid_ratio = float(invalid_mask.mean().item())
            _denom_invalid = invalid_mask.sum().clamp_min_(1.0)
            _denom_valid = (1.0 - invalid_mask).sum().clamp_min_(1.0)
            _B_loss_invalid = float((per_elem * invalid_mask).sum().item() / _denom_invalid.item())
            _B_loss_valid = float((per_elem * (1.0 - invalid_mask)).sum().item() / _denom_valid.item())
            loss = (per_elem * weight).mean()
        else:
            loss = F.mse_loss(pred_velocity, target_velocity)
        mcp_loss_val: float | None = None
        if mcp is not None:
            assert target_next_clean is not None and next_indices_grid is not None
            mcp_sigma = self._sample_next_forcing_sigma(target_clean.dtype, latent_frames=K)
            mcp_loss, _mcp_log = mcp.compute_loss(
                x0_next=target_next_clean,
                noise=torch.randn_like(target_next_clean),
                sigma=mcp_sigma,
                next_indices_grid=next_indices_grid,
                fps=self.cfg.sample.fps,
                K=K,
                H=H_lat,
                W=W_lat,
            )
            mcp_loss_val = float(mcp_loss.item())
            _nf_w = float(self.cfg.next_forcing.loss_weight)
            _nf_ws = int(getattr(self.cfg.next_forcing, "loss_weight_warmup_steps", 0) or 0)
            if _nf_ws > 0:
                _nf_w = _nf_w * min(1.0, float(self.global_step) / float(_nf_ws))
            loss = loss + _nf_w * mcp_loss
        loss.backward()
        if self.dist.is_main and (self.global_step % 100 == 0) and spatial_latent is not None:
            print(
                f"[SpatialMaskDbg] step={self.global_step} "
                f"drift_sp={_drift_applied} drift_nb={_drift_nearby_applied} "
                f"bank_sp={_bank_sp_applied} bank_nb={_bank_nb_applied} "
                f"A_trig={_A_triggered} A_ratio={_A_ratio:.2f} "
                f"A_invalid_before={_A_invalid_before} A_invalid_after={_A_invalid_after} "
                f"B_invalid_ratio={_B_invalid_ratio:.3f} "
                f"B_loss_valid={_B_loss_valid:.4f} B_loss_invalid={_B_loss_invalid:.4f}",
                flush=True,
            )
        if self.dist.is_main and (self.global_step % 50 == 0):
            _dc = get_drift_counters()
            _tot = max(1, int(_dc.get('total', 0)))
            _drift_summary = (
                f"drift_total={_dc['total']} "
                f"noise={_dc['noise']}({100*_dc['noise']/_tot:.0f}%) "
                f"downsample={_dc['downsample']}({100*_dc['downsample']/_tot:.0f}%) "
                f"clean_s1={_dc['clean_s1']}({100*_dc['clean_s1']/_tot:.0f}%) "
                f"saturation={_dc['saturation']}({100*_dc['saturation']/_tot:.0f}%) "
                f"clean_s2={_dc['clean_s2']}({100*_dc['clean_s2']/_tot:.0f}%)"
            )
            if self.error_bank is not None:
                _bs = self.error_bank.stats()
                _per_grid_top5 = _bs.get('per_grid_top5', {})
                print(
                    f"[BankDbg] step={self.global_step} "
                    f"is_warm={self.error_bank.is_warm()} "
                    f"n_push={_bs.get('n_push', 0)} "
                    f"total_residuals={_bs.get('total_size', 0)} "
                    f"n_buckets={_bs.get('n_buckets', 0)} "
                    f"per_K={_bs.get('per_K', {})} "
                    f"per_grid_top5={dict(list(_per_grid_top5.items())[:5])} | "
                    f"{_drift_summary}",
                    flush=True,
                )
            else:
                print(
                    f"[BankDbg] step={self.global_step} bank=None | {_drift_summary}",
                    flush=True,
                )

        self._sync_grads_outside_fsdp()   # all-reduce params outside the FSDP tree (HistoryEncoder / LoRA)
        trainable = []
        if self.history_encoder is not None:
            trainable += [p for p in self.history_encoder.parameters() if p.requires_grad]
        trainable += [p for p in self.components.transformer.parameters() if p.requires_grad]
        if self.components.lora_manager is not None:
            trainable += self.components.lora_manager.get_trainable_parameters()
        if self.components.next_forcing_head is not None:
            nf_params = self.components.next_forcing_head.trainable_parameters()
            trainable += nf_params
            self._allreduce_grads(nf_params)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.cfg.optimizer.max_grad_norm)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self._push_error_bank(target_noisy, sigma_view, pred_velocity, target_clean, float(sigma.item()))

        return float(loss.item()), float(grad_norm.item()), {
            "K": K,
            "gap_steps": gap_steps,
            "cond_mode": cond_mode,
            "cond_end": cond_end,
            "control_modes": control_modes,
            "spatial_context": 0 if spatial_latent is None else int(spatial_latent.shape[2]),
            "vigeo_scale": (
                None if spatial_context is None else spatial_context.get("vigeo_scale")
            ),
            "sink_prefix_pixel_index": sink_prefix_pixel_index,
            "sigma": float(sigma.item()),
            "memory_dropped": memory_dropped,
            "condition_prefix_fixed": condition_prefix_fixed,
            "mcp_loss": mcp_loss_val,
            "source": str(metadata.get("source", "")),
            "video_id": str(metadata.get("video_id", "unknown")),
            "frame_start": int(_first_scalar(metadata.get("frame_start", -1))),
            "frame_end": int(_first_scalar(metadata.get("frame_end", -1))),
            **anti_drift_info,
        }

    def _train_one_step_inline(self, batch: Any) -> tuple[float, float, dict[str, Any]]:
        """In-clip clean-mask conditioning for bidirectional pretraining (i2v / v2v / t2v).

        The condition frames are a clean prefix of the same K-frame sequence (sigma=0) while the rest are
        noised, all in one sequence; the loss is computed on the generated frames only.
        Used when layout.condition.type == 'inline' (history=0); it involves no memory, nearby context,
        sink or spatial condition.
        """
        assert self.components is not None
        assert self.optimizer is not None
        self.components.transformer.train()
        self.optimizer.zero_grad(set_to_none=True)

        video_pixels, caption, metadata = self._unpack_batch(batch)
        prompt_caption = self._caption_with_prefix(caption, metadata)
        context = self._encode_caption(prompt_caption, sync=False)

        latent_full = self._encode_video(video_pixels, metadata=metadata)          # [B, C, T, H, W]
        B, _C_lat, K, H_lat, W_lat = latent_full.shape
        target_clean = latent_full.contiguous()

        if self.dist.is_main:
            r = random.random()
            ratio = random.uniform(
                self.cfg.layout.condition.v2v_ratio_min,
                self.cfg.layout.condition.v2v_ratio_max,
            )
            pick = torch.tensor([r, ratio], device=self.dist.device, dtype=torch.float32)
        else:
            pick = torch.zeros(2, device=self.dist.device, dtype=torch.float32)
        broadcast_tensor(pick)
        r = float(pick[0].item())
        ratio = float(pick[1].item())
        i2v_p = float(self.cfg.layout.condition.i2v_prob)
        v2v_p = float(self.cfg.layout.condition.v2v_prob)
        if r < i2v_p:
            cond_mode, cond_end = "i2v", min(1, K - 1)
        elif r < i2v_p + v2v_p:
            cond_mode, cond_end = "v2v", max(1, min(K - 1, int(K * ratio)))
        else:
            cond_mode, cond_end = "t2v", 0

        sigma = self._sample_sigma(target_clean.dtype, latent_frames=int(K))
        sigma_view = sigma.view(1, 1, 1, 1, 1)
        noise = torch.randn_like(target_clean)   # sampled independently per rank
        target_noisy = (1.0 - sigma_view) * target_clean + sigma_view * noise
        if cond_end > 0:
            cond_mask = torch.zeros(1, 1, K, 1, 1, device=self.dist.device, dtype=target_clean.dtype)
            cond_mask[:, :, :cond_end] = 1.0
            target_noisy = cond_mask * target_clean + (1.0 - cond_mask) * target_noisy

        gen_t_indices = torch.arange(0, K, device=self.dist.device, dtype=torch.float32)
        pred_velocity = self.components.transformer(
            x=[target_noisy.squeeze(0)],
            t=sigma * 1000.0,
            context=[context],
            seq_len=K * H_lat * W_lat,
            fps=self.cfg.sample.fps,
            gen_t_indices_override=gen_t_indices,
            cond_latent_frames=cond_end,   # the first cond_end frames are clean conditions (sigma=0 inside the model)
        )
        target_velocity = (noise - target_clean).to(dtype=pred_velocity.dtype)
        if cond_end > 0:
            loss = F.mse_loss(pred_velocity[:, :, cond_end:], target_velocity[:, :, cond_end:])
        else:
            loss = F.mse_loss(pred_velocity, target_velocity)
        loss.backward()

        self._sync_grads_outside_fsdp()   # all-reduce params outside the FSDP tree
        trainable = [p for p in self.components.transformer.parameters() if p.requires_grad]
        if self.components.lora_manager is not None:
            trainable += self.components.lora_manager.get_trainable_parameters()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.cfg.optimizer.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return float(loss.item()), float(grad_norm.item()), {
            "K": K,
            "gap_steps": 0,
            "cond_mode": cond_mode,
            "cond_end": cond_end,
            "control_modes": [],
            "spatial_context": 0,
            "memory_dropped": False,
            "anti_drift": "clean",
            "sigma": float(sigma.item()),
            "source": str(metadata.get("source", "")),
            "video_id": str(metadata.get("video_id", "unknown")),
            "frame_start": int(_first_scalar(metadata.get("frame_start", -1))),
            "frame_end": int(_first_scalar(metadata.get("frame_end", -1))),
        }

    @torch.no_grad()
    def validate(self, step: int) -> None:
        assert self.components is not None

        self.components.transformer.eval()
        if self.history_encoder is not None:
            self.history_encoder.eval()
        for mode_name, mode_cfg in self.cfg.validation.modes.items():
            self.validate_one_mode(mode_name, mode_cfg, step)
            self._cleanup_after_validation()
        if dist.is_initialized():
            dist.barrier()
        if self.dist.is_main:
            self._plot_validation_metric_history(Path(self.cfg.run.output_dir) / "validation" / "metrics")
        self._cleanup_after_validation()
        self.components.transformer.train()
        if self.history_encoder is not None:
            self.history_encoder.train(self.cfg.memory.train)
        if dist.is_initialized():
            dist.barrier()
        self._cleanup_after_validation()

    @torch.no_grad()
    def validate_one_mode(
        self,
        mode_name: str,
        mode_cfg: ValidationModeConfig,
        step: int,
    ) -> None:
        assert self.components is not None

        K = int(mode_cfg.layout.output_latent_frames)
        requested_rounds = int(mode_cfg.rollout_rounds)
        N = self._validation_history_latents(mode_cfg)
        max_gap_sec = float(mode_cfg.layout.max_gap_sec or 0.0)
        gap_steps = int(max_gap_sec * self.cfg.sample.fps / self.cfg.sample.temporal_stride)
        cond_end = self._validation_cond_end(mode_cfg, K)
        if N > 0 and cond_end > N:
            raise ValueError(f"validation mode {mode_name}: condition_latent_frames={cond_end} > history_latent_frames={N}")
        if N > 0 and self.history_encoder is None:
            raise ValueError(f"validation mode {mode_name} needs memory/history, but training layout disabled memory")
        explicit_condition = cond_end if N == 0 else 0
        dynamic_rounds = bool(getattr(self.cfg.validation, "dynamic_rounds", False))

        def _needed_pixels_for_rounds(rounds: int) -> tuple[int, int, int]:
            if self._uses_vigeo_prefix_last_frame():
                prefix_latents = N if N > 0 else 1
                required = (
                    self.cfg.layout.sink_latent_frames
                    + prefix_latents
                    + int(rounds) * K
                )
                needed = required + 1
                exact_pixels = (
                    self._vigeo_target_prefix_pixel_frames(
                        history_latent_frames=N
                    )
                    + int(rounds) * K * int(self.cfg.sample.temporal_stride)
                )
                stride = int(self.cfg.sample.temporal_stride)
                pixels = 1 + ((exact_pixels - 1 + stride - 1) // stride) * stride
                return required, needed, pixels
            required = self.cfg.layout.sink_latent_frames + gap_steps + N + explicit_condition + int(rounds) * K
            # The LTX VAE/downstream dataset path can produce one fewer latent than
            # the nominal (frames - 1) // stride + 1 count on long validation clips.
            # Request one guard latent for GT slicing; rollout/metrics still consume
            # exactly required_latents.
            needed = required + 1
            pixels = (needed - 1) * self.cfg.sample.temporal_stride + 1
            return required, needed, pixels

        _wbench = self._is_wbench_mode(mode_cfg)
        _genonly = _wbench or str(mode_cfg.dataset.source) == "custom_i2v"
        if _genonly:
            stride = int(self.cfg.sample.temporal_stride)
            if self._uses_vigeo_prefix_last_frame():
                _prefix_pixels = self._vigeo_target_prefix_pixel_frames(history_latent_frames=N)
            else:
                _prefix_pixels = (self.cfg.layout.sink_latent_frames + gap_steps + N + explicit_condition) * stride
            _prefix_pixels = max(1, _prefix_pixels, int(self.cfg.spatial_memory.num_context_frames))
            dataset = self.validation_datasets.get(mode_name)
            if dataset is None:
                dataset = build_validation_dataset(
                    self.cfg, mode_cfg, min_frames=_prefix_pixels, max_frames=_prefix_pixels
                )
                self.validation_datasets[mode_name] = dataset
            dynamic_rounds = False
            if _wbench:
                _cpt = max(1, int(getattr(mode_cfg, "wbench_chunks_per_turn", 3)))
                _tc = (
                    (getattr(dataset, "full_turn_counts", None) or dataset.turn_counts)
                    if bool(getattr(mode_cfg, "wbench_all_turns", False))
                    else dataset.turn_counts
                )
                requested_rounds = max(1, int(max(_tc)) * _cpt)
            rank0_print(
                self.dist, "[Validation]",
                f"{mode_name}: genonly source={mode_cfg.dataset.source} n={len(dataset)} "
                f"rounds={requested_rounds} prefix_pixels={_prefix_pixels}",
            )

        min_rounds = 1 if dynamic_rounds else requested_rounds
        _min_required_latents, _min_needed_latents, min_needed_pixels = _needed_pixels_for_rounds(min_rounds)
        _required_latents, needed_latents, needed_pixels = _needed_pixels_for_rounds(requested_rounds)
        if _genonly:
            min_needed_pixels = _prefix_pixels

        dataset = self.validation_datasets.get(mode_name)
        if dataset is None:
            dataset = build_validation_dataset(
                self.cfg,
                mode_cfg,
                min_frames=min_needed_pixels,
                max_frames=needed_pixels,
            )
            self.validation_datasets[mode_name] = dataset
            filter_suffix = f" filter={mode_cfg.dataset.filter}" if mode_cfg.dataset.filter else ""
            rank0_print(
                self.dist,
                "[Validation]",
                f"{mode_name}: dataset={mode_cfg.dataset.source}{filter_suffix} real={len(dataset)} "
                f"frames=[{min_needed_pixels},{needed_pixels}] dynamic_rounds={dynamic_rounds}",
            )

        if len(dataset) == 0:
            rank0_print(self.dist, "[Validation]", f"{mode_name}: empty dataset")
            return
        start_idx = int(getattr(self.cfg.validation, "sample_offset", 0)) + self.dist.rank * self.cfg.validation.max_samples
        end_idx = start_idx + self.cfg.validation.max_samples
        if bool(getattr(mode_cfg, "all_samples", False)):
            _sample_schedule: list[int | None] = shard_indices_with_padding(
                list(range(len(dataset))), rank=self.dist.rank, world_size=self.dist.world_size
            )
            start_idx, end_idx = 0, len(_sample_schedule)
        else:
            _sample_schedule = list(range(start_idx, end_idx))

        suffix = self.cfg.validation.step_dir_suffix
        step_dir_name = f"step-{step:06d}{suffix}"
        mode_dir = Path(self.cfg.run.output_dir) / "validation" / step_dir_name / mode_name
        skip_existing = mode_dir.exists()
        if dist.is_initialized():
            skip_tensor = torch.tensor(
                [int(skip_existing) if self.dist.is_main else 0],
                device=self.dist.device,
                dtype=torch.long,
            )
            dist.broadcast(skip_tensor, src=0)
            skip_existing = bool(skip_tensor.item())
        if skip_existing:
            rank0_print(self.dist, "[Validation]", f"skip existing mode_dir={mode_dir}")
            return
        mode_dir.mkdir(parents=True, exist_ok=True)
        rank0_print(
            self.dist,
            "[Validation]",
            f"step={step} mode={mode_name} rank={self.dist.rank} samples=[{start_idx},{end_idx}) "
            f"K={K} rounds={requested_rounds} cond={mode_cfg.layout.condition}:{cond_end} "
            f"dynamic_rounds={dynamic_rounds}",
        )

        _n_variants = max(1, int(getattr(mode_cfg, "prompt_variants", 1)))
        _prompt_pool : list[str] = []
        if _n_variants > 1:
            rank0_print(
                self.dist, "[Validation]",
            )


        for _slot_pos, _slot in enumerate(_sample_schedule):
            _pad_slot = _slot is None
            global_idx = int(_slot) if _slot is not None else 0
            sample_idx = global_idx % len(dataset)
            batch = dataset[sample_idx]
            video_pixels, caption, metadata = self._unpack_batch(batch)
            target_base_start = self.cfg.layout.sink_latent_frames + gap_steps + N + explicit_condition
            if self._uses_vigeo_prefix_last_frame():
                latent_full = self._build_vigeo_validation_latent_full(
                    video_pixels=video_pixels,
                    metadata=metadata,
                    required_latents=_required_latents,
                    target_base_start=target_base_start,
                    history_latent_frames=N,
                    allow_short=dynamic_rounds or _genonly,
                    allow_empty_target=_genonly,
                )
            else:
                latent_full = self._encode_video(video_pixels, needed_latents=needed_latents)
            self._extend_validation_camera_static(metadata, min_frames=needed_pixels)
            actual_cfg_scale = self._validation_cfg_scale()
            negative_context = (
                self._encode_caption(self.cfg.validation.negative_prompt, sync=False)
                if actual_cfg_scale > 1.0
                else None
            )
            rounds = requested_rounds
            local_rounds = requested_rounds
            if dynamic_rounds:
                available_target_latents = max(0, int(latent_full.shape[2]) - int(target_base_start))
                local_rounds = min(requested_rounds, available_target_latents // K)
                # Keep all ranks doing the same number of FSDP forwards.
                rounds = requested_rounds
                if local_rounds <= 0:
                    rank0_print(
                        self.dist,
                        "[Validation]",
                        f"{mode_name}: sample={sample_idx} has no GT rollout rounds "
                        f"latent_T={int(latent_full.shape[2])} target_start={target_base_start} "
                        f"K={K}; running requested_rounds={requested_rounds} for rank alignment",
                    )

            if _wbench:
                self._prepare_wbench_validation_metadata(
                    metadata=metadata,
                    mode_cfg=mode_cfg,
                    K=K,
                    target_base_start=target_base_start,
                    needed_pixels=needed_pixels,
                    rollout_rounds=rounds,
                    video_pixels=video_pixels,
                )

            _all_random = bool(getattr(mode_cfg, "prompt_variants_all_random", False))
            _variant_caps = self._build_variant_captions(
                caption=caption,
                prompt_pool=_prompt_pool,
                n_variants=_n_variants,
                seed_key=int(sample_idx) + int(getattr(mode_cfg, 'prompt_variants_seed', 0)) * 7919,
                all_random=_all_random,
            )

            for _vi, _raw_cap in enumerate(_variant_caps):
                prompt_caption = self._caption_with_prefix(_raw_cap, metadata)
                context = self._encode_caption(prompt_caption, sync=False)
                scheduled_prompt_captions = self._validation_prompt_schedule(
                    mode_cfg=mode_cfg,
                    prompt_caption=prompt_caption,
                    caption=_raw_cap,
                    metadata=metadata,
                    rounds=requested_rounds,
                )
                scheduled_contexts = (
                    [self._encode_caption(p, sync=False) for p in scheduled_prompt_captions]
                    if scheduled_prompt_captions
                    else None
                )
                (
                    metrics,
                    pred_latents,
                    nearby_condition_latents,
                    spatial_condition_latents,
                    spatial_condition_masks,
                    spatial_condition_prefix_latents,
                ) = self._validate_rollout_sample(
                    video_pixels=video_pixels,
                    latent_full=latent_full,
                    context=context,
                    scheduled_contexts=scheduled_contexts,
                    scheduled_prompt_captions=scheduled_prompt_captions,
                    negative_context=negative_context,
                    metadata=metadata,
                    mode_cfg=mode_cfg,
                    K=K,
                    rounds=rounds,
                    N=N,
                    gap_steps=gap_steps,
                    cond_end=cond_end,
                )
                _vtag = "" if _n_variants <= 1 else f"_var{_vi:02d}"
                stem = f"rank-{self.dist.rank:03d}_global-{global_idx:06d}_sample-{sample_idx:06d}{_vtag}"
                if _pad_slot:
                    del context, scheduled_contexts, metrics, pred_latents, nearby_condition_latents, spatial_condition_latents, spatial_condition_masks, spatial_condition_prefix_latents
                    self._cleanup_after_validation()
                    continue
                payload = {
                    "step": step,
                    "mode": mode_name,
                    "rank": self.dist.rank,
                    "global_idx": global_idx,
                    "sample_idx": sample_idx,
                    "prompt_variant_index": _vi,
                    "prompt_variant_total": len(_variant_caps),
                    "prompt_variant_is_own": bool(_vi == 0),
                    "caption": _raw_cap,
                    "prompt_caption": prompt_caption,
                    "prompt_schedule": scheduled_prompt_captions,
                    "K": K,
                    "rollout_rounds": rounds,
                    "requested_rollout_rounds": requested_rounds,
                    "generated_rollout_rounds": rounds,
                    "local_rollout_rounds": local_rounds,
                    "local_gt_rollout_rounds": local_rounds,
                    "actual_rollout_rounds": rounds,
                    "dynamic_rounds": dynamic_rounds,
                    "min_needed_pixels": min_needed_pixels,
                    "max_needed_pixels": needed_pixels,
                    "loaded_latent_frames": int(latent_full.shape[2]),
                    "validation_camera_extension": bool(metadata.get("validation_camera_extension", False)),
                    "validation_camera_extension_mode": metadata.get("validation_camera_extension_mode"),
                    "validation_camera_static_extension": bool(metadata.get("validation_camera_static_extension", False)),
                    "validation_camera_original_frames": metadata.get("validation_camera_original_frames"),
                    "validation_camera_extended_frames": metadata.get("validation_camera_extended_frames"),
                    "validation_video_available_frames": metadata.get("validation_video_available_frames"),
                    "validation_video_requested_frames": metadata.get("validation_video_requested_frames"),
                    "validation_video_used_frames": metadata.get("validation_video_used_frames"),
                    "history_latent_frames": N,
                    "gap_steps": gap_steps,
                    "condition": mode_cfg.layout.condition,
                    "condition_latent_frames": cond_end,
                    "dataset_source": mode_cfg.dataset.source,
                    "dataset_filter": mode_cfg.dataset.filter,
                    "control": list(mode_cfg.control),
                    "use_memory": bool(mode_cfg.use_memory),
                    "action_cfg_scale": float(mode_cfg.action_cfg_scale),
                    "validation_scheduler": self.cfg.validation.scheduler,
                    "cfg_scale": self.cfg.validation.cfg_scale,
                    "actual_cfg_scale": actual_cfg_scale,
                    "stg_scale": self.cfg.validation.stg_scale,
                    "stg_blocks": list(self.cfg.validation.stg_blocks),
                    "rescale_scale": self.cfg.validation.rescale_scale,
                    "video_history_latent_frames": self.cfg.validation.video_history_latent_frames or N,
                    "video_decode_mode": (
                        (
                            "full_i2v_latent_stream"
                            if self.cfg.validation.vigeo_pred_decode_mode == "full"
                            else "chunk_i2v_rgb_motion_overlap"
                        )
                        if self._uses_vigeo_prefix_last_frame()
                        else "full_latent"
                    ),
                    "vigeo_handoff_mode": (
                        self.cfg.validation.vigeo_handoff_mode
                        if self._uses_vigeo_prefix_last_frame()
                        else None
                    ),
                    "vigeo_pred_decode_mode": (
                        self.cfg.validation.vigeo_pred_decode_mode
                        if self._uses_vigeo_prefix_last_frame()
                        else None
                    ),
                    "metrics": metrics,
                }
                out_path = mode_dir / f"{stem}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                self._write_validation_prompt_file(
                    out_path.with_suffix(".txt"),
                    payload=payload,
                    metadata=metadata,
                    prompt_caption=prompt_caption,
                    caption=_raw_cap,
                    actual_cfg_scale=actual_cfg_scale,
                    K=K,
                    cond_end=cond_end,
                )
                self._update_validation_metric_history(mode_name, payload)
                if self.cfg.validation.save_videos:
                    self._save_validation_videos(
                        mode_dir=mode_dir,
                        stem=stem,
                        latent_full=latent_full,
                        pred_latents=pred_latents,
                        nearby_condition_latents=nearby_condition_latents,
                        metadata=metadata,
                        K=K,
                        N=N,
                        gap_steps=gap_steps,
                        cond_end=cond_end,
                        spatial_condition_latents=spatial_condition_latents,
                        spatial_condition_masks=spatial_condition_masks,
                        spatial_condition_prefix_latents=spatial_condition_prefix_latents,
                    )
                    if _wbench:
                        self._save_wbench_output_video(
                            mode_cfg=mode_cfg,
                            mode_dir=mode_dir,
                            stem=stem,
                            pred_latents=pred_latents,
                            metadata=metadata,
                            scheduled_prompt_captions=scheduled_prompt_captions,
                            latent_full=latent_full,
                            N=N,
                            gap_steps=gap_steps,
                            cond_end=cond_end,
                        )
                del context, scheduled_contexts, metrics, pred_latents, nearby_condition_latents, spatial_condition_latents, spatial_condition_masks, spatial_condition_prefix_latents, payload
                self._cleanup_after_validation()
            del batch, video_pixels, latent_full, negative_context
            self._cleanup_after_validation()

    def _is_wbench_mode(self, mode_cfg: ValidationModeConfig) -> bool:
        return str(mode_cfg.dataset.source) == "wbench_navi"

    def _prepare_wbench_validation_metadata(
        self,
        *,
        metadata: dict[str, Any],
        mode_cfg: ValidationModeConfig,
        K: int,
        target_base_start: int,
        needed_pixels: int,
        rollout_rounds: int,
        video_pixels: torch.Tensor | None = None,
    ) -> None:
        if bool(getattr(mode_cfg, "wbench_all_turns", False)):
            actions = [str(x) for x in (metadata.get("wbench_turn_actions") or metadata.get("wbench_actions") or [])]
        else:
            actions = [str(x) for x in (metadata.get("wbench_actions") or [])]
        chunks_per_turn = max(1, int(getattr(mode_cfg, "wbench_chunks_per_turn", 3)))
        output_rounds = min(int(rollout_rounds), len(actions) * chunks_per_turn)
        metadata["wbench_chunks_per_turn"] = int(chunks_per_turn)
        metadata["wbench_output_rounds"] = int(output_rounds)
        metadata["wbench_used_actions"] = list(actions)
        metadata["wbench_used_n_turns"] = int(len(actions))
        metadata["wbench_frames_per_turn"] = int(chunks_per_turn) * int(K) * int(self.cfg.sample.temporal_stride)
        metadata["wbench_target_base_start"] = int(target_base_start)
        _lock = bool(getattr(mode_cfg, "wbench_lock_subject_foreground", False))
        if str(metadata.get("wbench_perspective", "third_person")) == "first_person":
            _lock = _lock and bool(
                getattr(mode_cfg, "wbench_lock_subject_foreground_first_person", False)
            )
        metadata["wbench_lock_subject_foreground"] = _lock
        metadata["wbench_subject_anchor_refresh"] = bool(
            getattr(mode_cfg, "wbench_subject_anchor_refresh", False)
        )
        metadata["wbench_subject_anchor_refresh_alpha"] = float(
            getattr(mode_cfg, "wbench_subject_anchor_refresh_alpha", 1.0) or 1.0
        )
        metadata["wbench_subject_anchor_adaptive"] = bool(
            getattr(mode_cfg, "wbench_subject_anchor_adaptive", False)
        )
        metadata["wbench_subject_anchor_lost_mae"] = float(
            getattr(mode_cfg, "wbench_subject_anchor_lost_mae", 0.10) or 0.10
        )
        metadata["wbench_spatial_drop_rounds"] = [
            int(x) for x in (getattr(mode_cfg, "wbench_spatial_drop_rounds", None) or [])
        ]
        metadata["wbench_subject_mask_dilation_pixels"] = max(
            0, int(getattr(mode_cfg, "wbench_subject_mask_dilation_pixels", 12))
        )

        orbit_pivot = None
        if (
            bool(getattr(mode_cfg, "wbench_orbit_subject_depth", False))
            and str(metadata.get("wbench_perspective", "third_person")) != "first_person"
            and video_pixels is not None
        ):
            orbit_pivot = self._wbench_subject_pivot(
                video_pixels=video_pixels,
                mask_path=str(metadata.get("wbench_subject_mask") or ""),
            )
            if orbit_pivot is not None:
                metadata["wbench_orbit_pivot"] = [float(value) for value in orbit_pivot]
                metadata["wbench_orbit_radius"] = float(np.linalg.norm(orbit_pivot))
                rank0_print(
                    self.dist, "[Validation]",
                    f"wbench case={metadata.get('wbench_case_id')} "
                    f"orbit_pivot={np.asarray(orbit_pivot).round(4).tolist()} "
                    f"radius={np.linalg.norm(orbit_pivot):.3f}",
                )
        cam = self._build_wbench_camera_trajectory(
            actions=actions,
            perspective=str(metadata.get("wbench_perspective", "third_person")),
            mode_cfg=mode_cfg,
            K=K,
            chunks_per_turn=chunks_per_turn,
            target_base_start=target_base_start,
            frames=int(needed_pixels),
            orbit_pivot=orbit_pivot,
        )
        metadata["cam_c2w"] = cam
        metadata["cam_c2w_raw"] = cam.clone()
        metadata["has_camera"] = True

    def _wbench_subject_pivot(
        self,
        *,
        video_pixels: torch.Tensor,
        mask_path: str,
    ) -> np.ndarray | None:
        """Estimate the subject's robust XYZ center in the initial camera frame."""
        try:
            first = self._select_video_pixel_frames(video_pixels, [0])
            geometry = self._get_vigeo_geometry().infer_stream_geometry(
                video_pixels=first,
                kv_caches=None,
                reset_cache=True,
                chunk_size=1,
                total_budget=int(self.cfg.spatial_memory.vigeo_cache_budget),
            )
            points = geometry.pointmaps[0].detach().float().cpu()
            valid = geometry.valid_masks[0].detach().bool().cpu()
            valid = valid & torch.isfinite(points).all(dim=-1) & (points[..., 2] > 0)
            if mask_path:
                from PIL import Image

                mask_img = Image.open(mask_path).convert("L").resize(
                    (int(points.shape[-2]), int(points.shape[-3])),
                    resample=Image.Resampling.NEAREST,
                )
                mask = torch.from_numpy(np.array(mask_img)) > 127
                sel = valid & mask
            else:
                h, w = points.shape[:2]
                center = torch.zeros_like(valid)
                center[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
                sel = valid & center
            if int(sel.sum()) < 64:
                return None

            selected_points = points[sel]
            z = selected_points[:, 2]
            z_lo, z_hi = torch.quantile(z, torch.tensor([0.1, 0.9]))
            selected_points = selected_points[(z >= z_lo) & (z <= z_hi)]
            if int(selected_points.shape[0]) < 64:
                return None
            pivot = selected_points.median(dim=0).values.numpy().astype(np.float64)
            if not np.isfinite(pivot).all() or float(pivot[2]) <= 1e-4:
                return None
            return pivot
        except Exception as exc:
            rank0_print(self.dist, "[Validation]", f"wbench subject pivot failed (fail-open): {exc}")
            return None

    def _wbench_subject_foreground_layers(
        self,
        *,
        source_pixel: torch.Tensor,
        metadata: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Build a camera-locked subject layer and a dilated bank exclusion mask."""
        if not _as_bool(metadata.get("wbench_lock_subject_foreground", False)):
            return None
        mask_path = str(metadata.get("wbench_subject_mask") or "")
        if not mask_path or not Path(mask_path).is_file():
            rank0_print(
                self.dist,
                "[Validation]",
                f"wbench case={metadata.get('wbench_case_id')} has no subject mask; "
                "foreground lock disabled for this sample",
            )
            return None
        try:
            from PIL import Image

            height, width = int(source_pixel.shape[-2]), int(source_pixel.shape[-1])
            mask_img = Image.open(mask_path).convert("L").resize(
                (width, height), resample=Image.Resampling.NEAREST
            )
            subject_mask = torch.from_numpy(np.array(mask_img)) > 127
            if int(subject_mask.sum()) < 64:
                raise ValueError(f"subject mask has only {int(subject_mask.sum())} pixels")
            dilation = max(
                0, int(metadata.get("wbench_subject_mask_dilation_pixels", 12))
            )
            exclusion_mask = subject_mask
            if dilation > 0:
                kernel = 2 * dilation + 1
                exclusion_mask = (
                    F.max_pool2d(
                        subject_mask.float().view(1, 1, height, width),
                        kernel_size=kernel,
                        stride=1,
                        padding=dilation,
                    )[0, 0]
                    > 0.5
                )
            rank0_print(
                self.dist,
                "[Validation]",
                f"wbench case={metadata.get('wbench_case_id')} foreground locked "
                f"subject={float(subject_mask.float().mean()):.3f} "
                f"excluded={float(exclusion_mask.float().mean()):.3f}",
            )
            return (
                source_pixel.detach().to(device="cpu", dtype=torch.float16).contiguous(),
                subject_mask.contiguous(),
                exclusion_mask.contiguous(),
            )
        except Exception as exc:
            rank0_print(
                self.dist,
                "[Validation]",
                f"wbench foreground lock failed (fail-open): {exc}",
            )
            return None

    @staticmethod
    def _exclude_subject_from_valid_mask(
        valid_mask: torch.Tensor,
        exclusion_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if exclusion_mask is None:
            return valid_mask.bool().contiguous()
        excluded = exclusion_mask.to(device=valid_mask.device, dtype=torch.bool)
        while excluded.dim() < valid_mask.dim():
            excluded = excluded.unsqueeze(0)
        return (valid_mask.bool() & (~excluded.expand_as(valid_mask))).contiguous()

    def _composite_wbench_subject_anchor(
        self,
        *,
        bank: _RolloutSpatialBank,
        warped_pixels: torch.Tensor,
        coverage_pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bank.subject_anchor_pixel is None or bank.subject_anchor_mask is None:
            return warped_pixels, coverage_pixels
        if not getattr(bank, "subject_anchor_paste", True):
            return warped_pixels, coverage_pixels
        anchor = bank.subject_anchor_pixel.to(
            device=warped_pixels.device, dtype=warped_pixels.dtype
        ).unsqueeze(2)
        if int(anchor.shape[0]) != int(warped_pixels.shape[0]):
            if int(anchor.shape[0]) != 1:
                raise ValueError(
                    "subject anchor batch does not match warped pixels: "
                    f"{anchor.shape[0]} != {warped_pixels.shape[0]}"
                )
            anchor = anchor.expand(int(warped_pixels.shape[0]), -1, -1, -1, -1)
        mask = bank.subject_anchor_mask.to(
            device=warped_pixels.device, dtype=torch.bool
        ).view(1, 1, 1, *bank.subject_anchor_mask.shape[-2:])
        warped_pixels = torch.where(mask, anchor, warped_pixels)
        coverage_pixels = torch.where(
            mask,
            torch.ones((), device=coverage_pixels.device, dtype=coverage_pixels.dtype),
            coverage_pixels,
        )
        return warped_pixels.contiguous(), coverage_pixels.contiguous()

    def _build_wbench_camera_trajectory(
        self,
        *,
        actions: list[str],
        perspective: str,
        mode_cfg: ValidationModeConfig,
        K: int,
        chunks_per_turn: int,
        target_base_start: int,
        frames: int,
        orbit_pivot: np.ndarray | None = None,
    ) -> torch.Tensor:
        stride = max(1, int(self.cfg.sample.temporal_stride))
        forward_step = float(getattr(mode_cfg, "wbench_forward_speed_per_latent", 0.16)) / float(stride)
        yaw_step = math.radians(float(getattr(mode_cfg, "wbench_yaw_deg_per_latent", 6.0))) / float(stride)
        pitch_step = math.radians(float(getattr(mode_cfg, "wbench_pitch_deg_per_latent", 6.0))) / float(stride)
        frames_per_turn = max(1, int(chunks_per_turn) * int(K) * stride)
        action_start_pixel = (
            self._vigeo_target_prefix_pixel_frames(
                history_latent_frames=self._validation_history_latents(mode_cfg)
            )
            if self._uses_vigeo_prefix_last_frame()
            else max(0, int(target_base_start) * stride)
        )
        orbit = str(perspective) != "first_person"

        if orbit:
            poses = self._build_wbench_orbit_camera(
                actions=actions,
                frames=int(frames),
                action_start_pixel=action_start_pixel,
                frames_per_turn=frames_per_turn,
                forward_step=forward_step,
                yaw_step=yaw_step,
                pitch_step=pitch_step,
                pivot=orbit_pivot,
            )
        else:
            poses = self._build_wbench_first_person_camera(
                actions=actions,
                frames=int(frames),
                action_start_pixel=action_start_pixel,
                frames_per_turn=frames_per_turn,
                forward_step=forward_step,
                yaw_step=yaw_step,
                pitch_step=pitch_step,
            )
        return torch.from_numpy(poses.astype(np.float32))

    def _build_wbench_first_person_camera(
        self,
        *,
        actions: list[str],
        frames: int,
        action_start_pixel: int,
        frames_per_turn: int,
        forward_step: float,
        yaw_step: float,
        pitch_step: float,
    ) -> np.ndarray:
        poses: list[np.ndarray] = []
        T = np.eye(4, dtype=np.float64)
        for frame_idx in range(int(frames)):
            if frame_idx > 0:
                nav = self._wbench_nav_for_frame(
                    actions=actions,
                    frame_idx=frame_idx,
                    action_start_pixel=action_start_pixel,
                    frames_per_turn=frames_per_turn,
                )
                if nav["yaw"] != 0:
                    T[:3, :3] = T[:3, :3] @ self._wbench_rot_y(yaw_step * np.sign(nav["yaw"]))
                if nav["pitch"] != 0:
                    T[:3, :3] = T[:3, :3] @ self._wbench_rot_x(pitch_step * np.sign(nav["pitch"]))
                if nav["forward"] != 0:
                    T[:3, 3] += T[:3, :3] @ np.array([0.0, 0.0, forward_step * np.sign(nav["forward"])])
                if nav["right"] != 0:
                    T[:3, 3] += T[:3, :3] @ np.array([forward_step * np.sign(nav["right"]), 0.0, 0.0])
            poses.append(T.copy())
        return np.stack(poses, axis=0)

    def _build_wbench_orbit_camera(
        self,
        *,
        actions: list[str],
        frames: int,
        action_start_pixel: int,
        frames_per_turn: int,
        forward_step: float,
        yaw_step: float,
        pitch_step: float,
        pivot: np.ndarray | None = None,
    ) -> np.ndarray:
        # Work directly in the initial camera frame. C0 is exactly identity and
        # C(theta) = [Q, p + s - Qp], so Q^T((p+s)-t) == p: the subject keeps
        # the same camera-local coordinates under every orbit rotation.
        if pivot is None:
            pivot = np.array(
                [0.0, 0.0, 0.16 / math.radians(6.0)], dtype=np.float64
            )
        else:
            pivot = np.asarray(pivot, dtype=np.float64).reshape(3)
        yaw = 0.0
        elevation = 0.0
        rig_translation = np.zeros(3, dtype=np.float64)

        def cam_pose() -> np.ndarray:
            rotation = self._wbench_rot_y(yaw) @ self._wbench_rot_x(elevation)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rotation
            T[:3, 3] = pivot + rig_translation - rotation @ pivot
            return T

        def screen_space_ground_axes() -> tuple[np.ndarray, np.ndarray]:
            rotation = cam_pose()[:3, :3]
            camera_right = rotation[:, 0].copy()
            camera_forward = rotation[:, 2].copy()
            camera_right[1] = 0.0
            camera_forward[1] = 0.0
            camera_right = camera_right / (np.linalg.norm(camera_right) + 1e-8)
            camera_forward = camera_forward / (np.linalg.norm(camera_forward) + 1e-8)
            return camera_forward, camera_right

        poses: list[np.ndarray] = []
        for frame_idx in range(int(frames)):
            if frame_idx > 0:
                nav = self._wbench_nav_for_frame(
                    actions=actions,
                    frame_idx=frame_idx,
                    action_start_pixel=action_start_pixel,
                    frames_per_turn=frames_per_turn,
                )
                if nav["yaw"] != 0:
                    yaw -= yaw_step * np.sign(nav["yaw"])
                if nav["pitch"] != 0:
                    elevation = float(np.clip(
                        elevation + pitch_step * np.sign(nav["pitch"]),
                        math.radians(-60.0),
                        math.radians(60.0),
                    ))
                if nav["forward"] != 0 or nav["right"] != 0:
                    char_forward, char_right = screen_space_ground_axes()
                    rig_translation += char_forward * (forward_step * np.sign(nav["forward"]))
                    rig_translation += char_right * (forward_step * np.sign(nav["right"]))
            poses.append(cam_pose())
        return np.stack(poses, axis=0)


    def _wbench_nav_for_frame(
        self,
        *,
        actions: list[str],
        frame_idx: int,
        action_start_pixel: int,
        frames_per_turn: int,
    ) -> dict[str, float]:
        if frame_idx < action_start_pixel:
            return {"forward": 0.0, "right": 0.0, "yaw": 0.0, "pitch": 0.0}
        action_idx = int(frame_idx - action_start_pixel) // max(1, int(frames_per_turn))
        if action_idx < 0 or action_idx >= len(actions):
            return {"forward": 0.0, "right": 0.0, "yaw": 0.0, "pitch": 0.0}
        return self._wbench_action_to_nav(actions[action_idx])

    @staticmethod
    def _wbench_action_to_nav(action: str) -> dict[str, float]:
        aliases = {
            "forward": "W",
            "backward": "S",
            "cam_left": "left",
            "cam_right": "right",
            "cam_up": "up",
            "cam_down": "down",
            "look_left": "left",
            "look_right": "right",
            "look_up": "up",
            "look_down": "down",
            "pitch_up": "up",
            "pitch_down": "down",
            "yaw_left": "left",
            "yaw_right": "right",
            "stop": "stop",
        }
        single = {
            "W": {"forward": 1.0, "right": 0.0, "yaw": 0.0, "pitch": 0.0},
            "S": {"forward": -1.0, "right": 0.0, "yaw": 0.0, "pitch": 0.0},
            "A": {"forward": 0.0, "right": -1.0, "yaw": 0.0, "pitch": 0.0},
            "D": {"forward": 0.0, "right": 1.0, "yaw": 0.0, "pitch": 0.0},
            "right": {"forward": 0.0, "right": 0.0, "yaw": 1.0, "pitch": 0.0},
            "left": {"forward": 0.0, "right": 0.0, "yaw": -1.0, "pitch": 0.0},
            "up": {"forward": 0.0, "right": 0.0, "yaw": 0.0, "pitch": 1.0},
            "down": {"forward": 0.0, "right": 0.0, "yaw": 0.0, "pitch": -1.0},
            "stop": {"forward": 0.0, "right": 0.0, "yaw": 0.0, "pitch": 0.0},
        }
        out = {"forward": 0.0, "right": 0.0, "yaw": 0.0, "pitch": 0.0}
        raw_parts = str(action).strip().replace(",", "+").split("+")
        for raw in raw_parts:
            part = raw.strip()
            if not part:
                continue
            if part in {"w", "a", "s", "d"}:
                part = part.upper()
            if part in {"->", "→"}:
                part = "right"
            if part in {"<-", "←"}:
                part = "left"
            part = aliases.get(part, part)
            nav = single.get(part)
            if nav is None:
                continue
            for key, value in nav.items():
                out[key] += float(value)
        return out

    @staticmethod
    def _wbench_rot_x(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)

    @staticmethod
    def _wbench_rot_y(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)

    def _save_wbench_output_video(
        self,
        *,
        mode_cfg: ValidationModeConfig,
        mode_dir: Path,
        stem: str,
        pred_latents: list[torch.Tensor],
        metadata: dict[str, Any],
        scheduled_prompt_captions: list[str] | None = None,
        latent_full: torch.Tensor | None = None,
        N: int = 0,
        gap_steps: int = 0,
        cond_end: int = 0,
    ) -> None:
        output_rounds = int(metadata.get("wbench_output_rounds", 0))
        if output_rounds <= 0 or not pred_latents:
            return
        output_rounds = min(output_rounds, len(pred_latents))
        K = int(pred_latents[0].shape[2])
        pred_seq = [p.detach().to(dtype=self.dtype) for p in pred_latents[:output_rounds]]
        prefix_latents = None
        if latent_full is not None:
            sink_count = self.cfg.layout.sink_latent_frames
            hist_end = sink_count + gap_steps + N
            if N > 0:
                keep = max(1, min(int(self.cfg.validation.video_history_latent_frames or N), N))
                keep_start, prefix_end = hist_end - keep, hist_end
            else:
                keep = max(0, cond_end)
                keep_start, prefix_end = hist_end, hist_end + keep
            if keep > 0 and prefix_end <= int(latent_full.shape[2]):
                prefix_latents = latent_full[:, :, keep_start:prefix_end].to(dtype=self.dtype).contiguous()
        if prefix_latents is not None:
            latent = torch.cat([prefix_latents] + pred_seq, dim=2).contiguous()
        else:
            latent = torch.cat(pred_seq, dim=2).contiguous()
        frames = self._decode_latent_to_video_frames(latent)
        if prefix_latents is not None:
            drop = int(prefix_latents.shape[2]) * int(self.cfg.sample.temporal_stride)
            if 0 < drop < int(frames.shape[0]):
                frames = frames[drop:]

        output_dir_raw = getattr(mode_cfg, "wbench_output_dir", None)
        output_dir = Path(output_dir_raw) if output_dir_raw else Path(self.cfg.run.output_dir) / "work_dirs" / "alaya_vigeo" / "videos"
        case_id = str(metadata.get("wbench_case_id") or metadata.get("video_id", stem)).replace("case_", "")
        output_path = output_dir / f"case_{case_id}_combined.mp4"
        self._write_video(output_path, frames)

        chunks_per_turn = int(metadata.get("wbench_chunks_per_turn", 0))
        prompt_schedule = None
        if scheduled_prompt_captions:
            prompt_schedule = [
                {
                    "round": int(r),
                    "turn": int(r // max(1, chunks_per_turn)),
                    "prompt": _strip_camera_blocks(
                        scheduled_prompt_captions[min(r, len(scheduled_prompt_captions) - 1)]
                    ),
                }
                for r in range(int(output_rounds))
            ]

        sidecar = {
            "case_id": case_id,
            "source_stem": stem,
            "mode_dir": str(mode_dir),
            "output_rounds": output_rounds,
            "chunks_per_turn": chunks_per_turn,
            "n_turns": int(metadata.get("wbench_used_n_turns", metadata.get("wbench_n_turns", 0))),
            "nav_only_n_turns": int(metadata.get("wbench_n_turns", 0)),
            "actions": list(metadata.get("wbench_used_actions") or metadata.get("wbench_actions") or []),
            "turn_segments": self._build_wbench_turn_segments(
                metadata=metadata,
                output_rounds=output_rounds,
                K=K,
                num_frames=int(frames.shape[0]),
            ),
            "prompt_schedule": prompt_schedule,
            "perspective": metadata.get("wbench_perspective"),
            "video_path": str(output_path),
            "num_frames": int(frames.shape[0]),
            "fps": int(self.cfg.sample.fps),
        }
        output_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")

    def _build_wbench_turn_segments(
        self,
        *,
        metadata: dict[str, Any],
        output_rounds: int,
        K: int,
        num_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        actions = [str(x) for x in (metadata.get("wbench_used_actions") or metadata.get("wbench_actions") or [])]
        chunks_per_turn = max(1, int(metadata.get("wbench_chunks_per_turn", 0) or 1))
        stride = int(self.cfg.sample.temporal_stride)
        segments: list[dict[str, Any]] = []
        for turn_idx, action in enumerate(actions):
            chunk_start = int(turn_idx * chunks_per_turn)
            if chunk_start >= int(output_rounds):
                break
            chunk_end = min(int(output_rounds), int((turn_idx + 1) * chunks_per_turn))
            latent_start = chunk_start * int(K)
            latent_end = chunk_end * int(K)
            frame_start = latent_start * stride
            frame_end = latent_end * stride
            if num_frames is not None:
                frame_start = min(frame_start, int(num_frames))
                frame_end = min(frame_end, int(num_frames))
            if frame_end <= frame_start:
                break
            segments.append(
                {
                    "turn_index": int(turn_idx),
                    "action": action,
                    "chunk_start": int(chunk_start),
                    "chunk_end_exclusive": int(chunk_end),
                    "chunk_count": int(chunk_end - chunk_start),
                    "latent_start": int(latent_start),
                    "latent_end_exclusive": int(latent_end),
                    "latent_count": int(latent_end - latent_start),
                    "frame_start": int(frame_start),
                    "frame_end_exclusive": int(frame_end),
                    "frame_count": int(frame_end - frame_start),
                    "nav": self._wbench_action_to_nav(action),
                }
            )
        return segments



    def _build_variant_captions(
        self,
        *,
        caption: str,
        prompt_pool: list[str],
        n_variants: int,
        seed_key: int,
        all_random: bool,
    ) -> list[str]:
        """Build the list of raw caption variants.
        all_random=False (default): the sample's own caption plus (N-1) random ones from the pool.
        all_random=True: all N come from the pool, drawn with a fixed seed for reproducibility.
        """
        n_variants = max(1, int(n_variants))
        if n_variants <= 1 and not all_random:
            return [caption]
        if all_random:
            if not prompt_pool:
                return [caption] * n_variants
            others = [p for p in prompt_pool if p and p.strip()]
            random.Random((int(seed_key) + 1) * 100003).shuffle(others)
            if len(others) >= n_variants:
                return others[:n_variants]
            return [others[i % len(others)] for i in range(n_variants)]
        if not prompt_pool:
            return [caption]
        others = [p for p in prompt_pool if p and p.strip() and p.strip() != str(caption).strip()]
        random.Random((int(seed_key) + 1) * 100003).shuffle(others)
        return [caption] + others[: max(0, n_variants - 1)]




    @torch.no_grad()
    def _validate_rollout_sample(
        self,
        *,
        video_pixels: torch.Tensor,
        latent_full: torch.Tensor,
        context: torch.Tensor,
        scheduled_contexts: list[torch.Tensor] | None,
        scheduled_prompt_captions: list[str] | None,
        negative_context: torch.Tensor | None,
        metadata: dict[str, Any],
        mode_cfg: ValidationModeConfig,
        K: int,
        rounds: int,
        N: int,
        gap_steps: int,
        cond_end: int,
    ) -> tuple[
        list[dict[str, Any]],
        list[torch.Tensor],
        list[torch.Tensor | None],
        list[torch.Tensor | None],
        list[torch.Tensor | None],
        list[torch.Tensor | None],
    ]:
        assert self.components is not None

        B, _, _, H_lat, W_lat = latent_full.shape
        sink_count = self.cfg.layout.sink_latent_frames
        explicit_condition = cond_end if N == 0 else 0
        sink_latent = latent_full[:, :, :sink_count].contiguous() if sink_count > 0 else None
        hist_start = sink_count + gap_steps
        hist_end = hist_start + N
        condition_start = hist_end
        target_base_start = condition_start + explicit_condition
        history = latent_full[:, :, hist_start:hist_end].clone().contiguous() if N > 0 else None
        history_action_t_indices = (
            torch.arange(hist_start, hist_end, device=self.dist.device, dtype=torch.float32)
            if bool(getattr(self.cfg.control, "action_history_memory", False)) and N > 0
            else None
        )
        explicit_nearby = (
            latent_full[:, :, condition_start:target_base_start].contiguous()
            if explicit_condition > 0
            else None
        )
        vigeo_prefix_mode = self._uses_vigeo_prefix_last_frame()
        validation_prefix_last_pixel: int | None = None
        if vigeo_prefix_mode:
            prefix_frames = self._vigeo_target_prefix_pixel_frames(
                history_latent_frames=N
            )
            validation_prefix_last_pixel = prefix_frames - 1
            sink_latent = latent_full[:, :, :sink_count].contiguous()
            explicit_nearby = latent_full[
                :, :, condition_start:target_base_start
            ].contiguous()
            motion_start = prefix_frames - self._vigeo_motion_pixel_frames()
            rollout_decode_anchor, rollout_motion_nearby = self._encode_vigeo_motion_window(
                self._slice_video_pixel_frames(video_pixels, motion_start, prefix_frames)
            )
            explicit_nearby = rollout_motion_nearby
        else:
            rollout_decode_anchor = None
            rollout_motion_nearby = None

        sink_indices = (
            self._indices_grid(B, sink_count, H_lat, W_lat, t_offset=self._local_sink_t_offset(N))
            if sink_count > 0
            else None
        )
        sigmas = self._validation_sigmas(latent_frames=int(K))
        stg_perturbations = self._validation_stg_perturbations()
        actual_cfg_scale = self._validation_cfg_scale()
        metrics = []
        pred_latents = []
        nearby_condition_latents: list[torch.Tensor | None] = []
        spatial_condition_latents: list[torch.Tensor | None] = []
        spatial_condition_masks: list[torch.Tensor | None] = []
        spatial_condition_prefix_latents: list[torch.Tensor | None] = []
        spatial_bank = self._init_validation_rollout_spatial_bank(
            video_pixels=video_pixels,
            metadata=metadata,
            target_start=target_base_start,
            history_latent_frames=N,
        )

        for round_idx in range(rounds):
            round_context = context
            if scheduled_contexts:
                round_context = scheduled_contexts[min(round_idx, len(scheduled_contexts) - 1)]
            current_target_start = target_base_start + round_idx * K
            print(
                f"[Rollout] rank={self.dist.rank} src={mode_cfg.dataset.source} "
                f"video={metadata.get('video_id', '?')} round={round_idx + 1}/{rounds} "
                f"target_start={current_target_start}",
                flush=True,
            )
            current_target_rope_t_indices = self._local_target_t_indices(
                K,
                history_latent_frames=N,
                condition_latent_frames=cond_end,
                gap_steps=gap_steps,
            )
            current_target_action_t_indices = torch.arange(
                current_target_start,
                current_target_start + K,
                device=self.dist.device,
                dtype=torch.float32,
            )
            mem_tokens = None
            mem_indices = None
            round_uses_memory = bool(mode_cfg.use_memory) and (
                round_idx >= int(mode_cfg.memory_start_round)
            )
            if history is not None and round_uses_memory:
                assert self.history_encoder is not None
                mem_tokens, mem_indices = self.history_encoder(history)
                mem_indices = mem_indices.clone()
                history_t_offset = self._local_memory_t_offset(N, cond_end)
                mem_indices[:, 0, :, :] += history_t_offset

            if vigeo_prefix_mode:
                if rollout_motion_nearby is None:
                    raise RuntimeError("ViGeo rollout is missing its RGB motion nearby latent")
                nearby_latent = rollout_motion_nearby
            elif cond_end <= 0:
                nearby_latent = None
            elif history is not None:
                nearby_latent = history[:, :, -cond_end:].contiguous()
            elif round_idx == 0:
                nearby_latent = explicit_nearby
            else:
                nearby_latent = pred_latents[-1][:, :, -cond_end:].to(latent_full.dtype).contiguous()
            nearby_indices = (
                self._indices_grid(
                    B, cond_end, H_lat, W_lat,
                    t_offset=self._local_nearby_t_offset(N, cond_end, gap_steps=gap_steps),
                )
                if cond_end > 0
                else None
            )
            _r20_drop = {int(x) for x in (metadata.get("wbench_spatial_drop_rounds") or [])}
            if int(round_idx) in _r20_drop:
                spatial_context = None
            elif spatial_bank is not None:
                spatial_context = self._build_validation_rollout_bank_spatial_context(
                    bank=spatial_bank,
                    metadata=metadata,
                    target_start=current_target_start,
                    K=K,
                    target_rope_t_indices=current_target_rope_t_indices,
                )
            else:
                spatial_context = self._build_spatial_context(
                    latent_full=latent_full,
                    video_pixels=video_pixels,
                    metadata=metadata,
                    target_start=current_target_start,
                    K=K,
                    cond_end=cond_end,
                    target_rope_t_indices=current_target_rope_t_indices,
                )
            force_spatial_invalid = False
            if force_spatial_invalid:
                spatial_context = self._maybe_force_spatial_all_invalid(spatial_context, force=True)
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
            decoder_prefix = nearby_latent
            if vigeo_prefix_mode:
                if rollout_decode_anchor is None or nearby_latent is None:
                    raise RuntimeError("ViGeo rollout is missing its decoder motion prefix")
                decoder_prefix = torch.cat(
                    [rollout_decode_anchor, nearby_latent], dim=2
                ).contiguous()
            nearby_condition_latents.append(
                None
                if decoder_prefix is None
                else decoder_prefix.detach().to(dtype=latent_full.dtype).cpu()
            )
            spatial_condition_latents.append(
                None if spatial_latent is None else spatial_latent.detach().to(dtype=latent_full.dtype).cpu()
            )
            spatial_condition_masks.append(
                None if spatial_mask_patch is None else spatial_mask_patch.detach().cpu()
            )
            spatial_prefix_latent = (
                spatial_context.get("vae_prefix_latent")
                if spatial_context is not None
                else None
            )
            spatial_condition_prefix_latents.append(
                None
                if spatial_prefix_latent is None
                else spatial_prefix_latent.detach().to(dtype=latent_full.dtype).cpu()
            )
            if vigeo_prefix_mode:
                stride = int(self.cfg.sample.temporal_stride)
                target_pixel_start = self._validation_vigeo_target_pixel_start(
                    target_start=current_target_start,
                    target_base_start=target_base_start,
                )
                previous_pixel = (
                    int(validation_prefix_last_pixel)
                    if round_idx == 0 and validation_prefix_last_pixel is not None
                    else target_pixel_start - 1
                )
                control_kwargs = self._build_explicit_pixel_control_kwargs(
                    metadata=metadata,
                    control_modes=list(mode_cfg.control),
                    target_pixel_start=target_pixel_start + stride - 1,
                    target_latent_frames=K,
                    nearby_pixel=previous_pixel,
                    dtype=self.dtype,
                )
            else:
                control_kwargs = self._build_control_kwargs(
                    metadata=metadata,
                    control_modes=list(mode_cfg.control),
                    target_t_indices=current_target_action_t_indices,
                    condition_t_indices=(
                        torch.arange(
                            current_target_start - cond_end,
                            current_target_start,
                            device=self.dist.device,
                            dtype=torch.float32,
                        )
                        if cond_end > 0
                        else None
                    ),
                    history_t_indices=(
                        history_action_t_indices
                        if mem_tokens is not None and bool(mode_cfg.use_memory)
                        else None
                    ),
                    dtype=self.dtype,
                )

            x_t = torch.randn(B, latent_full.shape[1], K, H_lat, W_lat, device=self.dist.device, dtype=self.dtype)
            for sample_step in range(len(sigmas) - 1):
                sigma_now = sigmas[sample_step]
                sigma_next = sigmas[sample_step + 1]

                def _forward_velocity(
                    context_tensor: torch.Tensor,
                    *,
                    control: dict[str, torch.Tensor],
                    perturbations: BatchedPerturbationConfig | None = None,
                ) -> torch.Tensor:
                    return self.components.transformer(
                        x=[x_t.squeeze(0)],
                        t=(sigma_now * 1000.0).view(1).to(device=self.dist.device, dtype=x_t.dtype),
                        context=[context_tensor],
                        seq_len=K * H_lat * W_lat,
                        fps=self.cfg.sample.fps,
                        perturbations=perturbations,
                        history_kv_tokens=mem_tokens,
                        history_indices_grid=mem_indices,
                        gen_t_indices_override=current_target_rope_t_indices,
                        sink_latent=sink_latent,
                        sink_indices_grid=sink_indices,
                        spatial_latent=spatial_latent,
                        spatial_mask_patch=spatial_mask_patch,
                        spatial_indices_grid=spatial_indices,
                        nearby_latent=nearby_latent,
                        nearby_indices_grid=nearby_indices,
                        **control,
                    )

                pos_v = _forward_velocity(round_context, control=control_kwargs)
                pred_v = pos_v

                if actual_cfg_scale > 1.0 and negative_context is not None:
                    neg_v = _forward_velocity(negative_context, control=control_kwargs)
                    pred_v = pred_v + (actual_cfg_scale - 1.0) * (pos_v - neg_v)

                action_cfg_scale = float(mode_cfg.action_cfg_scale)
                if action_cfg_scale > 1.0 and "action_vectors" in control_kwargs:
                    no_action_kwargs: dict[str, torch.Tensor] = {}
                    no_action_pos_v = _forward_velocity(round_context, control=no_action_kwargs)
                    no_action_v = no_action_pos_v
                    if actual_cfg_scale > 1.0 and negative_context is not None:
                        no_action_neg_v = _forward_velocity(negative_context, control=no_action_kwargs)
                        no_action_v = no_action_v + (actual_cfg_scale - 1.0) * (no_action_pos_v - no_action_neg_v)
                    pred_v = no_action_v + action_cfg_scale * (pred_v - no_action_v)

                if stg_perturbations is not None:
                    ptb_v = _forward_velocity(round_context, control=control_kwargs, perturbations=stg_perturbations)
                    pred_v = pred_v + float(self.cfg.validation.stg_scale) * (pos_v - ptb_v)

                if self.cfg.validation.rescale_scale > 0.0 and pred_v is not pos_v:
                    factor = pos_v.float().std() / (pred_v.float().std() + 1e-8)
                    factor = float(self.cfg.validation.rescale_scale) * factor + (1.0 - float(self.cfg.validation.rescale_scale))
                    pred_v = pred_v * factor

                if sigma_next.item() > 1e-5:
                    dt = (sigma_now - sigma_next).to(dtype=x_t.dtype)
                    x_t = x_t - dt * pred_v
                else:
                    x_t = (x_t.float() - pred_v.float() * sigma_now.float()).to(x_t.dtype)

            pred = x_t
            if (
                bool(getattr(self.cfg.validation, "vigeo_seam_dc_correct", False))
                and vigeo_prefix_mode
                and nearby_latent is not None
            ):
                _ref = nearby_latent.to(device=pred.device, dtype=pred.dtype)
                _dc_ref = _ref.mean(dim=(-2, -1), keepdim=True)
                _first = pred[:, :, :1]
                _dc_first = _first.mean(dim=(-2, -1), keepdim=True)
                pred = torch.cat(
                    [_first + (_dc_ref - _dc_first), pred[:, :, 1:]], dim=2
                ).contiguous()
            decoded_target_pixels = None
            next_decode_anchor = None
            next_motion_nearby = None
            if vigeo_prefix_mode:
                if rollout_decode_anchor is None or nearby_latent is None:
                    raise RuntimeError("ViGeo rollout cannot decode without its RGB motion prefix")
                (
                    decoded_target_pixels,
                    next_decode_anchor,
                    next_motion_nearby,
                ) = self._decode_and_reencode_vigeo_motion_chunk(
                    anchor_latent=rollout_decode_anchor,
                    motion_latent=nearby_latent,
                    target_latent=pred.detach(),
                )
            pred_latents.append(pred.detach())
            real_start = current_target_start
            real_end = real_start + K
            gt = latent_full[:, :, real_start:real_end]
            metric: dict[str, Any] = {
                "round": round_idx,
                "drift_frames": int(round_idx * K),
                "gt_latent_frames": int(gt.shape[2]),
                "gt_available": bool(gt.shape[2] == K),
                "nearby_from_rgb_motion_window": bool(vigeo_prefix_mode),
                "vigeo_handoff_mode": str(self.cfg.validation.vigeo_handoff_mode),
                "vigeo_pred_decode_mode": str(
                    self.cfg.validation.vigeo_pred_decode_mode
                ),
                "spatial_condition_available": bool(spatial_latent is not None),
                "spatial_forced_invalid": bool(force_spatial_invalid),
            }
            prompt_label = self._validation_prompt_label(mode_cfg=mode_cfg, round_idx=round_idx)
            if prompt_label is not None:
                metric["prompt_label"] = prompt_label
            if scheduled_prompt_captions:
                metric["prompt"] = scheduled_prompt_captions[min(round_idx, len(scheduled_prompt_captions) - 1)]
            skip_spatial_bank_append = False
            metric["spatial_bank_appended"] = bool(spatial_bank is not None and not skip_spatial_bank_append)
            if gt.shape[2] == K:
                diff = pred.float() - gt.float()
                l2 = diff.pow(2).mean().sqrt().item()
                cos = F.cosine_similarity(pred.float().flatten(), gt.float().flatten(), dim=0).item()
                metric["l2"] = float(l2)
                metric["cos"] = float(cos)
            else:
                metric["l2"] = None
                metric["cos"] = None
            metrics.append(metric)

            if spatial_bank is not None and vigeo_prefix_mode:
                self._record_validation_vigeo_causal_prefix(
                    bank=spatial_bank,
                    decoded_pixels=decoded_target_pixels,
                    target_start=current_target_start,
                )
            if spatial_bank is not None and not skip_spatial_bank_append:
                self._append_validation_rollout_spatial_bank_prediction(
                    bank=spatial_bank,
                    pred_latent=pred.detach(),
                    decoded_pixels=decoded_target_pixels,
                    metadata=metadata,
                    target_start=current_target_start,
                )

            if vigeo_prefix_mode:
                if next_decode_anchor is None or next_motion_nearby is None:
                    raise RuntimeError("ViGeo rollout failed to produce its next RGB motion prefix")
                rollout_decode_anchor = next_decode_anchor
                rollout_motion_nearby = next_motion_nearby

            if history is not None:
                history = torch.cat([history, pred.to(history.dtype)], dim=2)[:, :, -N:].contiguous()
                if history.shape[2] != N:
                    raise RuntimeError(f"validation history length changed: {history.shape[2]} != {N}")
                if history_action_t_indices is not None:
                    history_action_t_indices = torch.cat(
                        [history_action_t_indices, current_target_action_t_indices], dim=0
                    )[-N:].contiguous()

        return (
            metrics,
            pred_latents,
            nearby_condition_latents,
            spatial_condition_latents,
            spatial_condition_masks,
            spatial_condition_prefix_latents,
        )

    def _validation_sigmas(self, *, latent_frames: int | None = None) -> torch.Tensor:
        steps = int(self.cfg.validation.sampling_steps)
        scheduler = self.cfg.validation.scheduler
        if scheduler == "uniform":
            sigmas = torch.linspace(1.0, 0.0, steps + 1)
        elif scheduler == "linear_quadratic":
            sigmas = LinearQuadraticScheduler().execute(steps=steps)
        else:
            if bool(getattr(self.cfg.training, "adaptive_sigma_shift", False)):
                m = self._adaptive_sigma_shift_m(latent_frames or steps)
                shift = math.log(max(float(m), 1e-6))
                max_shift = shift
                base_shift = shift
            else:
                max_shift = 2.05
                base_shift = 0.95
            sigmas = LTX2Scheduler().execute(
                steps=steps,
                latent=None,
                max_shift=max_shift,
                base_shift=base_shift,
                stretch=True,
                terminal=0.1,
            )
        return sigmas.to(device=self.dist.device, dtype=torch.float32)

    def _validation_cfg_scale(self) -> float:
        cfg_scale = float(self.cfg.validation.cfg_scale)
        return 3.0 if cfg_scale <= 1.0 else cfg_scale

    def _validation_stg_perturbations(self) -> BatchedPerturbationConfig | None:
        if self.cfg.validation.stg_scale <= 0.0 or not self.cfg.validation.stg_blocks:
            return None
        return BatchedPerturbationConfig(
            perturbations=[
                PerturbationConfig(
                    perturbations=[
                        Perturbation(
                            type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                            blocks=[int(block) for block in self.cfg.validation.stg_blocks],
                        )
                    ]
                )
            ]
        )

    def _cleanup_after_validation(self) -> None:
        self._cleanup_cuda_cache()

    def _cleanup_cuda_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.dist.device)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _update_validation_metric_history(self, mode_name: str, payload: dict[str, Any]) -> None:
        history_dir = Path(self.cfg.run.output_dir) / "validation" / "metrics"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / f"{mode_name}_rank-{self.dist.rank:03d}.jsonl"
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _plot_validation_metric_history(self, history_dir: Path) -> None:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        records = []
        for history_path in sorted(history_dir.glob("*_rank-*.jsonl")):
            with history_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        if not records:
            return

        values: dict[int, dict[int, dict[str, list[float]]]] = {}
        for record in records:
            step = int(record["step"])
            for metric in record.get("metrics", []):
                if metric.get("cos") is None or metric.get("l2") is None:
                    continue
                round_id = int(metric["round"])
                round_values = values.setdefault(round_id, {}).setdefault(step, {"cos": [], "l2": []})
                round_values["cos"].append(float(metric["cos"]))
                round_values["l2"].append(float(metric["l2"]))

        if not values:
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for round_id in sorted(values):
            steps = sorted(values[round_id])
            cos_values = [
                sum(values[round_id][step]["cos"]) / len(values[round_id][step]["cos"])
                for step in steps
            ]
            l2_values = [
                sum(values[round_id][step]["l2"]) / len(values[round_id][step]["l2"])
                for step in steps
            ]
            label = f"round {round_id}"
            axes[0].plot(steps, cos_values, marker="o", linewidth=1.5, label=label)
            axes[1].plot(steps, l2_values, marker="o", linewidth=1.5, label=label)

        axes[0].set_ylabel("cos")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best", fontsize=8)
        axes[1].set_ylabel("l2")
        axes[1].set_xlabel("validation step")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best", fontsize=8)
        fig.suptitle("Validation Metrics: mean over all test cases")
        fig.tight_layout()
        for stale_path in history_dir.glob("*_metrics.png"):
            stale_path.unlink(missing_ok=True)
        fig.savefig(history_dir / "metrics_mean.png", dpi=160)
        plt.close(fig)

    def _save_validation_videos(
        self,
        *,
        mode_dir: Path,
        stem: str,
        latent_full: torch.Tensor,
        pred_latents: list[torch.Tensor],
        nearby_condition_latents: list[torch.Tensor | None],
        spatial_condition_latents: list[torch.Tensor | None] | None,
        spatial_condition_masks: list[torch.Tensor | None] | None,
        spatial_condition_prefix_latents: list[torch.Tensor | None] | None,
        metadata: dict[str, Any],
        K: int,
        N: int,
        gap_steps: int,
        cond_end: int,
    ) -> None:
        if not pred_latents:
            return
        sink_count = self.cfg.layout.sink_latent_frames
        hist_start = sink_count + gap_steps
        hist_end = hist_start + N
        if N > 0:
            history_keep = self.cfg.validation.video_history_latent_frames or N
            history_keep = max(1, min(int(history_keep), N))
            keep_start = hist_end - history_keep
            prefix_end = hist_end
        else:
            history_keep = max(0, cond_end)
            keep_start = hist_end
            prefix_end = hist_end + history_keep
        n_show = history_keep + len(pred_latents) * K
        gt_latent = latent_full[:, :, keep_start:keep_start + n_show].contiguous()
        pred_latent = torch.cat(
            [latent_full[:, :, keep_start:prefix_end].contiguous()] + [p.to(latent_full.dtype) for p in pred_latents],
            dim=2,
        ).contiguous()
        condition_latent = None
        filled_conditions: list[torch.Tensor] | None = None
        mask_latent = None
        if spatial_condition_latents and len(spatial_condition_latents) == len(pred_latents):
            if any(c is not None for c in spatial_condition_latents):
                filled_conditions = [
                    (
                        c.to(device=latent_full.device, dtype=latent_full.dtype)
                        if c is not None
                        else torch.zeros_like(p.to(device=latent_full.device, dtype=latent_full.dtype))
                    )
                    for c, p in zip(spatial_condition_latents, pred_latents)
                ]
                condition_latent = torch.cat(
                    [latent_full[:, :, keep_start:prefix_end].contiguous()]
                    + filled_conditions,
                    dim=2,
                ).contiguous()
        if condition_latent is not None and spatial_condition_masks and len(spatial_condition_masks) == len(pred_latents):
            B, _, _, H_lat, W_lat = latent_full.shape
            prefix_len = max(0, int(prefix_end) - int(keep_start))
            prefix_mask = torch.ones(
                B,
                1,
                prefix_len,
                H_lat,
                W_lat,
                device=latent_full.device,
                dtype=torch.float32,
            )
            mask_latent = torch.cat(
                [prefix_mask]
                + [
                    (
                        self._spatial_mask_patch_to_grid(
                            m.to(device=latent_full.device),
                            frames=K,
                            height=H_lat,
                            width=W_lat,
                        )
                        if m is not None
                        else torch.zeros(
                            B,
                            1,
                            K,
                            H_lat,
                            W_lat,
                            device=latent_full.device,
                            dtype=torch.float32,
                        )
                    )
                    for m in spatial_condition_masks
                ],
                dim=2,
            ).contiguous()
        use_chunk_i2v_decode = (
            self._uses_vigeo_prefix_last_frame()
            and len(nearby_condition_latents) == len(pred_latents)
            and all(latent is not None for latent in nearby_condition_latents)
        )
        if use_chunk_i2v_decode:
            first_decoder_prefix = nearby_condition_latents[0]
            assert first_decoder_prefix is not None
            gt_targets = latent_full[
                :, :, prefix_end : prefix_end + len(pred_latents) * K
            ].contiguous()
            gt_decode_latent = torch.cat(
                [
                    first_decoder_prefix.to(
                        device=gt_targets.device, dtype=gt_targets.dtype
                    ),
                    gt_targets,
                ],
                dim=2,
            ).contiguous()
            gt_frames = self._decode_latent_to_video_frames(gt_decode_latent)
            # Keep only the last input-prefix RGB frame plus the GT target stream.
            gt_frames = gt_frames[self._vigeo_motion_pixel_frames() - 1 :]
        else:
            gt_frames = self._decode_latent_to_video_frames(gt_latent)
        if (
            use_chunk_i2v_decode
            and self.cfg.validation.vigeo_pred_decode_mode == "full"
        ):
            first_decoder_prefix = nearby_condition_latents[0]
            assert first_decoder_prefix is not None
            full_decode_latent = torch.cat(
                [
                    first_decoder_prefix.to(
                        device=latent_full.device,
                        dtype=latent_full.dtype,
                    )
                ]
                + [
                    p.to(device=latent_full.device, dtype=latent_full.dtype)
                    for p in pred_latents
                ],
                dim=2,
            ).contiguous()
            pred_frames = self._decode_latent_to_video_frames(full_decode_latent)
            # Keep the latest input-context frame, followed by all generated frames.
            pred_frames = pred_frames[self._vigeo_motion_pixel_frames() - 1 :]
        elif use_chunk_i2v_decode:
            pred_frames = self._decode_i2v_rollout_chunks_to_video_frames(
                nearby_latents=nearby_condition_latents,
                chunks=pred_latents,
            )
        else:
            pred_frames = self._decode_latent_to_video_frames(pred_latent)
        condition_frames = None
        use_spatial_chunk_decode = (
            use_chunk_i2v_decode
            and filled_conditions is not None
            and spatial_condition_prefix_latents is not None
            and len(spatial_condition_prefix_latents) == len(filled_conditions)
            and all(prefix is not None for prefix in spatial_condition_prefix_latents)
        )
        if use_spatial_chunk_decode:
            condition_frames = self._decode_i2v_rollout_chunks_to_video_frames(
                nearby_latents=spatial_condition_prefix_latents,
                chunks=filled_conditions,
            )
        elif condition_latent is not None:
            condition_frames = self._decode_latent_to_video_frames(condition_latent)
        mask_frames = None
        if mask_latent is not None:
            mask_frames = self._mask_latent_to_video_frames(
                mask_latent,
                like_frames=condition_frames if condition_frames is not None else pred_frames,
            )
        self._write_video(mode_dir / f"{stem}_pred_clean.mp4", pred_frames)
        if self.cfg.validation.save_joystick:
            gt_label_indices = torch.arange(
                keep_start,
                keep_start + int(gt_latent.shape[2]),
                dtype=torch.long,
            )
            pred_label_indices = torch.cat(
                [
                    torch.arange(keep_start, prefix_end, dtype=torch.long),
                    torch.arange(prefix_end, prefix_end + len(pred_latents) * K, dtype=torch.long),
                ],
                dim=0,
            )
            gt_labels = self._build_joystick_labels(
                metadata=metadata,
                latent_indices=gt_label_indices,
            )
            pred_labels = self._build_joystick_labels(
                metadata=metadata,
                latent_indices=pred_label_indices,
            )
            if gt_labels is not None:
                gt_frames = self._add_joystick_overlay(gt_frames, gt_labels)
            if pred_labels is not None:
                pred_frames = self._add_joystick_overlay(pred_frames, pred_labels)
            if condition_frames is not None and pred_labels is not None:
                condition_frames = self._add_joystick_overlay(condition_frames, pred_labels)

        comparison_count = int(pred_frames.shape[0])

        def _fit_or_black(frames: torch.Tensor | None, *, like: torch.Tensor) -> torch.Tensor:
            if frames is None:
                return torch.zeros(
                    (comparison_count, *like.shape[1:]),
                    dtype=like.dtype,
                    device=like.device,
                )
            frames = frames[:comparison_count]
            if int(frames.shape[0]) >= comparison_count:
                return frames
            pad = torch.zeros(
                (comparison_count - int(frames.shape[0]), *like.shape[1:]),
                dtype=frames.dtype,
                device=frames.device,
            )
            return torch.cat([frames, pad], dim=0)

        gt_frames = _fit_or_black(gt_frames, like=pred_frames)
        condition_frames = _fit_or_black(condition_frames, like=pred_frames)
        mask_frames = _fit_or_black(mask_frames, like=pred_frames)
        comparison_parts = [gt_frames, pred_frames, condition_frames, mask_frames]
        comparison = torch.cat(comparison_parts, dim=2)
        self._write_video(mode_dir / f"{stem}_comparison.mp4", comparison)

    def _decode_i2v_rollout_chunks_to_video_frames(
        self,
        *,
        nearby_latents: list[torch.Tensor | None],
        chunks: list[torch.Tensor],
    ) -> torch.Tensor:
        if not chunks or len(nearby_latents) != len(chunks):
            raise ValueError(
                "i2v rollout decode requires one nearby latent per target chunk"
            )
        _dump_dir = os.environ.get("ALAYA_DUMP_PRED_LATENTS", "")
        if _dump_dir:
            os.makedirs(_dump_dir, exist_ok=True)
            _tag = f"rank{self.dist.rank}_{len(os.listdir(_dump_dir))}"
            torch.save(
                [c.detach().to("cpu", torch.float32) for c in chunks],
                os.path.join(_dump_dir, f"pred_chunks_{_tag}.pt"),
            )
        decoded_chunks: list[torch.Tensor] = []
        stride = int(self.cfg.sample.temporal_stride)
        for index, (nearby, chunk) in enumerate(zip(nearby_latents, chunks)):
            if nearby is None or int(nearby.shape[2]) <= 0:
                raise ValueError(
                    f"i2v rollout chunk {index} requires a non-empty decoder prefix"
                )
            prefix_latents = int(nearby.shape[2])
            context_pixels = 1 + (prefix_latents - 1) * stride
            local_latent = torch.cat(
                [nearby.to(dtype=chunk.dtype, device=chunk.device), chunk],
                dim=2,
            ).contiguous()
            local_frames = self._decode_latent_to_video_frames(local_latent)
            expected_frames = context_pixels + int(chunk.shape[2]) * stride
            if int(local_frames.shape[0]) != expected_frames:
                raise RuntimeError(
                    f"i2v rollout chunk {index} decoded {local_frames.shape[0]} frames, "
                    f"expected {expected_frames}"
                )
            # The decoder sees nine RGB context frames, but saved videos retain
            # only the latest context frame for the first chunk and no duplicate
            # context frames for later chunks.
            crop = context_pixels - 1 if index == 0 else context_pixels
            decoded_chunks.append(local_frames[crop:])
        return torch.cat(decoded_chunks, dim=0)

    def _write_validation_prompt_file(
        self,
        path: Path,
        *,
        payload: dict[str, Any],
        metadata: dict[str, Any],
        prompt_caption: str,
        caption: str,
        actual_cfg_scale: float,
        K: int,
        cond_end: int,
    ) -> None:
        video_id = metadata.get("video_id", "unknown")
        source = metadata.get("source", "")
        caption_type = metadata.get("caption_type", "")
        num_frames = (K + cond_end - 1) * self.cfg.sample.temporal_stride + 1
        lines = [
            f"Step: {payload['step']}",
            f"Rank: {payload['rank']}",
            f"Video ID: {video_id}",
            f"Prompt: {prompt_caption}",
            f"Original Caption: {caption}",
            f"Negative Prompt: {self.cfg.validation.negative_prompt}",
            "Distilled Mode: False",
            f"Actual CFG Scale: {actual_cfg_scale}",
            f"Sampling Steps: {self.cfg.validation.sampling_steps}",
            f"Num Frames: {num_frames}",
            f"Rollout Rounds: {payload.get('actual_rollout_rounds', payload.get('rollout_rounds', 1))}",
            f"Requested Rollout Rounds: {payload.get('requested_rollout_rounds', payload.get('rollout_rounds', 1))}",
            f"Generated Rollout Rounds: {payload.get('generated_rollout_rounds', payload.get('rollout_rounds', 1))}",
            f"Local GT Rollout Rounds: {payload.get('local_gt_rollout_rounds', payload.get('local_rollout_rounds', payload.get('rollout_rounds', 1)))}",
            f"Resolution: {self.cfg.sample.width}x{self.cfg.sample.height}",
            f"i2v Conditioning: {payload['condition'] == 'i2v'}",
            f"Conditioning Latent Frames: {cond_end}",
            f"Source: {source}",
            f"Caption Type: {caption_type}",
            f"Camera Injected: {'action' in payload['control']}",
            f"Memory Enabled: {payload.get('use_memory', True)}",
            f"Action CFG Scale: {payload.get('action_cfg_scale', 1.0)}",
            f"Inference Mode: {payload['condition']}",
            f"Condition Frames: {cond_end}",
            f"Task Type: {payload['condition']}",
        ]
        prompt_schedule = payload.get("prompt_schedule")
        if prompt_schedule:
            lines.append("Prompt Schedule:")
            for idx, prompt in enumerate(prompt_schedule):
                lines.append(f"  chunk{idx}: {prompt}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _build_joystick_labels(
        self,
        *,
        metadata: dict[str, Any],
        latent_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        if not _as_bool(metadata.get("has_camera", False)):
            rank0_print(self.dist, "[Validation]", "joystick skipped: sample has no camera")
            return None
        c2w = metadata.get("cam_c2w_raw")
        if c2w is None:
            c2w = metadata.get("cam_c2w")
        if c2w is None:
            rank0_print(self.dist, "[Validation]", "joystick skipped: missing cam_c2w")
            return None

        from ltx2.modules.camera_control import c2w_to_action_labels

        c2w_np = _to_numpy(c2w)
        if c2w_np.ndim == 4:
            c2w_np = c2w_np[0]
        labels = c2w_to_action_labels(
            c2w_np,
            vae_temporal_stride=int(self.cfg.sample.temporal_stride),
            classify_mode="threshold",
        )
        indices = latent_indices.to(dtype=torch.long).clamp(min=0, max=labels.numel() - 1)
        return labels.index_select(0, indices.cpu()).cpu()

    def _add_joystick_overlay(self, frames: torch.Tensor, action_labels: torch.Tensor) -> torch.Tensor:
        from ltx2.modules.camera_control import add_joystick_overlay

        frame_list = [frame.numpy() for frame in frames.cpu()]
        overlaid = add_joystick_overlay(
            frame_list,
            action_labels,
            vae_temporal_stride=int(self.cfg.sample.temporal_stride),
            smooth_alpha=0.3,
        )
        return torch.from_numpy(np.stack(overlaid, axis=0)).to(torch.uint8)

    def _spatial_mask_patch_to_grid(
        self,
        mask_patch: torch.Tensor,
        *,
        frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        patchifier = VideoLatentPatchifier(patch_size=1)
        pt, ph, pw = patchifier.patch_size
        if int(frames) % pt != 0 or int(height) % ph != 0 or int(width) % pw != 0:
            raise ValueError(
                f"mask target shape {(frames, height, width)} is not divisible by patch size {(pt, ph, pw)}"
            )
        grid_frames = int(frames) // pt
        grid_height = int(height) // ph
        grid_width = int(width) // pw
        expected_tokens = grid_frames * grid_height * grid_width
        if mask_patch.dim() != 3 or int(mask_patch.shape[1]) != expected_tokens:
            raise ValueError(
                f"mask_patch must be [B,{expected_tokens},C_patch], got {tuple(mask_patch.shape)}"
            )
        mask = mask_patch[..., 0].to(device=self.dist.device, dtype=torch.float32)
        mask = mask.reshape(int(mask_patch.shape[0]), grid_frames, grid_height, grid_width)
        if pt > 1:
            mask = mask.repeat_interleave(pt, dim=1)
        if ph > 1:
            mask = mask.repeat_interleave(ph, dim=2)
        if pw > 1:
            mask = mask.repeat_interleave(pw, dim=3)
        return mask[:, None, : int(frames), : int(height), : int(width)].clamp_(0.0, 1.0)

    def _mask_latent_to_video_frames(self, mask_latent: torch.Tensor, *, like_frames: torch.Tensor) -> torch.Tensor:
        target_frames, target_height, target_width = [int(x) for x in like_frames.shape[:3]]
        mask = mask_latent[:1, :1].to(device=self.dist.device, dtype=torch.float32)
        mask = F.interpolate(mask, size=(target_frames, target_height, target_width), mode="nearest")
        frames = mask.squeeze(0).permute(1, 2, 3, 0).contiguous()
        frames = frames.expand(-1, -1, -1, 3)
        return (frames * 255.0).round().clamp(0, 255).to(torch.uint8).cpu()

    def _decode_latent_to_video_frames(self, latent: torch.Tensor) -> torch.Tensor:
        assert self.components is not None
        decoder = self.components.vae_decoder
        decode_chunk = self.cfg.runtime.vae_decode_chunk_latents
        chunk_latents = max(1, int(decode_chunk if decode_chunk is not None else self.cfg.runtime.vae_chunk_size))
        frames = []
        total_latents = int(latent.shape[2])
        for start in range(0, total_latents, chunk_latents):
            end = min(total_latents, start + chunk_latents)
            chunk = latent[:, :, start:end].to(device=self.dist.device, dtype=self.dtype)
            with torch.no_grad():
                pixel = decoder(chunk)
            pixel = (pixel * 0.5 + 0.5).clamp(0, 1)
            chunk_frames = pixel.squeeze(0).permute(1, 2, 3, 0).contiguous()
            if start > 0 and chunk_frames.shape[0] > 0:
                chunk_frames = chunk_frames[1:]
            frames.append((chunk_frames * 255.0).to(torch.uint8).cpu())
            del chunk, pixel, chunk_frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(frames, dim=0)

    def _decode_latent_to_bank_pixels(self, latent: torch.Tensor) -> torch.Tensor:
        assert self.components is not None
        chunk = latent.to(device=self.dist.device, dtype=self.dtype)
        with torch.no_grad():
            pixel = self.components.vae_decoder(chunk)
        return pixel.detach().to(device=self.dist.device, dtype=self.dtype).contiguous()

    def _encode_vigeo_motion_window(
        self,
        video_pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        motion_pixels = self._vigeo_motion_pixel_frames()
        actual_pixels = self._video_pixel_frame_count(video_pixels)
        if actual_pixels != motion_pixels:
            raise ValueError(
                f"ViGeo motion window needs {motion_pixels} pixels, got {actual_pixels}"
            )
        latent = self._encode_video(video_pixels, needed_latents=2)
        if int(latent.shape[2]) != 2:
            raise RuntimeError(
                f"ViGeo motion window VAE produced {latent.shape[2]} latents, expected 2"
            )
        return latent[:, :, :1].contiguous(), latent[:, :, 1:2].contiguous()

    def _decode_and_reencode_vigeo_motion_chunk(
        self,
        *,
        anchor_latent: torch.Tensor,
        motion_latent: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(anchor_latent.shape[2]) != 1 or int(motion_latent.shape[2]) != 1:
            raise ValueError(
                "ViGeo motion decode requires one anchor and one motion latent"
            )
        continuation = torch.cat(
            [anchor_latent, motion_latent, target_latent], dim=2
        ).contiguous()
        pixels = self._decode_latent_to_bank_pixels(continuation)
        motion_pixels = self._vigeo_motion_pixel_frames()
        expected_frames = motion_pixels + int(target_latent.shape[2]) * int(
            self.cfg.sample.temporal_stride
        )
        if int(pixels.shape[2]) != expected_frames:
            raise RuntimeError(
                "ViGeo motion continuation decode length mismatch: "
                f"got {pixels.shape[2]}, expected {expected_frames}"
            )
        target_pixels = pixels[:, :, motion_pixels:].contiguous()
        handoff_mode = str(self.cfg.validation.vigeo_handoff_mode)
        if handoff_mode == "rgb_reencode":
            next_anchor, next_motion = self._encode_vigeo_motion_window(
                pixels[:, :, -motion_pixels:].contiguous()
            )
        elif handoff_mode == "direct_latent":
            # DiT receives next_motion; both latents remain available as the
            # online chunk decoder prefix used to produce RGB for the bank.
            next_anchor = continuation[:, :, -2:-1].contiguous()
            next_motion = continuation[:, :, -1:].contiguous()
        else:
            raise ValueError(f"unsupported ViGeo handoff mode: {handoff_mode}")
        return target_pixels, next_anchor, next_motion

    def _build_vigeo_validation_latent_full(
        self,
        *,
        video_pixels: torch.Tensor,
        metadata: dict[str, Any],
        required_latents: int,
        target_base_start: int,
        history_latent_frames: int,
        allow_short: bool,
        allow_empty_target: bool = False,
    ) -> torch.Tensor:
        """Pack sink, optional history, and the GT target continuation."""
        sink_count = int(self.cfg.layout.sink_latent_frames)
        history_latents = max(0, int(history_latent_frames))
        prefix_frames = self._vigeo_target_prefix_pixel_frames(
            history_latent_frames=history_latents
        )
        expected_target_base = sink_count + (
            history_latents if history_latents > 0 else 1
        )
        if sink_count != 1 or int(target_base_start) != expected_target_base:
            raise ValueError(
                "ViGeo validation packing requires one sink followed by history "
                "or one explicit nearby latent: "
                f"expected target_base_start={expected_target_base}, "
                f"got sink={sink_count}, history={history_latents}, "
                f"target_base_start={target_base_start}"
            )
        if prefix_frames < 1:
            raise ValueError("ViGeo validation requires at least one prefix pixel frame")

        target_latents = max(1, int(required_latents) - int(target_base_start))
        stride = int(self.cfg.sample.temporal_stride)
        target_pixels = target_latents * stride
        available_pixels = self._video_pixel_frame_count(video_pixels)
        requested_pixel_end = prefix_frames + target_pixels
        if available_pixels < requested_pixel_end and bool(
            getattr(self.cfg.layout, "variable_length", False)
        ):
            video_bcfhw = self._video_pixels_to_bcfhw(video_pixels)
            pad = requested_pixel_end - available_pixels
            last = video_bcfhw[:, :, -1:].expand(-1, -1, pad, -1, -1)
            video_pixels = torch.cat([video_bcfhw, last], dim=2)
            available_pixels = requested_pixel_end
        target_pixel_end = requested_pixel_end
        if available_pixels < requested_pixel_end and not allow_short:
            raise ValueError(
                "ViGeo validation target stream is too short: "
                f"need pixels [0,{requested_pixel_end}), got {available_pixels}"
            )
        if available_pixels < requested_pixel_end:
            available_target_pixels = available_pixels - prefix_frames
            compatible_target_pixels = stride * (available_target_pixels // stride)
            if compatible_target_pixels <= 0:
                if not allow_empty_target:
                    raise ValueError(
                        "ViGeo validation clip has no complete target latent after its prefix: "
                        f"prefix={prefix_frames}, available={available_pixels}"
                    )
                compatible_target_pixels = 0
            target_pixel_end = prefix_frames + compatible_target_pixels

        metadata["validation_video_available_frames"] = int(available_pixels)
        metadata["validation_video_requested_frames"] = int(requested_pixel_end)
        metadata["validation_video_used_frames"] = int(target_pixel_end)

        video_key = str(metadata.get("video_id", ""))
        sink_pixel_index = sum(video_key.encode("utf-8")) % prefix_frames
        sink_latent = self._encode_video(
            self._slice_video_pixel_frames(
                video_pixels, sink_pixel_index, sink_pixel_index + 1
            ),
            needed_latents=1,
        )
        history_latent = None
        if history_latents > 0:
            history_latent = self._encode_video(
                self._slice_video_pixel_frames(video_pixels, 0, prefix_frames),
                needed_latents=history_latents,
            )
            if int(history_latent.shape[2]) != history_latents:
                raise RuntimeError(
                    "ViGeo validation history VAE length mismatch: "
                    f"got {history_latent.shape[2]}, expected {history_latents}"
                )
        used_target_latents = (target_pixel_end - prefix_frames) // stride
        motion_pixels = self._vigeo_motion_pixel_frames()
        continuation_latent = self._encode_video(
            self._slice_video_pixel_frames(
                video_pixels, prefix_frames - motion_pixels, target_pixel_end
            ),
            needed_latents=used_target_latents + 2,
        )
        if int(continuation_latent.shape[2]) != used_target_latents + 2:
            raise RuntimeError(
                "ViGeo validation continuation VAE length mismatch: "
                f"got {continuation_latent.shape[2]}, expected {used_target_latents + 2}"
            )
        nearby_latent = continuation_latent[:, :, 1:2].contiguous()
        target_latent = continuation_latent[:, :, 2:].contiguous()
        prefix_latent = history_latent if history_latent is not None else nearby_latent
        return torch.cat([sink_latent, prefix_latent, target_latent], dim=2).contiguous()

    def _validation_vigeo_target_pixel_start(
        self,
        *,
        target_start: int,
        target_base_start: int | None = None,
    ) -> int:
        if target_base_start is None:
            history_latents = int(self.cfg.layout.history_latent_frames)
            target_base_start = int(self.cfg.layout.sink_latent_frames) + (
                history_latents if history_latents > 0 else 1
            )
        return self._vigeo_target_prefix_pixel_frames() + (
            int(target_start) - int(target_base_start)
        ) * int(self.cfg.sample.temporal_stride)

    def _init_validation_rollout_spatial_bank(
        self,
        *,
        video_pixels: torch.Tensor,
        metadata: dict[str, Any],
        target_start: int,
        history_latent_frames: int,
    ) -> _RolloutSpatialBank | None:
        cfg = self.cfg.spatial_memory
        context_mode = str(getattr(cfg, "context_mode", "retrieval"))
        if not bool(cfg.enabled) or context_mode not in {
            "target_prefix_pixels",
            "vigeo_prefix_last_frame",
        }:
            return None
        if not _as_bool(metadata.get("has_camera", False)):
            return None
        cam_c2w = metadata.get("cam_c2w")
        intrinsic = metadata.get("intrinsic")
        if cam_c2w is None or intrinsic is None:
            return None

        stride = int(self.cfg.sample.temporal_stride)
        history_pixels = (
            self._vigeo_prefix_pixel_frames()
            if context_mode == "vigeo_prefix_last_frame"
            else max(1, int(cfg.num_context_frames))
        )
        fixed_single_frame_scale = (
            context_mode == "vigeo_prefix_last_frame"
            and str(metadata.get("source") or "").lower() in {"arena", "wbench_navi", "custom_i2v"}
        )
        if context_mode == "vigeo_prefix_last_frame":
            target_pixel_start = self._vigeo_target_prefix_pixel_frames(
                history_latent_frames=history_latent_frames
            )
            source_indices = (
                [target_pixel_start - 1]
                if fixed_single_frame_scale
                else self._vigeo_scale_context_pixel_indices(
                    history_latent_frames=history_latent_frames
                )
            )
        else:
            target_pixel_start = int(target_start) * stride
            source_floor = 0
            if not bool(cfg.include_sink):
                source_floor = max(0, int(self.cfg.layout.sink_latent_frames) * stride)
            source_start = max(source_floor, target_pixel_start - history_pixels)
            source_indices = list(range(source_start, target_pixel_start))

        video_frames = self._video_pixel_frame_count(video_pixels)
        cam_frames = int(cam_c2w.shape[1] if cam_c2w.dim() == 4 else cam_c2w.shape[0])
        max_frames = min(video_frames, cam_frames)
        source_indices = [idx for idx in source_indices if 0 <= int(idx) < max_frames]
        if not source_indices:
            return None
        if (
            not fixed_single_frame_scale
            and bool(getattr(cfg, "require_full_context", True))
            and len(source_indices) < history_pixels
        ):
            return None

        pixels = self._select_video_pixel_frames(video_pixels, source_indices)
        if context_mode == "vigeo_prefix_last_frame":
            geometry = self._get_vigeo_geometry().infer_stream_geometry(
                video_pixels=pixels,
                kv_caches=None,
                reset_cache=True,
                chunk_size=int(cfg.vigeo_stream_chunk_size),
                total_budget=int(cfg.vigeo_cache_budget),
            )
            if fixed_single_frame_scale:
                scale = float(cfg.vigeo_single_frame_scale)
                pairwise_scales: tuple[float, ...] = ()
            else:
                scale, pairwise_scales = self._get_vigeo_geometry().estimate_translation_scale(
                    predicted_poses=geometry.predicted_poses,
                    cam_c2w=cam_c2w,
                    frame_indices=source_indices,
                    fallback_scale=float(cfg.vigeo_single_frame_scale),
                )
            stored_pixels = pixels.detach().to(device="cpu", dtype=torch.float16)
            subject_layers = self._wbench_subject_foreground_layers(
                source_pixel=stored_pixels[:, :, -1],
                metadata=metadata,
            )
            if subject_layers is None:
                subject_anchor_pixel = None
                subject_anchor_mask = None
                subject_exclusion_mask = None
            else:
                subject_anchor_pixel, subject_anchor_mask, subject_exclusion_mask = subject_layers
            return _RolloutSpatialBank(
                pixels=[stored_pixels[:, :, i].contiguous() for i in range(int(stored_pixels.shape[2]))],
                frame_indices=[int(i) for i in source_indices],
                depths=[None for _ in source_indices],
                vigeo_pointmaps=[
                    pointmap.to(dtype=torch.float16).contiguous()
                    for pointmap in geometry.pointmaps
                ],
                vigeo_valid_masks=[
                    self._exclude_subject_from_valid_mask(mask, subject_exclusion_mask)
                    for mask in geometry.valid_masks
                ],
                vigeo_predicted_poses=[pose.float().contiguous() for pose in geometry.predicted_poses],
                vigeo_intrinsics=[K.float().contiguous() for K in geometry.intrinsics],
                vigeo_kv_caches=geometry.kv_caches,
                vigeo_scale=float(scale),
                vigeo_pairwise_scales=tuple(pairwise_scales),
                vigeo_generated_chunks=0,
                vigeo_pixel_offset=int(target_pixel_start) - int(target_start) * stride,
                subject_anchor_pixel=subject_anchor_pixel,
                subject_anchor_mask=subject_anchor_mask,
                subject_exclusion_mask=subject_exclusion_mask,
            )

        depth_by_local = self._infer_validation_bank_depths(
            pixels=pixels,
            metadata=metadata,
            frame_indices=source_indices,
        )
        return _RolloutSpatialBank(
            pixels=[pixels[:, :, i].detach().contiguous() for i in range(int(pixels.shape[2]))],
            frame_indices=[int(i) for i in source_indices],
            depths=[depth_by_local.get(i) if depth_by_local is not None else None for i in range(int(pixels.shape[2]))],
        )

    def _append_validation_rollout_spatial_bank_prediction(
        self,
        *,
        bank: _RolloutSpatialBank,
        pred_latent: torch.Tensor,
        decoded_pixels: torch.Tensor | None,
        metadata: dict[str, Any],
        target_start: int,
    ) -> None:
        cam_c2w = metadata.get("cam_c2w")
        if cam_c2w is None:
            return
        stride = int(self.cfg.sample.temporal_stride)
        if self._uses_vigeo_prefix_last_frame():
            if decoded_pixels is None:
                raise RuntimeError(
                    "ViGeo rollout bank append requires the already decoded target pixels"
                )
            pixels = decoded_pixels
        else:
            pixels = self._decode_latent_to_bank_pixels(pred_latent)
        frame_count = int(pixels.shape[2])
        if self._uses_vigeo_prefix_last_frame():
            cfg = self.cfg.spatial_memory
            if not bank.vigeo_intrinsics:
                raise RuntimeError("ViGeo rollout bank has no initialized intrinsic")
            target_pixel_start = int(target_start) * stride + int(bank.vigeo_pixel_offset)
            frame_indices = list(range(target_pixel_start, target_pixel_start + frame_count))
            geometry = self._get_vigeo_geometry().infer_stream_geometry(
                video_pixels=pixels,
                kv_caches=bank.vigeo_kv_caches,
                shared_intrinsic=bank.vigeo_intrinsics[0],
                reset_cache=False,
                chunk_size=int(cfg.vigeo_stream_chunk_size),
                total_budget=int(cfg.vigeo_cache_budget),
            )
            if int(geometry.pointmaps.shape[0]) != frame_count:
                raise RuntimeError(
                    "ViGeo rollout append length mismatch: "
                    f"geometry={geometry.pointmaps.shape[0]} decoded={frame_count}"
                )
            _dyn_masks = None
            if bool(getattr(cfg, "vigeo_dynamic_mask", False)) and len(bank.pixels) > 0:
                try:
                    with torch.no_grad():
                        _h, _w = int(pixels.shape[-2]), int(pixels.shape[-1])
                        _src_idx = len(bank.pixels) - 1
                        _cam4 = cam_c2w if cam_c2w.dim() == 4 else cam_c2w.unsqueeze(0)
                        _src_all_c2w = self._validation_vigeo_bank_source_c2w(
                            bank=bank, cam_c2w=_cam4
                        )
                        _sel = torch.tensor([_src_idx], device=_src_all_c2w.device, dtype=torch.long)
                        _tgt_c2w, _tgt_K = self._validation_vigeo_target_cameras(
                            bank=bank, metadata=metadata, cam_c2w=_cam4,
                            intrinsic=metadata["intrinsic"],
                            target_pixel_indices=[int(i) for i in frame_indices],
                            height=_h, width=_w,
                        )
                        _wr = render_colored_pointmaps_to_camera_targets(
                            source_pixels=bank.pixels[_src_idx].unsqueeze(2).to(
                                device=self.dist.device, dtype=self.dtype
                            ),
                            source_pointmaps=bank.vigeo_pointmaps[_src_idx].unsqueeze(0).unsqueeze(0).to(
                                device=self.dist.device, dtype=torch.float32
                            ) * float(bank.vigeo_scale),
                            source_valid_masks=bank.vigeo_valid_masks[_src_idx].unsqueeze(0).unsqueeze(0).to(
                                device=self.dist.device, dtype=torch.bool
                            ),
                            source_c2w=_src_all_c2w.index_select(1, _sel),
                            target_c2w=_tgt_c2w,
                            target_intrinsic=_tgt_K,
                            height=_h, width=_w,
                            depth_threshold=min(float(cfg.retrieval_depth_threshold), 1e-3),
                            fill_value=None,
                            return_coverage=True,
                        )
                        if _wr is not None:
                            _warped, _cov = _wr
                            _px = pixels.to(device=_warped.device, dtype=torch.float32)
                            _res = (
                                (_warped.float() - _px).abs().mean(dim=1)
                            ).reshape(frame_count, _h, _w)
                            _covf = _cov.float().reshape(frame_count, _h, _w)
                            _tau = float(getattr(cfg, "vigeo_dynamic_mask_threshold", 0.25))
                            _motion = ((_res > _tau) & (_covf > 0.5)).float()
                            _motion = F.max_pool2d(
                                _motion.unsqueeze(1), kernel_size=7, stride=1, padding=3
                            ).squeeze(1) > 0.5
                            _dyn_masks = _motion.to(device="cpu")
                except Exception as _e:
                    _dyn_masks = None
                    print(f"[vigeo_dynamic_mask] skip (fail-open): {_e}", flush=True)

            stored_pixels = pixels.detach().to(device="cpu", dtype=torch.float16)
            if (
                _as_bool(metadata.get("wbench_subject_anchor_refresh", False))
                and bank.subject_anchor_pixel is not None
                and bank.subject_anchor_mask is not None
                and int(stored_pixels.shape[2]) > 0
            ):
                try:
                    _a = float(metadata.get("wbench_subject_anchor_refresh_alpha", 1.0) or 1.0)
                    _new = stored_pixels[:, :, -1]
                    if 0.0 < _a < 1.0 and bank.subject_anchor_pixel is not None \
                       and bank.subject_anchor_pixel.shape == _new.shape:
                        bank.subject_anchor_pixel = (
                            _a * _new.float() + (1.0 - _a) * bank.subject_anchor_pixel.float()
                        ).to(dtype=_new.dtype).contiguous()
                    else:
                        bank.subject_anchor_pixel = _new.contiguous()
                except Exception as _e:
                    print(f"[subject_anchor_refresh] skip (fail-open): {_e}", flush=True)
            if (
                _as_bool(metadata.get("wbench_subject_anchor_adaptive", False))
                and bank.subject_anchor_pixel is not None
                and bank.subject_anchor_mask is not None
                and int(stored_pixels.shape[2]) > 0
            ):
                try:
                    _latest = stored_pixels[:, :, -1].float()
                    _anchor = bank.subject_anchor_pixel.float()
                    _m = bank.subject_anchor_mask.to(dtype=torch.bool)
                    if _anchor.shape == _latest.shape and _m.any():
                        _mae = float(
                            (_latest - _anchor).abs()[..., _m].mean().item()
                        ) / 2.0
                        _thr = float(metadata.get("wbench_subject_anchor_lost_mae", 0.10) or 0.10)
                        bank.subject_anchor_paste = _mae >= _thr
                except Exception as _e:
                    bank.subject_anchor_paste = True
                    print(f"[subject_anchor_adaptive] fail-open: {_e}", flush=True)
            for local_idx, frame_idx in enumerate(frame_indices):
                bank.pixels.append(stored_pixels[:, :, local_idx].contiguous())
                bank.frame_indices.append(int(frame_idx))
                bank.depths.append(None)
                bank.vigeo_pointmaps.append(
                    geometry.pointmaps[local_idx].to(dtype=torch.float16).contiguous()
                )
                _vm = geometry.valid_masks[local_idx].bool()
                if _dyn_masks is not None:
                    _vm = _vm & (~_dyn_masks[local_idx].to(device=_vm.device).view_as(_vm))
                _vm = self._exclude_subject_from_valid_mask(
                    _vm, bank.subject_exclusion_mask
                )
                bank.vigeo_valid_masks.append(
                    _vm.contiguous()
                )
                bank.vigeo_predicted_poses.append(
                    geometry.predicted_poses[local_idx].float().contiguous()
                )
                bank.vigeo_intrinsics.append(
                    geometry.intrinsics[local_idx].float().contiguous()
                )
            bank.vigeo_kv_caches = geometry.kv_caches
            bank.vigeo_generated_chunks += 1
            return

        frame_indices = list(range(int(target_start) * stride, int(target_start) * stride + frame_count))
        cam_frames = int(cam_c2w.shape[1] if cam_c2w.dim() == 4 else cam_c2w.shape[0])
        keep = [i for i, frame_idx in enumerate(frame_indices) if 0 <= int(frame_idx) < cam_frames]
        if not keep:
            return
        pixels = pixels[:, :, keep].contiguous()
        frame_indices = [frame_indices[i] for i in keep]
        depth_by_local = (
            {}
            if self._uses_vigeo_prefix_last_frame()
            else self._infer_validation_bank_depths(
                pixels=pixels,
                metadata=metadata,
                frame_indices=frame_indices,
            )
        )
        for local_idx, frame_idx in enumerate(frame_indices):
            bank.pixels.append(pixels[:, :, local_idx].detach().contiguous())
            bank.frame_indices.append(int(frame_idx))
            bank.depths.append(depth_by_local.get(local_idx) if depth_by_local is not None else None)

    def _record_validation_vigeo_causal_prefix(
        self,
        *,
        bank: _RolloutSpatialBank,
        decoded_pixels: torch.Tensor | None,
        target_start: int,
    ) -> None:
        """Keep the latest RGB only for causal VAE encoding, not geometry retrieval."""
        if decoded_pixels is None or int(decoded_pixels.shape[2]) <= 0:
            raise RuntimeError("ViGeo causal prefix update requires decoded target pixels")
        target_pixel_start = (
            int(target_start) * int(self.cfg.sample.temporal_stride)
            + int(bank.vigeo_pixel_offset)
        )
        bank.causal_prefix_frame_index = target_pixel_start + int(decoded_pixels.shape[2]) - 1
        bank.causal_prefix_pixel = (
            decoded_pixels[:, :, -1].detach().to(device="cpu", dtype=torch.float16).contiguous()
        )

    def _build_validation_rollout_bank_spatial_context(
        self,
        *,
        bank: _RolloutSpatialBank,
        metadata: dict[str, Any],
        target_start: int,
        K: int,
        target_rope_t_indices: torch.Tensor | None,
    ) -> dict[str, Any] | None:
        cfg = self.cfg.spatial_memory
        if not bank.pixels:
            return None
        cam_c2w = metadata.get("cam_c2w")
        intrinsic = metadata.get("intrinsic")
        if cam_c2w is None or intrinsic is None:
            return None
        if cam_c2w.dim() == 3:
            cam_c2w = cam_c2w.unsqueeze(0)
        cam_c2w = cam_c2w.to(device=self.dist.device, dtype=torch.float32)
        intrinsic = intrinsic.to(device=self.dist.device, dtype=torch.float32)

        stride = int(self.cfg.sample.temporal_stride)
        target_pixel_start = int(target_start) * stride
        if self._uses_vigeo_prefix_last_frame():
            target_pixel_start += int(bank.vigeo_pixel_offset)
        target_pixel_count = (
            int(K) * stride
            if self._uses_vigeo_prefix_last_frame()
            else 1 + max(0, int(K) - 1) * stride
        )
        target_pixel_indices = list(range(target_pixel_start, target_pixel_start + target_pixel_count))
        cam_frames = int(cam_c2w.shape[1])
        if not target_pixel_indices or target_pixel_indices[-1] >= cam_frames:
            return None

        candidate_indices = [
            local_idx
            for local_idx, frame_idx in enumerate(bank.frame_indices)
            if 0 <= int(frame_idx) < int(target_pixel_start)
        ]
        if not candidate_indices:
            return None

        if self._uses_vigeo_prefix_last_frame():
            return self._build_validation_vigeo_bank_spatial_context(
                bank=bank,
                candidate_indices=candidate_indices,
                metadata=metadata,
                cam_c2w=cam_c2w,
                intrinsic=intrinsic,
                target_pixel_indices=target_pixel_indices,
                target_start=target_start,
                K=K,
                target_rope_t_indices=target_rope_t_indices,
            )

        selected = self._select_validation_rollout_bank_sources(
            bank=bank,
            candidate_indices=candidate_indices,
            target_pixel_indices=target_pixel_indices,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
        )
        num_context = max(1, int(cfg.num_context_frames))
        if bool(getattr(cfg, "require_full_context", True)) and len(selected) < num_context:
            return None
        if not selected:
            return None

        source_video = torch.stack(bank.pixels, dim=2).to(device=self.dist.device, dtype=self.dtype).contiguous()
        pixel_height, pixel_width = int(source_video.shape[-2]), int(source_video.shape[-1])
        depth_by_source = {
            int(local_idx): bank.depths[local_idx]
            for local_idx in selected
            if bank.depths[local_idx] is not None
        }
        warp_result = forward_warp_indexed_pixel_sources_to_pixel_targets(
            source_pixels=source_video,
            source_pixel_indices=[int(i) for i in selected],
            source_camera_pixel_indices=[int(i) for i in bank.frame_indices],
            target_pixel_indices=target_pixel_indices,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            depth_by_source_index=depth_by_source,
            height=pixel_height,
            width=pixel_width,
            constant_depth=float(cfg.constant_depth),
            depth_threshold=min(float(cfg.retrieval_depth_threshold), 1e-3),
            fill_value=None,
            return_coverage=True,
        )
        if warp_result is None:
            return None
        warped_pixels, coverage_pixels = warp_result

        spatial_latent = self._encode_spatial_context_video(warped_pixels, expected_latent_frames=int(K))
        mask_patch = self._build_spatial_mask_patch(
            coverage_pixels=coverage_pixels,
            spatial_latent=spatial_latent,
        )
        if target_rope_t_indices is None:
            rope_t_indices: list[float] = list(range(int(target_start), int(target_start) + int(K)))
        else:
            rope_t_indices = [float(x) for x in target_rope_t_indices.detach().cpu().tolist()]
        return self._maybe_force_spatial_all_invalid({
            "latent": spatial_latent,
            "mask_patch": mask_patch,
            "source_indices": [int(bank.frame_indices[i]) for i in selected],
            "target_indices": list(range(int(target_start), int(target_start) + int(K))),
            "source_pixel_indices": [int(bank.frame_indices[i]) for i in selected],
            "target_pixel_indices": target_pixel_indices,
            "rope_t_indices": rope_t_indices,
        }, metadata=metadata)

    def _build_validation_vigeo_bank_spatial_context(
        self,
        *,
        bank: _RolloutSpatialBank,
        candidate_indices: list[int],
        metadata: dict[str, Any],
        cam_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
        target_pixel_indices: list[int],
        target_start: int,
        K: int,
        target_rope_t_indices: torch.Tensor | None,
    ) -> dict[str, Any] | None:
        geometry_count = len(bank.vigeo_pointmaps)
        if geometry_count == 0 or geometry_count != len(bank.pixels):
            return None
        if not (
            len(bank.vigeo_valid_masks) == geometry_count
            and len(bank.vigeo_predicted_poses) == geometry_count
            and len(bank.vigeo_intrinsics) == geometry_count
        ):
            raise RuntimeError("ViGeo rollout bank geometry fields have inconsistent lengths")

        pixel_height, pixel_width = int(bank.pixels[0].shape[-2]), int(bank.pixels[0].shape[-1])
        source_global_c2w = self._validation_vigeo_bank_source_c2w(
            bank=bank,
            cam_c2w=cam_c2w,
        )
        target_c2w, target_intrinsic = self._validation_vigeo_target_cameras(
            bank=bank,
            metadata=metadata,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            target_pixel_indices=target_pixel_indices,
            height=pixel_height,
            width=pixel_width,
        )
        retrieval_coverages: dict[int, float] = {}
        if int(bank.vigeo_generated_chunks) == 0:
            selected = [candidate_indices[-1]]
        else:
            selected, retrieval_coverages = self._select_validation_vigeo_bank_sources(
                bank=bank,
                candidate_indices=candidate_indices,
                source_global_c2w=source_global_c2w,
                target_c2w=target_c2w,
                target_intrinsic=target_intrinsic,
                height=pixel_height,
                width=pixel_width,
            )
            if not selected:
                return None
        selected_tensor = torch.tensor(
            selected, device=source_global_c2w.device, dtype=torch.long
        )
        source_c2w = source_global_c2w.index_select(1, selected_tensor)

        source_video = torch.stack(
            [bank.pixels[index] for index in selected], dim=2
        ).to(device=self.dist.device, dtype=self.dtype).contiguous()
        source_pointmaps = torch.stack(
            [bank.vigeo_pointmaps[index] for index in selected], dim=0
        ).unsqueeze(0).to(device=self.dist.device, dtype=torch.float32)
        source_pointmaps = source_pointmaps * float(bank.vigeo_scale)
        source_valid_masks = torch.stack(
            [bank.vigeo_valid_masks[index] for index in selected], dim=0
        ).unsqueeze(0).to(device=self.dist.device, dtype=torch.bool)

        warp_result = render_colored_pointmaps_to_camera_targets(
            source_pixels=source_video,
            source_pointmaps=source_pointmaps,
            source_valid_masks=source_valid_masks,
            source_c2w=source_c2w,
            target_c2w=target_c2w,
            target_intrinsic=target_intrinsic,
            height=pixel_height,
            width=pixel_width,
            depth_threshold=min(
                float(self.cfg.spatial_memory.retrieval_depth_threshold), 1e-3
            ),
            fill_value=None,
            return_coverage=True,
        )
        if warp_result is None:
            return None
        warped_pixels, coverage_pixels = warp_result
        warped_pixels, coverage_pixels = self._composite_wbench_subject_anchor(
            bank=bank,
            warped_pixels=warped_pixels,
            coverage_pixels=coverage_pixels,
        )
        previous_pixel_index = int(target_pixel_indices[0]) - 1
        previous_bank_indices = [
            index
            for index, frame_index in enumerate(bank.frame_indices)
            if int(frame_index) == previous_pixel_index
        ]
        if previous_bank_indices:
            previous_bank_index = previous_bank_indices[-1]
            previous_pixels = bank.pixels[previous_bank_index].unsqueeze(2)
        elif (
            bank.causal_prefix_pixel is not None
            and bank.causal_prefix_frame_index == previous_pixel_index
        ):
            previous_pixels = bank.causal_prefix_pixel.unsqueeze(2)
        else:
            raise RuntimeError(
                "ViGeo rollout bank is missing the causal VAE prefix frame "
                f"{previous_pixel_index}; bank range={bank.frame_indices[:1]}..{bank.frame_indices[-1:]} "
                f"cached_prefix={bank.causal_prefix_frame_index}"
            )
        spatial_latent, vae_prefix_latent = self._encode_spatial_continuation_video(
            previous_pixels=previous_pixels,
            target_pixels=warped_pixels,
            expected_target_latent_frames=int(K),
        )
        mask_patch = self._build_spatial_mask_patch(
            coverage_pixels=coverage_pixels,
            spatial_latent=spatial_latent,
        )
        if target_rope_t_indices is None:
            rope_t_indices = list(range(int(target_start), int(target_start) + int(K)))
        else:
            rope_t_indices = [
                float(value)
                for value in target_rope_t_indices.detach().cpu().tolist()
            ]
        return self._maybe_force_spatial_all_invalid(
            {
                "latent": spatial_latent,
                "mask_patch": mask_patch,
                "source_indices": [int(bank.frame_indices[index]) for index in selected],
                "target_indices": list(range(int(target_start), int(target_start) + int(K))),
                "scale_context_pixel_indices": self._vigeo_scale_context_pixel_indices(),
                "source_pixel_indices": [int(bank.frame_indices[index]) for index in selected],
                "source_bank_indices": [int(index) for index in selected],
                "vae_prefix_pixel_index": previous_pixel_index,
                "vae_prefix_latent": vae_prefix_latent,
                "target_pixel_indices": target_pixel_indices,
                "rope_t_indices": rope_t_indices,
                "vigeo_scale": float(bank.vigeo_scale),
                "vigeo_source_pose_mode": str(
                    self.cfg.spatial_memory.vigeo_generated_source_pose_mode
                ),
                "vigeo_pairwise_scales": bank.vigeo_pairwise_scales,
                "vigeo_retrieval_coverages": {
                    str(bank.frame_indices[index]): float(retrieval_coverages.get(index, 0.0))
                    for index in selected
                },
            },
            metadata=metadata,
        )

    def _vigeo_bank_global_c2w(self, bank: _RolloutSpatialBank) -> torch.Tensor:
        poses = self._homogeneous_camera_poses(
            torch.stack(bank.vigeo_predicted_poses, dim=0)
        )
        anchor_inv = torch.linalg.inv(poses[:1])
        relative = torch.matmul(anchor_inv, poses)
        relative[:, :3, 3] *= float(bank.vigeo_scale)
        return relative.unsqueeze(0)

    def _validation_vigeo_bank_cplan_c2w(
        self,
        *,
        bank: _RolloutSpatialBank,
        cam_c2w: torch.Tensor,
    ) -> torch.Tensor:
        if cam_c2w.dim() == 3:
            cam_c2w = cam_c2w.unsqueeze(0)
        frame_indices = [int(frame_index) for frame_index in bank.frame_indices]
        if not frame_indices:
            raise ValueError("cannot build ViGeo source cameras for an empty bank")
        if min(frame_indices) < 0 or max(frame_indices) >= int(cam_c2w.shape[1]):
            raise ValueError(
                "ViGeo bank frame indices exceed the planned camera path: "
                f"range={min(frame_indices)}..{max(frame_indices)}, "
                f"camera_frames={int(cam_c2w.shape[1])}"
            )
        frame_tensor = torch.tensor(
            frame_indices,
            device=cam_c2w.device,
            dtype=torch.long,
        )
        planned_sources = cam_c2w.index_select(1, frame_tensor)
        return torch.matmul(
            torch.linalg.inv(cam_c2w[:, :1]),
            planned_sources,
        ).to(device=self.dist.device, dtype=torch.float32).contiguous()

    def _validation_vigeo_bank_source_c2w(
        self,
        *,
        bank: _RolloutSpatialBank,
        cam_c2w: torch.Tensor,
    ) -> torch.Tensor:
        planned = self._validation_vigeo_bank_cplan_c2w(
            bank=bank,
            cam_c2w=cam_c2w,
        )
        mode = str(self.cfg.spatial_memory.vigeo_generated_source_pose_mode)
        if mode == "cplan" or int(bank.vigeo_generated_chunks) == 0:
            return planned

        predicted = self._vigeo_bank_global_c2w(bank).to(
            device=planned.device,
            dtype=planned.dtype,
        )
        target_prefix_frames = self._vigeo_target_prefix_pixel_frames()
        anchor_frame = target_prefix_frames - 1
        anchor_candidates = [
            index
            for index, frame_index in enumerate(bank.frame_indices)
            if int(frame_index) == anchor_frame
        ]
        if not anchor_candidates:
            raise RuntimeError(
                "ViGeo rollout bank is missing the final scale-prefix frame "
                f"{anchor_frame}"
            )
        anchor_index = anchor_candidates[-1]

        # Both trajectories are first expressed relative to P0. A single rigid
        # correction then makes ViGeo's final observed prefix camera equal to
        # c_plan at the history endpoint, while preserving post-prefix motion.
        correction = planned[:, anchor_index] @ torch.linalg.inv(
            predicted[:, anchor_index]
        )
        aligned_predicted = correction.unsqueeze(1) @ predicted
        generated_indices = [
            index
            for index, frame_index in enumerate(bank.frame_indices)
            if int(frame_index) >= target_prefix_frames
        ]
        if not generated_indices:
            return planned
        generated_tensor = torch.tensor(
            generated_indices,
            device=planned.device,
            dtype=torch.long,
        )
        result = planned.clone()
        result[:, generated_tensor] = aligned_predicted[:, generated_tensor]
        return result.contiguous()

    def _validation_vigeo_target_cameras(
        self,
        *,
        bank: _RolloutSpatialBank,
        metadata: dict[str, Any],
        cam_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
        target_pixel_indices: list[int],
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_tensor = torch.tensor(
            target_pixel_indices, device=cam_c2w.device, dtype=torch.long
        )
        metadata_targets = cam_c2w.index_select(1, target_tensor)
        # Planned cameras own the shared P0 frame for both source and target views.
        target_c2w = torch.matmul(torch.linalg.inv(cam_c2w[:, :1]), metadata_targets)

        if self._is_uncalibrated_intrinsic_source(metadata):
            # GameVerse/WBench has no calibrated intrinsics. Re-fitting focal length
            # from every generated chunk creates a systematic zoom reset, so
            # keep the estimate from the final initial-prefix frame fixed.
            prefix_intrinsic_index = min(
                self._vigeo_prefix_pixel_frames(), len(bank.vigeo_intrinsics)
            ) - 1
            fitted = bank.vigeo_intrinsics[prefix_intrinsic_index].to(
                device=self.dist.device, dtype=torch.float32
            )
            target_K = fitted.view(1, 1, 3, 3).expand(
                int(cam_c2w.shape[0]), len(target_pixel_indices), -1, -1
            ).clone()
        else:
            target_K = torch.stack(
                [
                    self._intrinsic_for_pixel_frame(
                        intrinsic,
                        frame_idx,
                        batch=int(cam_c2w.shape[0]),
                        height=height,
                        width=width,
                    )
                    for frame_idx in target_pixel_indices
                ],
                dim=1,
            )
        return target_c2w.contiguous(), target_K.contiguous()

    def _select_validation_vigeo_bank_sources(
        self,
        *,
        bank: _RolloutSpatialBank,
        candidate_indices: list[int],
        source_global_c2w: torch.Tensor,
        target_c2w: torch.Tensor,
        target_intrinsic: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[list[int], dict[int, float]]:
        num_context = max(1, int(self.cfg.spatial_memory.num_context_frames))
        source_centers = source_global_c2w[0, :, :3, 3].detach().cpu()
        target_centers = target_c2w[0, :, :3, 3].detach().cpu()
        source_forwards = F.normalize(
            source_global_c2w[0, :, :3, 2].detach().float().cpu(), dim=-1
        )
        target_forwards = F.normalize(
            target_c2w[0, :, :3, 2].detach().float().cpu(), dim=-1
        )
        distances = {
            int(index): float(
                torch.linalg.vector_norm(
                    source_centers[index].view(1, 3) - target_centers,
                    dim=-1,
                ).min().item()
            )
            for index in candidate_indices
        }
        angular_distances = {
            int(index): float(
                torch.acos(
                    torch.matmul(
                        target_forwards,
                        source_forwards[index],
                    ).clamp(-1.0, 1.0)
                ).min().item()
            )
            for index in candidate_indices
        }
        rotation_weight = max(
            0.0,
            float(getattr(self.cfg.spatial_memory, "retrieval_rotation_weight", 0.0)),
        )
        pose_scores = {
            int(index): distances[int(index)]
            + rotation_weight * angular_distances[int(index)]
            for index in candidate_indices
        }
        nearest = sorted(
            candidate_indices,
            key=lambda index: (pose_scores[int(index)], -int(index)),
        )
        prefilter_count = min(len(nearest), max(num_context * 8, num_context))
        nearest = nearest[:prefilter_count]

        coverages = {
            int(index): self._vigeo_source_target_coverage(
                pointmap=bank.vigeo_pointmaps[index],
                valid_mask=bank.vigeo_valid_masks[index],
                source_c2w=source_global_c2w[:, index],
                target_c2w=target_c2w,
                target_intrinsic=target_intrinsic,
                scale=float(bank.vigeo_scale),
                height=height,
                width=width,
            )
            for index in nearest
        }
        visible = [index for index in nearest if coverages[int(index)] >= 0.002]
        if str(getattr(self.cfg.spatial_memory, "retrieval_sort", "nearest")) == "coverage":
            visible = sorted(visible, key=lambda index: -coverages[int(index)])
        selected = visible[:num_context]
        if not selected and nearest:
            selected = [max(nearest, key=lambda index: (coverages[int(index)], -distances[int(index)]))]
        return [int(index) for index in selected], coverages

    def _vigeo_source_target_coverage(
        self,
        *,
        pointmap: torch.Tensor,
        valid_mask: torch.Tensor,
        source_c2w: torch.Tensor,
        target_c2w: torch.Tensor,
        target_intrinsic: torch.Tensor,
        scale: float,
        height: int,
        width: int,
    ) -> float:
        sample_stride = max(4, int(self.cfg.spatial_memory.downsample))
        points = pointmap[::sample_stride, ::sample_stride].to(
            device=self.dist.device, dtype=torch.float32
        ) * float(scale)
        valid = valid_mask[::sample_stride, ::sample_stride].to(
            device=self.dist.device, dtype=torch.bool
        )
        valid = valid & torch.isfinite(points).all(dim=-1) & (points[..., 2] > 0)
        if not bool(valid.any()):
            return 0.0
        points = points[valid]
        ones = torch.ones((int(points.shape[0]), 1), device=points.device, dtype=points.dtype)
        world = torch.matmul(
            source_c2w[0].to(device=points.device, dtype=points.dtype),
            torch.cat([points, ones], dim=-1).unsqueeze(-1),
        )[:, :3, 0]

        target_count = int(target_c2w.shape[1])
        query_indices = sorted({0, target_count // 2, target_count - 1})
        query_weights = {0: 0.6, target_count // 2: 0.2, target_count - 1: 0.2}
        cell_size = 16
        grid_width = max(1, (int(width) + cell_size - 1) // cell_size)
        grid_height = max(1, (int(height) + cell_size - 1) // cell_size)
        total_cells = float(grid_width * grid_height)
        weighted_coverage = 0.0
        weight_total = 0.0
        world_h = torch.cat([world, torch.ones_like(world[:, :1])], dim=-1).unsqueeze(-1)
        for query_index in query_indices:
            camera = torch.matmul(
                torch.linalg.inv(target_c2w[0, query_index]).to(world_h),
                world_h,
            )[:, :3, 0]
            z = camera[:, 2]
            projected = torch.matmul(
                target_intrinsic[0, query_index].to(camera), camera.unsqueeze(-1)
            )[:, :, 0]
            x = torch.round(projected[:, 0] / torch.clamp(projected[:, 2], min=1e-6)).long()
            y = torch.round(projected[:, 1] / torch.clamp(projected[:, 2], min=1e-6)).long()
            visible = (
                torch.isfinite(camera).all(dim=-1)
                & (z > 0)
                & (x >= 0)
                & (x < int(width))
                & (y >= 0)
                & (y < int(height))
            )
            if bool(visible.any()):
                cells = (y[visible] // cell_size) * grid_width + (x[visible] // cell_size)
                coverage = float(torch.unique(cells).numel()) / total_cells
            else:
                coverage = 0.0
            weight = float(query_weights[query_index])
            weighted_coverage += weight * coverage
            weight_total += weight
        return weighted_coverage / max(weight_total, 1e-8)

    @staticmethod
    def _homogeneous_camera_poses(poses: torch.Tensor) -> torch.Tensor:
        poses = poses.detach().float()
        if poses.dim() != 3:
            raise ValueError(f"camera poses must be [F,3/4,4], got {tuple(poses.shape)}")
        if tuple(poses.shape[-2:]) == (4, 4):
            return poses
        if tuple(poses.shape[-2:]) != (3, 4):
            raise ValueError(f"unsupported camera pose shape {tuple(poses.shape)}")
        bottom = torch.zeros(
            (int(poses.shape[0]), 1, 4), device=poses.device, dtype=poses.dtype
        )
        bottom[:, 0, 3] = 1.0
        return torch.cat([poses, bottom], dim=1)

    def _select_validation_rollout_bank_sources(
        self,
        *,
        bank: _RolloutSpatialBank,
        candidate_indices: list[int],
        target_pixel_indices: list[int],
        cam_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> list[int]:
        cfg = self.cfg.spatial_memory
        num_context = max(1, int(cfg.num_context_frames))
        pixel_height, pixel_width = int(bank.pixels[0].shape[-2]), int(bank.pixels[0].shape[-1])
        selected: list[int] = []
        try:
            cache = Sparse3DCache(downsample=max(1, int(cfg.downsample)))
            for local_idx in candidate_indices:
                frame_idx = int(bank.frame_indices[local_idx])
                depth = bank.depths[local_idx]
                if depth is None:
                    depth = torch.full(
                        (cam_c2w.shape[0], 1, pixel_height, pixel_width),
                        float(cfg.constant_depth),
                        device=self.dist.device,
                        dtype=torch.float32,
                    )
                else:
                    depth = depth.to(device=self.dist.device, dtype=torch.float32)
                cache.add(
                    depth=depth,
                    w2c=torch.linalg.inv(cam_c2w[:, frame_idx]),
                    intrinsic=self._intrinsic_for_pixel_frame(
                        intrinsic,
                        frame_idx,
                        batch=int(cam_c2w.shape[0]),
                        height=pixel_height,
                        width=pixel_width,
                    ),
                    latent_index=int(local_idx),
                    frame_id=frame_idx,
                )

            retrieval_views = max(1, int(cfg.retrieval_views))
            if retrieval_views == 1:
                target_view_indices = [target_pixel_indices[-1]]
            else:
                offsets = torch.linspace(0, len(target_pixel_indices) - 1, retrieval_views)
                target_view_indices = [target_pixel_indices[int(round(float(x.item())))] for x in offsets]
            target_w2c = torch.stack([torch.linalg.inv(cam_c2w[:, idx]) for idx in target_view_indices], dim=1)
            target_K = torch.stack(
                [
                    self._intrinsic_for_pixel_frame(
                        intrinsic,
                        idx,
                        batch=int(cam_c2w.shape[0]),
                        height=pixel_height,
                        width=pixel_width,
                    )
                    for idx in target_view_indices
                ],
                dim=1,
            )
            retrieved = cache.retrieve(
                target_w2c=target_w2c,
                target_intrinsic=target_K,
                target_hw=(pixel_height, pixel_width),
                num_latents=num_context,
                max_coverage=bool(cfg.retrieval_max_coverage),
                depth_threshold=float(cfg.retrieval_depth_threshold),
            )
            selected = [int(local_idx) for local_idx, _frame_id in retrieved]
        except Exception as exc:
            rank0_print(self.dist, "[ValidationSpatialBank]", f"coverage source selection failed: {type(exc).__name__}: {exc}")
            selected = []

        if len(selected) < num_context:
            seen = set(selected)
            for local_idx in reversed(candidate_indices):
                if local_idx in seen:
                    continue
                selected.append(int(local_idx))
                seen.add(int(local_idx))
                if len(selected) >= num_context:
                    break
        return selected[:num_context]

    def _infer_validation_bank_depths(
        self,
        *,
        pixels: torch.Tensor,
        metadata: dict[str, Any],
        frame_indices: list[int],
    ) -> dict[int, torch.Tensor]:
        cfg = self.cfg.spatial_memory
        backend = str(cfg.depth_backend)
        if backend == "vigeo":
            # ViGeo needs the complete prefix jointly; rollout builds it lazily
            # in _build_validation_vigeo_bank_spatial_context.
            return {}
        if backend == "constant":
            return {}
        if backend == "metadata":
            depth = metadata.get("depth")
            if depth is None:
                return {}
            selected = _select_depth_by_frame_index(depth, frame_indices)
            return {
                local_idx: selected[int(frame_idx)]
                for local_idx, frame_idx in enumerate(frame_indices)
                if int(frame_idx) in selected
            }
        if backend != "da3":
            raise ValueError(f"unsupported spatial_memory.depth_backend={backend!r}")

        cam_c2w = metadata.get("cam_c2w")
        intrinsic = metadata.get("intrinsic")
        if cam_c2w is None or intrinsic is None:
            return {}
        local_frame_indices = list(range(int(pixels.shape[2])))
        cam_subset = self._select_camera_frames(cam_c2w, frame_indices)
        intrinsic_subset = self._select_intrinsic_frames(intrinsic, frame_indices)
        return self._build_spatial_depths_for_pixel_frames(
            video_pixels=pixels,
            metadata={},
            cam_c2w=cam_subset,
            intrinsic=intrinsic_subset,
            frame_indices=local_frame_indices,
        ) or {}

    def _select_video_pixel_frames(self, video_pixels: torch.Tensor, frame_indices: list[int]) -> torch.Tensor:
        video = self._video_pixels_to_bcfhw(video_pixels).to(device=self.dist.device, dtype=self.dtype)
        idx = torch.tensor([int(i) for i in frame_indices], device=video.device, dtype=torch.long)
        return video.index_select(2, idx).contiguous()

    def _slice_video_pixel_frames(
        self,
        video_pixels: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Return a [B,F,C,H,W] pixel slice accepted by the VAE path."""
        video = self._video_pixels_to_bcfhw(video_pixels)
        start = max(0, int(start))
        end = min(int(end), int(video.shape[2]))
        if end <= start:
            raise ValueError(f"empty video pixel slice [{start}:{end}]")
        return video[:, :, start:end].permute(0, 2, 1, 3, 4).contiguous()

    def _video_pixels_to_bcfhw(self, video_pixels: torch.Tensor) -> torch.Tensor:
        video = video_pixels.detach()
        if video.dim() == 5:
            if video.shape[1] == 3:
                return video.contiguous()
            if video.shape[2] == 3:
                return video.permute(0, 2, 1, 3, 4).contiguous()
        if video.dim() == 4:
            if video.shape[0] == 3:
                return video.unsqueeze(0).contiguous()
            if video.shape[1] == 3:
                return video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        raise ValueError(f"expected video tensor [B,C,F,H,W], [B,F,C,H,W], [C,F,H,W], or [F,C,H,W], got {tuple(video.shape)}")

    def _pixel_hw_from_video(self, video_pixels: torch.Tensor) -> tuple[int, int]:
        video = self._video_pixels_to_bcfhw(video_pixels)
        return int(video.shape[-2]), int(video.shape[-1])

    def _select_camera_frames(self, cam_c2w: torch.Tensor, frame_indices: list[int]) -> torch.Tensor:
        cam = cam_c2w.to(device=self.dist.device, dtype=torch.float32)
        idx = torch.tensor([int(i) for i in frame_indices], device=cam.device, dtype=torch.long)
        if cam.dim() == 4:
            idx = idx.clamp(0, cam.shape[1] - 1)
            return cam.index_select(1, idx).contiguous()
        if cam.dim() == 3:
            idx = idx.clamp(0, cam.shape[0] - 1)
            return cam.index_select(0, idx).unsqueeze(0).contiguous()
        raise ValueError(f"unexpected cam_c2w shape {tuple(cam_c2w.shape)}")

    def _select_intrinsic_frames(self, intrinsic: torch.Tensor, frame_indices: list[int]) -> torch.Tensor:
        K = intrinsic.to(device=self.dist.device, dtype=torch.float32)
        if K.dim() == 4:
            idx = torch.tensor([int(i) for i in frame_indices], device=K.device, dtype=torch.long).clamp(0, K.shape[1] - 1)
            return K.index_select(1, idx).contiguous()
        return K

    def _intrinsic_for_pixel_frame(
        self,
        intrinsic: torch.Tensor,
        frame_idx: int,
        *,
        batch: int,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        K = intrinsic.to(device=self.dist.device, dtype=torch.float32)
        if K.dim() == 4:
            idx = max(0, min(int(frame_idx), int(K.shape[1]) - 1))
            K = K[:, idx]
        elif K.dim() == 2:
            K = K.unsqueeze(0)
        pixel_height = int(height if height is not None else self.cfg.sample.height)
        pixel_width = int(width if width is not None else self.cfg.sample.width)
        K = pixel_intrinsics(K, height=pixel_height, width=pixel_width)
        if K.shape[0] == 1 and int(batch) > 1:
            K = K.expand(int(batch), -1, -1)
        return K.contiguous()

    def _write_video(self, path: Path, frames: torch.Tensor) -> None:
        from torchvision.io import write_video

        path.parent.mkdir(parents=True, exist_ok=True)
        write_video(
            str(path),
            frames,
            fps=int(self.cfg.sample.fps),
            options={"crf": "18", "preset": "veryfast"},
        )

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, str, dict[str, Any]]:
        if isinstance(batch, dict):
            video_pixels = batch["video_pixels"]
            caption = str(batch.get("caption", ""))
            metadata = dict(batch.get("metadata", {}))
            return video_pixels, caption, metadata
        # New dataset tuples always end with rolled_gap_steps, rolled_cond_mode_id.
        # Synced multi-K loaders append one more layout_K after those two.  Single-K
        # loaders do not, so avoid treating rolled_cond=-1 as layout_K=-1.
        configured_k = {int(k) for k in self.cfg.layout.output.latent_frames}
        layout_k_candidate = _maybe_int_scalar(batch[-1]) if len(batch) > 0 else None
        has_layout_k = layout_k_candidate is not None and int(layout_k_candidate) in configured_k
        layout_k = int(layout_k_candidate) if has_layout_k else None
        if has_layout_k:
            rolled_cond_mode_id = _maybe_int_scalar(batch[-2]) if len(batch) >= 3 else None
            rolled_gap_steps = _maybe_int_scalar(batch[-3]) if len(batch) >= 4 else None
            trailing_meta = 3 if rolled_gap_steps is not None and rolled_cond_mode_id is not None else 1
        else:
            rolled_cond_mode_id = _maybe_int_scalar(batch[-1]) if len(batch) >= 2 else None
            rolled_gap_steps = _maybe_int_scalar(batch[-2]) if len(batch) >= 3 else None
            trailing_meta = 2 if rolled_gap_steps is not None and rolled_cond_mode_id is not None else 0
        core_len = len(batch) - trailing_meta
        video_pixels = batch[0]
        caption = _first_scalar(batch[2])
        metadata = {
            "intrinsic": batch[3] if len(batch) > 3 else None,
            "cam_c2w": batch[4] if len(batch) > 4 else None,
            "video_id": str(_first_scalar(batch[5])) if len(batch) > 5 else "unknown",
            "has_camera": _first_scalar(batch[6]) if len(batch) > 6 else False,
            "source": str(_first_scalar(batch[7])) if len(batch) > 7 else "",
            "caption_type": str(_first_scalar(batch[8])) if len(batch) > 8 else "",
            "pose_orig_w": _first_scalar(batch[9]) if len(batch) > 9 else 0.0,
            "pose_orig_h": _first_scalar(batch[10]) if len(batch) > 10 else 0.0,
            "frame_start": _first_scalar(batch[11]) if len(batch) > 11 else -1,
            "frame_end": _first_scalar(batch[12]) if len(batch) > 12 else -1,
            "intrinsic_raw": batch[13] if core_len >= 15 else None,
            "cam_c2w_raw": batch[14] if core_len >= 15 else None,
        }
        if layout_k is not None:
            metadata["layout_K"] = int(layout_k)
        if rolled_gap_steps is not None and int(rolled_gap_steps) >= 0:
            metadata["rolled_gap_steps"] = int(rolled_gap_steps)
        if rolled_cond_mode_id is not None and int(rolled_cond_mode_id) >= 0:
            metadata["rolled_cond_mode_id"] = int(rolled_cond_mode_id)
        return video_pixels, str(caption), metadata

    def _extend_validation_camera_static(self, metadata: dict[str, Any], *, min_frames: int) -> None:
        if not _as_bool(metadata.get("has_camera", False)):
            return

        target_frames = max(0, int(min_frames))
        cam = metadata.get("cam_c2w")
        cam_frames = self._camera_frame_count(cam)
        if cam_frames <= 0 or cam_frames >= target_frames:
            return

        extension_mode = str(getattr(self.cfg.validation, "camera_extension", "static") or "static")
        for key in ("cam_c2w", "cam_c2w_raw"):
            value = metadata.get(key)
            if extension_mode == "forward":
                padded = self._pad_camera_tensor_forward(value, target_frames=target_frames)
            else:
                padded = self._pad_frame_tensor_static(value, target_frames=target_frames)
            if padded is not None:
                metadata[key] = padded
        for key in ("intrinsic", "intrinsic_raw"):
            value = metadata.get(key)
            padded = self._pad_frame_tensor_static(
                value,
                target_frames=target_frames,
                expected_frames=cam_frames,
            )
            if padded is not None:
                metadata[key] = padded
        metadata["validation_camera_extension"] = True
        metadata["validation_camera_extension_mode"] = extension_mode
        metadata["validation_camera_static_extension"] = extension_mode == "static"
        metadata["validation_camera_original_frames"] = int(cam_frames)
        metadata["validation_camera_extended_frames"] = int(target_frames)

    def _camera_frame_count(self, cam_c2w: Any) -> int:
        if not torch.is_tensor(cam_c2w):
            return 0
        if cam_c2w.dim() == 4:
            return int(cam_c2w.shape[1])
        if cam_c2w.dim() == 3:
            return int(cam_c2w.shape[0])
        return 0

    def _pad_frame_tensor_static(
        self,
        tensor: Any,
        *,
        target_frames: int,
        expected_frames: int | None = None,
    ) -> torch.Tensor | None:
        if not torch.is_tensor(tensor):
            return None
        if tensor.dim() == 4:
            frame_dim = 1
        elif tensor.dim() == 3 and (expected_frames is None or int(tensor.shape[0]) == int(expected_frames)):
            frame_dim = 0
        else:
            return None

        frames = int(tensor.shape[frame_dim])
        if frames <= 0 or frames >= int(target_frames):
            return tensor
        pad_count = int(target_frames) - frames
        last = tensor.narrow(frame_dim, frames - 1, 1).expand(
            *[
                (pad_count if dim == frame_dim else int(size))
                for dim, size in enumerate(tensor.shape)
            ]
        )
        return torch.cat([tensor, last.to(device=tensor.device, dtype=tensor.dtype)], dim=frame_dim).contiguous()

    def _pad_camera_tensor_forward(self, tensor: Any, *, target_frames: int) -> torch.Tensor | None:
        if not torch.is_tensor(tensor):
            return None
        if tensor.dim() == 4:
            batched = True
            cam = tensor
        elif tensor.dim() == 3 and tensor.shape[-2:] == (4, 4):
            batched = False
            cam = tensor.unsqueeze(0)
        else:
            return None
        if cam.shape[-2:] != (4, 4):
            return None

        frames = int(cam.shape[1])
        if frames <= 0 or frames >= int(target_frames):
            return tensor
        pad_count = int(target_frames) - frames
        step_per_latent = float(getattr(self.cfg.validation, "camera_forward_step_per_latent", 0.05))
        step_per_frame = step_per_latent / max(1, int(self.cfg.sample.temporal_stride))

        last = cam[:, frames - 1]
        padded = last[:, None].expand(-1, pad_count, -1, -1).clone()
        rel = torch.zeros(
            int(cam.shape[0]),
            pad_count,
            3,
            device=cam.device,
            dtype=cam.dtype,
        )
        rel[:, :, 2] = (
            torch.arange(1, pad_count + 1, device=cam.device, dtype=cam.dtype)
            * float(step_per_frame)
        )[None, :]
        world_delta = torch.einsum("bij,bpj->bpi", last[:, :3, :3], rel)
        padded[:, :, :3, 3] = last[:, None, :3, 3] + world_delta
        out = torch.cat([cam, padded.to(device=cam.device, dtype=cam.dtype)], dim=1).contiguous()
        return out if batched else out.squeeze(0)

    def _caption_with_prefix(self, caption: str, metadata: dict[str, Any]) -> str:
        source = str(metadata.get("source") or "")
        prefix = MultiSourceVideoDataset.SOURCE_CONFIGS.get(source, {}).get("caption_prefix", "")
        if not prefix:
            return caption
        if caption.startswith(prefix):
            return caption
        return f"{prefix}{caption}"

    def _validation_prompt_schedule(
        self,
        *,
        mode_cfg: ValidationModeConfig,
        prompt_caption: str,
        caption: str,
        metadata: dict[str, Any],
        rounds: int,
    ) -> list[str] | None:
        wb = metadata.get("wbench_prompt_schedule")
        if wb and rounds > 0:
            cpt = max(1, int(getattr(mode_cfg, "wbench_chunks_per_turn", 3)))
            return [str(wb[min(r // cpt, len(wb) - 1)]) for r in range(int(rounds))]
        schedule = list(getattr(mode_cfg, "prompt_schedule", []) or [])
        if not schedule:
            return None
        if rounds <= 0:
            return []
        return [
            self._validation_prompt_for_label(
                label=str(schedule[min(i, len(schedule) - 1)]),
                prompt_caption=prompt_caption,
                caption=caption,
                metadata=metadata,
            )
            for i in range(int(rounds))
        ]

    def _validation_prompt_for_label(
        self,
        *,
        label: str,
        prompt_caption: str,
        caption: str,
        metadata: dict[str, Any],
    ) -> str:
        normalized = label.strip().lower()
        source = str(metadata.get("source") or "").lower()
        if normalized in {"magic", "event", "caption", "base"}:
            return prompt_caption
        if normalized in {"raw_caption", "raw"}:
            return caption
        return label

    def _validation_prompt_label(self, *, mode_cfg: ValidationModeConfig, round_idx: int) -> str | None:
        schedule = list(getattr(mode_cfg, "prompt_schedule", []) or [])
        if not schedule:
            return None
        return str(schedule[min(int(round_idx), len(schedule) - 1)]).strip()



    def _optimizer_parameters(self) -> tuple[list[torch.nn.Parameter], list[dict[str, Any]]]:
        assert self.components is not None

        base_params = []
        if self.history_encoder is not None:
            base_params.extend(p for p in self.history_encoder.parameters() if p.requires_grad)
        action_params = []
        action_lr = self.cfg.control.action_learning_rate if self.cfg.control.uses("action") else None
        for name, param in self.components.transformer.named_parameters():
            if not param.requires_grad:
                continue
            if action_lr is not None and (id(param) in self.action_param_ids or _is_action_adaln_param(name)):
                action_params.append(param)
            else:
                base_params.append(param)

        groups: list[dict[str, Any]] = []
        if base_params:
            groups.append({"params": base_params, "lr": self.cfg.optimizer.lr})
        if action_params:
            groups.append({"params": action_params, "lr": float(action_lr)})
            rank0_print(
                self.dist,
                "[Optimizer]",
                f"action_adaln_params={self.action_param_count:,} "
                f"local_shard_numel={sum(p.numel() for p in action_params):,} "
                f"lr={float(action_lr):.2e}",
            )
        return base_params + action_params, groups

    def _video_pixel_frame_count(self, video_pixels: torch.Tensor) -> int:
        if video_pixels.dim() == 5:
            if video_pixels.shape[2] == 3:
                return int(video_pixels.shape[1])
            if video_pixels.shape[1] == 3:
                return int(video_pixels.shape[2])
            raise ValueError(f"expected video tensor [B,F,C,H,W] or [B,C,F,H,W], got {tuple(video_pixels.shape)}")
        if video_pixels.dim() != 4:
            raise ValueError(f"expected video tensor [F,C,H,W] or [C,F,H,W], got {tuple(video_pixels.shape)}")
        return int(video_pixels.shape[1] if video_pixels.shape[0] == 3 else video_pixels.shape[0])

    def _encode_video(
        self,
        video_pixels: torch.Tensor,
        *,
        needed_latents: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        video_pixels = video_pixels.to(device=self.dist.device, dtype=self.dtype)
        if video_pixels.dim() == 5:
            # DataLoader gives [B, F, C, H, W]. The first rewrite assumes batch_size=1.
            if video_pixels.shape[0] != 1:
                raise ValueError("first clean trainer supports train.batch_size=1")
            video_pixels = video_pixels[0]
        if video_pixels.dim() != 4:
            raise ValueError(f"expected video tensor [F,C,H,W] or [B,F,C,H,W], got {tuple(video_pixels.shape)}")
        if video_pixels.shape[0] != 3:
            video_pixels = video_pixels.permute(1, 0, 2, 3).contiguous()

        needed_latents = needed_latents or self._required_latents_for_max_K()
        needed_pixels = (needed_latents - 1) * self.cfg.sample.temporal_stride + 1
        use_pixels = min(video_pixels.shape[1], needed_pixels)
        use_pixels = 1 + self.cfg.sample.temporal_stride * ((use_pixels - 1) // self.cfg.sample.temporal_stride)
        video_pixels = video_pixels[:, :use_pixels]

        _t0 = time.time()
        latent = self._encode_video_via_latent_cache(video_pixels, metadata)
        if latent is None:
            with torch.no_grad():
                latent = self.components.vae_encoder.encode(
                    video_pixels.unsqueeze(0),
                    chunk_size=self.cfg.runtime.vae_chunk_size,
                    verbose=False,
                )
        latent = latent.to(device=self.dist.device, dtype=self.dtype)
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.dist.device)
        self._perf_vae_s += time.time() - _t0
        return latent

    def _encode_video_via_latent_cache(
        self, video_pixels: torch.Tensor, metadata: dict[str, Any] | None
    ) -> torch.Tensor | None:
        """Whole-clip latent cache path: the first FRESH_HEAD_LAT latents are encoded fresh and the tail

        is sliced from the cache, which is bit-identical to a fresh full-window encode (a causal VAE
        carries about 16 latents of memory). A miss, a misaligned start or a too-short window returns None.
        """
        cache_dir = getattr(self.cfg.runtime, "vae_latent_cache_dir", None)
        if not cache_dir or metadata is None:
            return None
        from alaya.data.vae_latent_cache import FRESH_HEAD_LAT, aligned_m_start, load_meta, load_slice

        T_pix = int(video_pixels.shape[1])
        n_lat = (T_pix - 1) // self.cfg.sample.temporal_stride + 1
        if n_lat <= FRESH_HEAD_LAT:
            return None  # short windows gain nothing: the whole window lies inside the fresh head
        source = str(metadata.get("source", ""))
        video_id = str(metadata.get("video_id", ""))
        frame_start = int(metadata.get("frame_start", -1))
        if not source or not video_id or frame_start < 0:
            return None
        H, W = int(self.cfg.sample.height), int(self.cfg.sample.width)
        fps = float(self.cfg.sample.fps)
        meta = load_meta(cache_dir, source, video_id, H, W, fps)
        if meta is None:
            self._vae_cache_misses += 1
            return None
        m = aligned_m_start(frame_start, float(meta["ratio"]))
        if m is None:
            self._vae_cache_misses += 1
            return None
        tail = load_slice(cache_dir, source, video_id, H, W, fps, m + FRESH_HEAD_LAT, n_lat - FRESH_HEAD_LAT)
        if tail is None:
            self._vae_cache_misses += 1
            return None
        head_pixels = video_pixels[:, : (FRESH_HEAD_LAT - 1) * self.cfg.sample.temporal_stride + 1]
        with torch.no_grad():
            head = self.components.vae_encoder.encode(
                head_pixels.unsqueeze(0),
                chunk_size=self.cfg.runtime.vae_chunk_size,
                verbose=False,
            )
        self._vae_cache_hits += 1
        tail = tail.unsqueeze(0).to(device=head.device, dtype=head.dtype)
        return torch.cat([head, tail], dim=2)

    def _encode_caption(self, caption: str, *, sync: bool) -> torch.Tensor:
        if sync and dist.is_initialized():
            captions = [caption]
            dist.broadcast_object_list(captions, src=0)
            caption = captions[0]
        cache_cap = int(getattr(self.cfg.runtime, "text_embed_cache_entries", 0) or 0)
        cache_dir = getattr(self.cfg.runtime, "text_embed_cache_dir", None)
        t0 = time.time()
        if cache_cap > 0:
            cached = self._text_cache.get(caption)
            if cached is not None:
                self._text_cache.move_to_end(caption)
                self._text_cache_hits += 1
                self._perf_text_s += time.time() - t0
                return cached.to(device=self.dist.device, dtype=self.dtype)
        if cache_dir:
            from alaya.data.text_embed_cache import disk_get
            hit = disk_get(cache_dir, caption)
            if hit is not None:
                self._text_cache_hits += 1
                if cache_cap > 0:
                    self._text_cache[caption] = hit
                    while len(self._text_cache) > cache_cap:
                        self._text_cache.popitem(last=False)
                self._perf_text_s += time.time() - t0
                return hit.to(device=self.dist.device, dtype=self.dtype)
        with torch.no_grad():
            output = self.components.encode_text(self.components.text_encoder, [caption])
        context = output[0][0] if isinstance(output, list) and output[0].dim() == 3 else output[0]
        context = context.to(device=self.dist.device, dtype=self.dtype)
        self._text_cache_misses += 1
        if cache_dir:
            from alaya.data.text_embed_cache import disk_put
            disk_put(cache_dir, caption, context)
        if cache_cap > 0:
            self._text_cache[caption] = context.detach().to("cpu")
            while len(self._text_cache) > cache_cap:
                self._text_cache.popitem(last=False)
        self._perf_text_s += time.time() - t0
        return context

    def _sample_layout(self, metadata: dict[str, Any] | None = None) -> tuple[int, int, str, int]:
        if self.dist.is_main:
            if metadata is not None and metadata.get("layout_K") is not None:
                K = int(metadata["layout_K"])
            else:
                K = random.choices(self.cfg.layout.output.latent_frames, weights=self.cfg.layout.output.probs, k=1)[0]
            max_gap = int(self.cfg.layout.max_gap_sec * self.cfg.sample.fps / self.cfg.sample.temporal_stride)
            _k8_valid = (
                (int(K) == 8 and bool(self.cfg.layout.k8_use_valid_starts))
                or (int(K) == 4 and bool(self.cfg.layout.k4_use_valid_starts))
            )
            _has_rolled = (
                metadata is not None
                and metadata.get("rolled_gap_steps") is not None
                and metadata.get("rolled_cond_mode_id") is not None
            )
            if _has_rolled:
                gap_steps = int(metadata["rolled_gap_steps"])
                rolled_cond_id = int(metadata["rolled_cond_mode_id"])
                cond_mode = {0: "hc", 1: "i2v"}[rolled_cond_id]
                cond_end = 1 if cond_mode == "i2v" else 0
            else:
                fixed_total = None
                r = random.random()
                if r < self.cfg.layout.condition.i2v_prob:
                    cond_mode, cond_end = "i2v", 1
                elif r < self.cfg.layout.condition.i2v_prob + self.cfg.layout.condition.v2v_prob:
                    ratio = random.uniform(
                        self.cfg.layout.condition.v2v_ratio_min,
                        self.cfg.layout.condition.v2v_ratio_max,
                    )
                    cond_mode, cond_end = "v2v", max(1, min(K - 1, int(K * ratio)))
                else:
                    cond_mode, cond_end = "hc", 0
                min_gap = self._min_gap_steps_for_target_prefix_context(cond_end=cond_end)
                if min_gap > max_gap:
                    raise ValueError(
                        "spatial_memory target_prefix_pixels requires "
                        f"gap_steps >= {min_gap} to use full {self.cfg.spatial_memory.num_context_frames} "
                        f"history pixel frames, but layout.max_gap_sec only allows {max_gap}; "
                        "increase layout.max_gap_sec or reduce spatial_memory.num_context_frames"
                    )
                gap_steps = random.randint(min_gap, max_gap)
            mode_id = {"hc": 0, "i2v": 1, "v2v": 2}[cond_mode]
            meta = torch.tensor([K, gap_steps, mode_id, cond_end], dtype=torch.long, device=self.dist.device)
        else:
            meta = torch.zeros(4, dtype=torch.long, device=self.dist.device)
        broadcast_tensor(meta)
        mode = ["hc", "i2v", "v2v"][int(meta[2].item())]
        return int(meta[0].item()), int(meta[1].item()), mode, int(meta[3].item())

    def _min_gap_steps_for_target_prefix_context(self, *, cond_end: int) -> int:
        cfg = self.cfg.spatial_memory
        if (
            not bool(cfg.enabled)
            or str(getattr(cfg, "context_mode", "retrieval")) != "target_prefix_pixels"
            or not bool(getattr(cfg, "require_full_context", True))
        ):
            return 0

        stride = max(1, int(self.cfg.sample.temporal_stride))
        history_pixels = max(1, int(cfg.num_context_frames))
        sink_count = max(0, int(self.cfg.layout.sink_latent_frames))
        history_latents = max(0, int(self.cfg.layout.history_latent_frames))
        explicit_condition = max(0, int(cond_end)) if history_latents == 0 else 0

        source_floor = 0
        if not bool(cfg.include_sink):
            source_floor = sink_count * stride
        min_target_pixel_start = source_floor + history_pixels
        min_target_start = (min_target_pixel_start + stride - 1) // stride
        target_start_without_gap = sink_count + history_latents + explicit_condition
        return max(0, int(min_target_start) - int(target_start_without_gap))

    def _sample_sigma(self, dtype: torch.dtype, *, latent_frames: int) -> torch.Tensor:
        if self.dist.is_main:
            sigma = torch.rand(1, dtype=dtype, device=self.dist.device)
            if bool(getattr(self.cfg.training, "adaptive_sigma_shift", False)):
                m = self._adaptive_sigma_shift_m(latent_frames)
                sigma = m * sigma / (1.0 + (m - 1.0) * sigma)
        else:
            sigma = torch.zeros(1, dtype=dtype, device=self.dist.device)
        return broadcast_tensor(sigma)

    def _sample_next_forcing_sigma(self, dtype: torch.dtype, *, latent_frames: int) -> torch.Tensor:
        sigma = self._sample_sigma(dtype, latent_frames=int(latent_frames))
        shift = float(self.cfg.next_forcing.sigma_shift)
        if shift != 1.0:
            sigma = (shift * sigma) / (1.0 + (shift - 1.0) * sigma)
        return sigma.clamp_(0.02, 0.98)

    def _broadcast_module_from_rank0(self, module: torch.nn.Module | None) -> None:
        if module is None or not dist.is_initialized():
            return
        for param in module.state_dict().values():
            if torch.is_tensor(param):
                dist.broadcast(param, src=0)

    def _sync_grads_outside_fsdp(self) -> None:
        """All-reduce gradients of trainable parameters that live outside the FSDP tree

        (the HistoryEncoder and the LoRA weights, plus transformer parameters when FSDP is disabled).
        LoRA parameters are injected through a forward hook and the HistoryEncoder is a separate module,
        so FSDP covers neither; without this all-reduce each rank would train its own copy.
        This is a no-op when world_size == 1.
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        if self.history_encoder is not None:
            self._allreduce_grads([p for p in self.history_encoder.parameters() if p.requires_grad])
        if self.components.lora_manager is not None:
            self._allreduce_grads(self.components.lora_manager.get_trainable_parameters())
        from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
        if not isinstance(self.components.transformer, _FSDP):
            self._allreduce_grads(
                [p for p in self.components.transformer.parameters() if p.requires_grad]
            )

    def _allreduce_grads(self, params: list[torch.nn.Parameter]) -> None:
        if not dist.is_initialized():
            return
        world = float(dist.get_world_size())
        for param in params:
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world)

    def _adaptive_sigma_shift_m(self, latent_frames: int | float) -> float:
        frame_lo = int(getattr(self.cfg.training, "adaptive_shift_frame_lo", 8))
        frame_hi = int(getattr(self.cfg.training, "adaptive_shift_frame_hi", 121))
        m_lo = float(getattr(self.cfg.training, "adaptive_shift_m_lo", 5.0))
        m_hi = float(getattr(self.cfg.training, "adaptive_shift_m_hi", 30.0))
        if frame_hi <= frame_lo:
            return m_lo
        t = (float(latent_frames) - float(frame_lo)) / float(frame_hi - frame_lo)
        t = max(0.0, min(1.0, t))
        return m_lo + t * (m_hi - m_lo)

    def _sample_control_modes(self) -> list[str]:
        if self.dist.is_main:
            idx = random.choices(
                range(len(self.cfg.control.candidates)),
                weights=self.cfg.control.probs,
                k=1,
            )[0]
            meta = torch.tensor([idx], dtype=torch.long, device=self.dist.device)
        else:
            meta = torch.zeros(1, dtype=torch.long, device=self.dist.device)
        broadcast_tensor(meta)
        return list(self.cfg.control.candidates[int(meta.item())])

    def _should_validate(self, step: int) -> bool:
        val = self.cfg.validation
        if not val.enabled or step <= 0:
            return False
        if val.first_step > 0 and step == val.first_step:
            return True
        return step >= val.first_step and step % val.interval == 0

    def _validation_cond_end(self, mode_cfg: ValidationModeConfig, K: int) -> int:
        condition = mode_cfg.layout.condition
        configured = int(mode_cfg.layout.condition_latent_frames)
        if condition == "hc":
            return 0
        if configured <= 0:
            raise ValueError(f"validation condition {condition} needs condition_latent_frames > 0")
        return configured

    def _build_control_kwargs(
        self,
        *,
        metadata: dict[str, Any],
        control_modes: list[str],
        target_t_indices: torch.Tensor,
        condition_t_indices: torch.Tensor | None = None,
        history_t_indices: torch.Tensor | None = None,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        kwargs: dict[str, torch.Tensor] = {}
        if "action" in control_modes:
            cam_c2w = metadata.get("cam_c2w")
            if cam_c2w is not None and _as_bool(metadata.get("has_camera", False)):
                kwargs["action_vectors"] = build_action_vectors(
                    cam_c2w=cam_c2w,
                    target_latent_indices=target_t_indices,
                    action_scale=self.cfg.control.action_scale,
                    temporal_stride=self.cfg.sample.temporal_stride,
                    device=self.dist.device,
                    dtype=dtype,
                )
                if condition_t_indices is not None and condition_t_indices.numel() > 0:
                    kwargs["action_condition_vectors"] = build_action_vectors(
                        cam_c2w=cam_c2w,
                        target_latent_indices=condition_t_indices,
                        action_scale=self.cfg.control.action_scale,
                        temporal_stride=self.cfg.sample.temporal_stride,
                        device=self.dist.device,
                        dtype=dtype,
                    )
                if (
                    bool(getattr(self.cfg.control, "action_history_memory", False))
                    and history_t_indices is not None
                    and history_t_indices.numel() > 0
                ):
                    kwargs["action_history_vectors"] = build_action_vectors(
                        cam_c2w=cam_c2w,
                        target_latent_indices=history_t_indices,
                        action_scale=self.cfg.control.action_scale,
                        temporal_stride=self.cfg.sample.temporal_stride,
                        device=self.dist.device,
                        dtype=dtype,
                    )
        return kwargs

    def _build_pixel_prefix_control_kwargs(
        self,
        *,
        metadata: dict[str, Any],
        control_modes: list[str],
        prefix_pixel_frames: int,
        target_latent_frames: int,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        stride = int(self.cfg.sample.temporal_stride)
        return self._build_explicit_pixel_control_kwargs(
            metadata=metadata,
            control_modes=control_modes,
            target_pixel_start=int(prefix_pixel_frames) + stride - 1,
            target_latent_frames=target_latent_frames,
            nearby_pixel=int(prefix_pixel_frames) - 1,
            dtype=dtype,
        )

    def _build_explicit_pixel_control_kwargs(
        self,
        *,
        metadata: dict[str, Any],
        control_modes: list[str],
        target_pixel_start: int,
        target_latent_frames: int,
        nearby_pixel: int,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        kwargs: dict[str, torch.Tensor] = {}
        if "action" not in control_modes:
            return kwargs
        cam_c2w = metadata.get("cam_c2w")
        if cam_c2w is None or not _as_bool(metadata.get("has_camera", False)):
            return kwargs

        target_pixels = [
            int(target_pixel_start) + index * int(self.cfg.sample.temporal_stride)
            for index in range(int(target_latent_frames))
        ]
        kwargs["action_vectors"] = build_action_vectors_from_pixel_indices(
            cam_c2w=cam_c2w,
            pixel_indices=target_pixels,
            previous_pixel_index=nearby_pixel,
            action_scale=self.cfg.control.action_scale,
            device=self.dist.device,
            dtype=dtype,
        )
        kwargs["action_condition_vectors"] = build_action_vectors_from_pixel_indices(
            cam_c2w=cam_c2w,
            pixel_indices=[nearby_pixel],
            previous_pixel_index=max(0, nearby_pixel - int(self.cfg.sample.temporal_stride)),
            action_scale=self.cfg.control.action_scale,
            device=self.dist.device,
            dtype=dtype,
        )
        return kwargs

    def _split_segments(self, latent_full: torch.Tensor, *, K: int, gap_steps: int, cond_end: int):
        sink_count = self.cfg.layout.sink_latent_frames
        N = self.cfg.layout.history_latent_frames
        if N > 0 and cond_end > N:
            raise ValueError(f"condition_latent_frames={cond_end} > history_latent_frames={N}")
        hist_start = sink_count + gap_steps
        hist_end = hist_start + N
        cond_start = hist_end
        target_start = self._segment_target_start(gap_steps=gap_steps, cond_end=cond_end)
        tgt_end = target_start + K
        if latent_full.shape[2] < tgt_end:
            raise ValueError(
                f"latent T={latent_full.shape[2]} < required={tgt_end}; check layout frame requirements"
            )
        sink = latent_full[:, :, :sink_count].contiguous() if sink_count > 0 else None
        history = latent_full[:, :, hist_start:hist_end].contiguous() if N > 0 else None
        if cond_end <= 0:
            nearby = None
        elif history is not None:
            nearby = history[:, :, -cond_end:].contiguous()
        else:
            nearby = latent_full[:, :, cond_start:target_start].contiguous()
        target = latent_full[:, :, target_start:tgt_end].contiguous()
        return sink, history, nearby, target, target_start

    def _segment_target_start(self, *, gap_steps: int, cond_end: int) -> int:
        sink_count = int(self.cfg.layout.sink_latent_frames)
        history_latents = int(self.cfg.layout.history_latent_frames)
        explicit_condition = int(cond_end) if history_latents == 0 else 0
        return sink_count + int(gap_steps) + history_latents + explicit_condition

    def _maybe_fix_condition_prefix_pixels(
        self,
        video_pixels: torch.Tensor,
        *,
        target_start: int,
    ) -> tuple[torch.Tensor, bool]:
        drift = self.cfg.anti_drift.drift
        prob = (
            float(getattr(drift, "condition_prefix_last_context_frame_prob", 0.0))
            if bool(drift.enabled)
            else 0.0
        )
        if prob <= 0.0 or random.random() >= prob:
            return video_pixels, False

        stride = int(self.cfg.sample.temporal_stride)
        target_pixel_start = int(target_start) * stride
        fixed_idx = target_pixel_start - 1
        if fixed_idx < 0:
            return video_pixels, False

        out = video_pixels.clone()
        if out.dim() == 5:
            # [B, F, C, H, W], as returned by the training dataloader.
            frame_count = int(out.shape[1])
            if target_pixel_start > frame_count:
                return video_pixels, False
            fixed = out[:, fixed_idx:fixed_idx + 1].clone()
            out[:, :target_pixel_start] = fixed.expand(-1, target_pixel_start, -1, -1, -1)
            return out, True

        if out.dim() == 4 and int(out.shape[0]) == 3:
            # [C, F, H, W]
            frame_count = int(out.shape[1])
            if target_pixel_start > frame_count:
                return video_pixels, False
            fixed = out[:, fixed_idx:fixed_idx + 1].clone()
            out[:, :target_pixel_start] = fixed.expand(-1, target_pixel_start, -1, -1)
            return out, True

        if out.dim() == 4:
            # [F, C, H, W]
            frame_count = int(out.shape[0])
            if target_pixel_start > frame_count:
                return video_pixels, False
            fixed = out[fixed_idx:fixed_idx + 1].clone()
            out[:target_pixel_start] = fixed.expand(target_pixel_start, -1, -1, -1)
            return out, True

        return video_pixels, False

    def _build_spatial_context(
        self,
        *,
        latent_full: torch.Tensor,
        video_pixels: torch.Tensor,
        metadata: dict[str, Any],
        target_start: int,
        K: int,
        cond_end: int,
        target_rope_t_indices: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        cfg = self.cfg.spatial_memory
        if not cfg.enabled:
            return None
        if float(cfg.dropout) > 0.0 and random.random() < float(cfg.dropout):
            return None
        if not _as_bool(metadata.get("has_camera", False)):
            return None
        cam_c2w = metadata.get("cam_c2w")
        intrinsic = metadata.get("intrinsic")
        if cam_c2w is None or intrinsic is None:
            return None
        context_mode = str(getattr(cfg, "context_mode", "retrieval"))
        if context_mode == "target_prefix_pixels":
            return self._build_target_prefix_pixel_spatial_context(
                video_pixels=video_pixels,
                metadata=metadata,
                cam_c2w=cam_c2w,
                intrinsic=intrinsic,
                target_start=target_start,
                K=K,
                target_rope_t_indices=target_rope_t_indices,
            )
        if context_mode == "vigeo_prefix_last_frame":
            # Training uses the explicit pixel-prefix builder. Validation uses
            # the rollout bank builder so generated frames can become the next prefix.
            return None
        if context_mode != "retrieval":
            raise ValueError(f"unsupported spatial_memory.context_mode={context_mode!r}")

        allowed: list[int] = []
        nearby_allowed: set[int] = set()
        stride = max(1, int(cfg.cache_stride))
        sink_count = int(self.cfg.layout.sink_latent_frames)
        recent_cutoff = int(target_start) - int(cfg.skip_recent_latents)
        if bool(cfg.include_sink) and sink_count > 0 and not bool(self.cfg.layout.sink_remote):
            allowed.extend(range(0, sink_count, stride))

        N = int(self.cfg.layout.history_latent_frames)
        hist_end = int(target_start) - int(cond_end)
        hist_start = max(sink_count, hist_end - N)
        if N > 0 and hist_end > hist_start:
            allowed.extend(range(hist_start, hist_end, stride))

        if bool(cfg.include_nearby) and cond_end > 0:
            cond_start = max(sink_count, int(target_start) - int(cond_end))
            nearby_allowed = set(range(cond_start, int(target_start), stride))
            allowed.extend(nearby_allowed)

        allowed = [
            idx
            for idx in sorted(set(allowed))
            if 0 <= idx < int(target_start)
            and (idx < recent_cutoff or idx in nearby_allowed)
        ]
        if not allowed:
            return None

        pixel_height, pixel_width = self._pixel_hw_from_video(video_pixels)
        depth_by_latent_index = self._build_spatial_depths(
            video_pixels=video_pixels,
            metadata=metadata,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            allowed=allowed,
        )

        retrieval_views = max(1, int(cfg.retrieval_views))
        if retrieval_views == 1:
            target_indices = [int(target_start + K - 1)]
        else:
            offsets = torch.linspace(0, max(0, int(K) - 1), retrieval_views, device=self.dist.device)
            target_indices = [int(target_start + int(round(float(x.item())))) for x in offsets]

        context = build_retrieved_latent_context(
            latent_full=latent_full,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            allowed_latent_indices=allowed,
            target_latent_indices=target_indices,
            height=pixel_height,
            width=pixel_width,
            temporal_stride=int(self.cfg.sample.temporal_stride),
            num_context_frames=int(cfg.num_context_frames),
            downsample=int(cfg.downsample),
            constant_depth=float(cfg.constant_depth),
            depth_by_latent_index=depth_by_latent_index,
            retrieval_max_coverage=bool(cfg.retrieval_max_coverage),
            retrieval_depth_threshold=float(cfg.retrieval_depth_threshold),
        )
        if context is None:
            return None
        spatial_latent = context.latent.to(device=self.dist.device, dtype=latent_full.dtype)
        if bool(cfg.use_warped_context):
            warped_pixels = forward_warp_video_to_targets(
                video_pixels=video_pixels,
                source_latent_indices=context.source_latent_indices,
                target_latent_indices=context.target_latent_indices,
                cam_c2w=cam_c2w.to(device=self.dist.device, dtype=torch.float32),
                intrinsic=intrinsic.to(device=self.dist.device, dtype=torch.float32),
                depth_by_latent_index=depth_by_latent_index,
                height=pixel_height,
                width=pixel_width,
                temporal_stride=int(self.cfg.sample.temporal_stride),
                constant_depth=float(cfg.constant_depth),
                depth_threshold=min(float(cfg.retrieval_depth_threshold), 1e-3),
            )
            if warped_pixels is not None:
                spatial_latent = self._encode_spatial_context_pixels(warped_pixels)
        return self._maybe_force_spatial_all_invalid({
            "latent": spatial_latent,
            "source_indices": context.source_latent_indices,
            "target_indices": context.target_latent_indices,
        }, metadata=metadata)

    def _build_target_prefix_pixel_spatial_context(
        self,
        *,
        video_pixels: torch.Tensor,
        metadata: dict[str, Any],
        cam_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
        target_start: int,
        K: int,
        target_rope_t_indices: torch.Tensor | None,
    ) -> dict[str, Any] | None:
        cfg = self.cfg.spatial_memory
        stride = int(self.cfg.sample.temporal_stride)
        pixel_height, pixel_width = self._pixel_hw_from_video(video_pixels)
        target_pixel_start = int(target_start) * stride
        target_pixel_count = 1 + max(0, int(K) - 1) * stride
        target_pixel_indices = list(range(target_pixel_start, target_pixel_start + target_pixel_count))

        video_frames = self._video_pixel_frame_count(video_pixels)
        cam_frames = int(cam_c2w.shape[1] if cam_c2w.dim() == 4 else cam_c2w.shape[0])
        max_frames = min(video_frames, cam_frames)
        if not target_pixel_indices or target_pixel_indices[-1] >= max_frames:
            return None

        history_pixels = max(1, int(cfg.num_context_frames))
        source_end = target_pixel_start
        source_floor = 0
        if not bool(cfg.include_sink):
            source_floor = max(0, int(self.cfg.layout.sink_latent_frames) * stride)
        source_start = max(source_floor, source_end - history_pixels)
        source_pixel_indices = list(range(source_start, source_end))
        if not source_pixel_indices:
            return None
        if bool(getattr(cfg, "require_full_context", True)) and len(source_pixel_indices) < history_pixels:
            return None

        depth_by_frame_index = self._build_spatial_depths_for_pixel_frames(
            video_pixels=video_pixels,
            metadata=metadata,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            frame_indices=source_pixel_indices,
        )
        warp_result = forward_warp_pixel_sources_to_pixel_targets(
            video_pixels=video_pixels,
            source_pixel_indices=source_pixel_indices,
            target_pixel_indices=target_pixel_indices,
            cam_c2w=cam_c2w.to(device=self.dist.device, dtype=torch.float32),
            intrinsic=intrinsic.to(device=self.dist.device, dtype=torch.float32),
            depth_by_frame_index=depth_by_frame_index,
            height=pixel_height,
            width=pixel_width,
            constant_depth=float(cfg.constant_depth),
            depth_threshold=min(float(cfg.retrieval_depth_threshold), 1e-3),
            fill_value=None,
            return_coverage=True,
        )
        if warp_result is None:
            return None
        warped_pixels, coverage_pixels = warp_result

        spatial_latent = self._encode_spatial_context_video(warped_pixels, expected_latent_frames=int(K))
        mask_patch = self._build_spatial_mask_patch(
            coverage_pixels=coverage_pixels,
            spatial_latent=spatial_latent,
        )
        if target_rope_t_indices is None:
            rope_t_indices: list[float] = list(range(int(target_start), int(target_start) + int(K)))
        else:
            rope_t_indices = [float(x) for x in target_rope_t_indices.detach().cpu().tolist()]
        return self._maybe_force_spatial_all_invalid({
            "latent": spatial_latent,
            "mask_patch": mask_patch,
            "source_indices": source_pixel_indices,
            "target_indices": list(range(int(target_start), int(target_start) + int(K))),
            "source_pixel_indices": source_pixel_indices,
            "target_pixel_indices": target_pixel_indices,
            "rope_t_indices": rope_t_indices,
        }, metadata=metadata)

    def _build_vigeo_prefix_last_frame_spatial_context(
        self,
        *,
        video_pixels: torch.Tensor,
        metadata: dict[str, Any],
        K: int,
        target_rope_t_indices: torch.Tensor | None,
    ) -> dict[str, Any] | None:
        cfg = self.cfg.spatial_memory
        if not bool(cfg.enabled):
            return None
        if float(cfg.dropout) > 0.0 and random.random() < float(cfg.dropout):
            return None
        if not _as_bool(metadata.get("has_camera", False)):
            return None
        cam_c2w = metadata.get("cam_c2w")
        intrinsic = metadata.get("intrinsic")
        if cam_c2w is None or intrinsic is None:
            return None

        scale_context_frames = self._vigeo_prefix_pixel_frames()
        target_prefix_frames = self._vigeo_target_prefix_pixel_frames()
        if scale_context_frames < self._vigeo_motion_pixel_frames():
            raise ValueError(
                "vigeo_prefix_last_frame prefix is shorter than its VAE motion window"
            )
        source_context_indices = self._vigeo_scale_context_pixel_indices()
        source_pixel_index = target_prefix_frames - 1
        stride = int(self.cfg.sample.temporal_stride)
        target_pixel_count = int(K) * stride
        target_pixel_indices = list(
            range(target_prefix_frames, target_prefix_frames + target_pixel_count)
        )
        available_frames = self._video_pixel_frame_count(video_pixels)
        if target_pixel_indices[-1] >= available_frames:
            raise ValueError(
                f"ViGeo spatial target ends at pixel {target_pixel_indices[-1]}, "
                f"but sample has {available_frames} frames"
            )

        geometry = self._get_vigeo_geometry().infer_prefix_geometry(
            video_pixels=video_pixels,
            frame_indices=source_context_indices,
            cam_c2w=cam_c2w,
        )
        pixel_height, pixel_width = self._pixel_hw_from_video(video_pixels)
        warp_intrinsics = self._vigeo_warp_intrinsics(
            intrinsic=intrinsic,
            cam_c2w=cam_c2w,
            source_pixel_index=source_pixel_index,
            source_intrinsic=geometry.source_intrinsic,
            height=pixel_height,
            width=pixel_width,
            use_vigeo_for_all_frames=self._is_uncalibrated_intrinsic_source(metadata),
        )
        warp_result = forward_warp_pixel_sources_to_pixel_targets(
            video_pixels=video_pixels,
            source_pixel_indices=[source_pixel_index],
            target_pixel_indices=target_pixel_indices,
            cam_c2w=cam_c2w.to(device=self.dist.device, dtype=torch.float32),
            intrinsic=warp_intrinsics,
            depth_by_frame_index={source_pixel_index: geometry.source_depth},
            height=pixel_height,
            width=pixel_width,
            constant_depth=float(cfg.constant_depth),
            depth_threshold=min(float(cfg.retrieval_depth_threshold), 1e-3),
            fill_value=None,
            return_coverage=True,
        )
        if warp_result is None:
            return None
        warped_pixels, coverage_pixels = warp_result
        previous_pixels = self._select_video_pixel_frames(
            video_pixels, [source_pixel_index]
        )
        spatial_latent, vae_prefix_latent = self._encode_spatial_continuation_video(
            previous_pixels=previous_pixels,
            target_pixels=warped_pixels,
            expected_target_latent_frames=int(K),
        )
        mask_patch = self._build_spatial_mask_patch(
            coverage_pixels=coverage_pixels,
            spatial_latent=spatial_latent,
        )
        if target_rope_t_indices is None:
            rope_t_indices = list(range(2, 2 + int(K)))
        else:
            rope_t_indices = [
                float(value)
                for value in target_rope_t_indices.detach().cpu().tolist()
            ]
        return self._maybe_force_spatial_all_invalid(
            {
                "latent": spatial_latent,
                "mask_patch": mask_patch,
                "source_indices": [source_pixel_index],
                "target_indices": rope_t_indices,
                "scale_context_pixel_indices": source_context_indices,
                "source_pixel_indices": [source_pixel_index],
                "vae_prefix_pixel_index": source_pixel_index,
                "vae_prefix_latent": vae_prefix_latent,
                "target_pixel_indices": target_pixel_indices,
                "rope_t_indices": rope_t_indices,
                "vigeo_scale": float(geometry.scale),
                "vigeo_pairwise_scales": geometry.pairwise_scales,
                "vigeo_source_intrinsic": geometry.source_intrinsic.detach().cpu(),
            },
            metadata=metadata,
        )

    def _get_vigeo_geometry(self) -> ViGeoGeometryEstimator:
        if self.vigeo_geometry is None:
            cfg = self.cfg.spatial_memory
            device = (
                self.dist.device
                if str(cfg.vigeo_device) == "auto"
                else torch.device(str(cfg.vigeo_device))
            )
            self.vigeo_geometry = ViGeoGeometryEstimator(
                repo_path=cfg.vigeo_repo_path,
                checkpoint=str(cfg.vigeo_checkpoint),
                device=device,
                num_tokens=int(cfg.vigeo_num_tokens),
            )
            rank0_print(
                self.dist,
                "[SpatialMemory]",
                f"loading ViGeo checkpoint={cfg.vigeo_checkpoint} repo={cfg.vigeo_repo_path}",
            )
        return self.vigeo_geometry

    def _vigeo_warp_intrinsics(
        self,
        *,
        intrinsic: torch.Tensor,
        cam_c2w: torch.Tensor,
        source_pixel_index: int,
        source_intrinsic: torch.Tensor,
        height: int,
        width: int,
        use_vigeo_for_all_frames: bool,
    ) -> torch.Tensor:
        cameras = cam_c2w
        if cameras.dim() == 3:
            cameras = cameras.unsqueeze(0)
        batch, frames = int(cameras.shape[0]), int(cameras.shape[1])
        K = intrinsic.to(device=self.dist.device, dtype=torch.float32)
        if K.dim() == 2:
            K = K.unsqueeze(0)
        if K.dim() == 3:
            K = pixel_intrinsics(K, height=height, width=width)
            K = K.unsqueeze(1).expand(-1, frames, -1, -1).clone()
        elif K.dim() == 4:
            K = pixel_intrinsics(
                K.reshape(-1, 3, 3), height=height, width=width
            ).reshape(K.shape)
            if int(K.shape[1]) < frames:
                K = torch.cat(
                    [K, K[:, -1:].expand(-1, frames - int(K.shape[1]), -1, -1)],
                    dim=1,
                )
            else:
                K = K[:, :frames].clone()
        else:
            raise ValueError(f"unexpected intrinsic shape {tuple(intrinsic.shape)}")
        if int(K.shape[0]) == 1 and batch > 1:
            K = K.expand(batch, -1, -1, -1).clone()
        source_K = source_intrinsic.to(device=K.device, dtype=K.dtype)
        if int(source_K.shape[0]) == 1 and batch > 1:
            source_K = source_K.expand(batch, -1, -1)
        if use_vigeo_for_all_frames:
            K = source_K.unsqueeze(1).expand(-1, frames, -1, -1).clone()
        else:
            source_index = max(0, min(int(source_pixel_index), frames - 1))
            K[:, source_index] = source_K
        return K.contiguous()

    def _encode_spatial_context_video(
        self,
        warped_pixels: torch.Tensor,
        *,
        expected_latent_frames: int,
    ) -> torch.Tensor:
        warped_pixels = warped_pixels.to(device=self.dist.device, dtype=self.dtype)
        with torch.no_grad():
            latent = self.components.vae_encoder.encode(
                warped_pixels,
                chunk_size=self.cfg.runtime.vae_chunk_size,
                verbose=False,
            )
        latent = latent.to(device=self.dist.device, dtype=self.dtype).contiguous()
        if int(latent.shape[2]) != int(expected_latent_frames):
            raise RuntimeError(
                f"spatial rendered VAE produced {latent.shape[2]} latents, "
                f"expected {expected_latent_frames}; warped_pixels={tuple(warped_pixels.shape)}"
            )
        return latent

    def _encode_spatial_continuation_video(
        self,
        *,
        previous_pixels: torch.Tensor,
        target_pixels: torch.Tensor,
        expected_target_latent_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous = self._video_pixels_to_bcfhw(previous_pixels).to(
            device=self.dist.device, dtype=self.dtype
        )
        target = self._video_pixels_to_bcfhw(target_pixels).to(
            device=self.dist.device, dtype=self.dtype
        )
        if int(previous.shape[2]) != 1:
            raise ValueError(
                "spatial continuation requires one previous pixel frame, "
                f"got {previous.shape[2]}"
            )
        expected_target_pixels = (
            int(expected_target_latent_frames) * int(self.cfg.sample.temporal_stride)
        )
        if int(target.shape[2]) != expected_target_pixels:
            raise ValueError(
                "spatial continuation target pixel length mismatch: "
                f"got {target.shape[2]}, expected {expected_target_pixels}"
            )
        if previous.shape[0] != target.shape[0] or previous.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                "spatial continuation prefix/target shape mismatch: "
                f"previous={tuple(previous.shape)} target={tuple(target.shape)}"
            )
        continuation = torch.cat([previous, target], dim=2)
        latent = self._encode_spatial_context_video(
            continuation,
            expected_latent_frames=int(expected_target_latent_frames) + 1,
        )
        return (
            latent[:, :, 1:].contiguous(),
            latent[:, :, :1].contiguous(),
        )

    def _maybe_force_spatial_all_invalid(
        self,
        context: dict[str, Any] | None,
        *,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        if context is None:
            return context
        force_invalid = bool(force) or bool(getattr(self.cfg.spatial_memory, "force_all_invalid", False))
        if not force_invalid:
            return context
        latent = context.get("latent")
        if latent is None:
            return context
        forced = dict(context)
        forced["latent"] = torch.zeros_like(latent)
        mask_patch = forced.get("mask_patch")
        if mask_patch is None:
            B, C, frames, height, width = latent.shape
            patchifier = VideoLatentPatchifier(patch_size=1)
            pt, ph, pw = patchifier.patch_size
            if frames % pt != 0 or height % ph != 0 or width % pw != 0:
                raise ValueError(
                    f"spatial latent shape {(frames, height, width)} is not divisible by patch size {(pt, ph, pw)}"
                )
            tokens = (frames // pt) * (height // ph) * (width // pw)
            channel_patch = int(C) * int(pt) * int(ph) * int(pw)
            mask_patch = torch.zeros(
                int(B),
                int(tokens),
                int(channel_patch),
                device=latent.device,
                dtype=latent.dtype,
            )
        else:
            mask_patch = torch.zeros_like(mask_patch)
        forced["mask_patch"] = mask_patch
        return forced



    def _is_uncalibrated_intrinsic_source(self, metadata: dict[str, Any]) -> bool:
        src = str(metadata.get("source") or "").lower()
        if src == "wbench_navi":
            return bool(getattr(self.cfg.spatial_memory, "wbench_fitted_intrinsic", True))
        if src == "custom_i2v":
            return not bool(metadata.get("has_real_intrinsic"))
        return False

    def _build_spatial_mask_patch(self, *, coverage_pixels: torch.Tensor, spatial_latent: torch.Tensor) -> torch.Tensor:
        B, C, frames, height, width = spatial_latent.shape
        if coverage_pixels.dim() != 5 or int(coverage_pixels.shape[0]) != B or int(coverage_pixels.shape[1]) != 1:
            raise ValueError(
                f"coverage_pixels must be [B,1,T,H,W] with B={B}, got {tuple(coverage_pixels.shape)}"
            )
        patchifier = VideoLatentPatchifier(patch_size=1)
        pt, ph, pw = patchifier.patch_size
        if frames % pt != 0 or height % ph != 0 or width % pw != 0:
            raise ValueError(
                f"spatial latent shape {(frames, height, width)} is not divisible by patch size {(pt, ph, pw)}"
            )
        # A token is valid when its spatio-temporal bin is mostly covered (> 0.5).
        mask_grid = F.adaptive_avg_pool3d(
            coverage_pixels.to(device=self.dist.device, dtype=torch.float32),
            output_size=(frames // pt, height // ph, width // pw),
        ).gt_(0.5).to(dtype=torch.float32)
        mask_flat = mask_grid[:, 0].reshape(B, -1, 1)
        channel_patch = int(C) * int(pt) * int(ph) * int(pw)
        return mask_flat.expand(B, -1, channel_patch).to(dtype=spatial_latent.dtype).contiguous()

    def _encode_spatial_context_pixels(self, warped_pixels: torch.Tensor) -> torch.Tensor:
        warped_pixels = warped_pixels.to(device=self.dist.device, dtype=self.dtype)
        latents = []
        with torch.no_grad():
            for frame_idx in range(int(warped_pixels.shape[2])):
                latent = self.components.vae_encoder.encode(
                    warped_pixels[:, :, frame_idx : frame_idx + 1],
                    chunk_size=self.cfg.runtime.vae_chunk_size,
                    verbose=False,
                )
                latents.append(latent[:, :, :1].to(device=self.dist.device, dtype=self.dtype))
        return torch.cat(latents, dim=2).contiguous()

    def _build_spatial_depths(
        self,
        *,
        video_pixels: torch.Tensor,
        metadata: dict[str, Any],
        cam_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
        allowed: list[int],
    ) -> dict[int, torch.Tensor] | None:
        cfg = self.cfg.spatial_memory
        backend = str(cfg.depth_backend)
        if backend == "constant":
            return None
        if backend == "metadata":
            depth = metadata.get("depth")
            if depth is None:
                return None
            return _select_depth_by_latent_index(depth, allowed, int(self.cfg.sample.temporal_stride))
        if backend != "da3":
            raise ValueError(f"unsupported spatial_memory.depth_backend={backend!r}")

        if self.da3_depth is None:
            device = self.dist.device if str(cfg.da3_device) == "auto" else torch.device(str(cfg.da3_device))
            self.da3_depth = DA3DepthEstimator(
                repo_path=cfg.da3_repo_path,
                model_name=str(cfg.da3_model_name),
                cache_dir=cfg.da3_cache_dir,
                device=device,
                process_res=int(cfg.da3_process_res),
                process_res_method=str(cfg.da3_process_res_method),
                align_to_input_scale=bool(cfg.da3_align_to_input_scale),
            )
            rank0_print(
                self.dist,
                "[SpatialMemory]",
                f"loading DA3 depth backend model={cfg.da3_model_name} repo={cfg.da3_repo_path}",
            )
        pixel_height, pixel_width = self._pixel_hw_from_video(video_pixels)
        return self.da3_depth.infer_latent_depths(
            video_pixels=video_pixels,
            latent_indices=allowed,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            height=pixel_height,
            width=pixel_width,
            temporal_stride=int(self.cfg.sample.temporal_stride),
        )

    def _build_spatial_depths_for_pixel_frames(
        self,
        *,
        video_pixels: torch.Tensor,
        metadata: dict[str, Any],
        cam_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
        frame_indices: list[int],
    ) -> dict[int, torch.Tensor] | None:
        cfg = self.cfg.spatial_memory
        backend = str(cfg.depth_backend)
        if backend == "constant":
            return None
        if backend == "metadata":
            depth = metadata.get("depth")
            if depth is None:
                return None
            return _select_depth_by_frame_index(depth, frame_indices)
        if backend != "da3":
            raise ValueError(f"unsupported spatial_memory.depth_backend={backend!r}")

        if self.da3_depth is None:
            device = self.dist.device if str(cfg.da3_device) == "auto" else torch.device(str(cfg.da3_device))
            self.da3_depth = DA3DepthEstimator(
                repo_path=cfg.da3_repo_path,
                model_name=str(cfg.da3_model_name),
                cache_dir=cfg.da3_cache_dir,
                device=device,
                process_res=int(cfg.da3_process_res),
                process_res_method=str(cfg.da3_process_res_method),
                align_to_input_scale=bool(cfg.da3_align_to_input_scale),
            )
            rank0_print(
                self.dist,
                "[SpatialMemory]",
                f"loading DA3 depth backend model={cfg.da3_model_name} repo={cfg.da3_repo_path}",
            )
        pixel_height, pixel_width = self._pixel_hw_from_video(video_pixels)
        return self.da3_depth.infer_frame_depths(
            video_pixels=video_pixels,
            frame_indices=frame_indices,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            height=pixel_height,
            width=pixel_width,
        )

    def _uses_parallel_history_degradation(self) -> bool:
        drift = self.cfg.anti_drift.drift
        return (
            bool(drift.enabled)
            and getattr(drift, "history_corrupt_prob", None) is not None
        )

    def _sample_history_degradation_mode(self) -> str:
        drift = self.cfg.anti_drift.drift
        drop_prob = float(self.cfg.memory.drop_prob)
        corrupt_prob = float(getattr(drift, "history_corrupt_prob", 0.0) or 0.0)
        if self.dist.is_main:
            r = torch.rand(1, device=self.dist.device)
            bounds = torch.tensor(
                [drop_prob, drop_prob + corrupt_prob],
                device=self.dist.device,
                dtype=torch.float32,
            )
            if r.item() < bounds[0].item():
                mode_id = 1
            elif r.item() < bounds[1].item():
                mode_id = 2
            else:
                mode_id = 0
            mode = torch.tensor([mode_id], device=self.dist.device, dtype=torch.long)
        else:
            mode = torch.zeros(1, device=self.dist.device, dtype=torch.long)
        broadcast_tensor(mode)
        return ["clean", "drop", "corrupt"][int(mode.item())]

    def _prepare_history(
        self,
        history: torch.Tensor,
        *,
        sigma: float,
        degradation_mode: str | None = None,
        force_clean: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        mode = "clean"
        inject_latent = False
        inject_history = False
        out = history

        if force_clean:
            return out, {"anti_drift": "clean_freeze_prefix", "inject_latent": False, "inject_history": False}

        if degradation_mode == "drop":
            mode = "drop"
        elif degradation_mode == "corrupt":
            mode = "corrupt"
            drift = self.cfg.anti_drift.drift
            out = corrupt_history_latents_helios(
                out,
                noise_mode_prob=drift.noise_mode_prob,
                corrupt_ratio=drift.corrupt_ratio,
                clean_prob=drift.clean_prob,
                downsample_min=drift.downsample_min,
                downsample_max=drift.downsample_max,
                saturation_clean_prob=drift.saturation_clean_prob,
                is_frame_independent=False,
                is_keep_x0=drift.keep_x0,
            )
        elif degradation_mode is None and self.error_bank is not None and self.error_bank.is_warm() and random.random() >= self.cfg.anti_drift.error_bank.bank_skip_prob:
            flags = self.error_bank.roll_inject_flags(ignore_clean=True)
            mode = "bank"
            inject_latent = bool(flags["latent"])
            inject_history = bool(flags["history"])
        elif degradation_mode is None and self.cfg.anti_drift.drift.enabled:
            mode = "drift"
            drift = self.cfg.anti_drift.drift
            out = corrupt_history_latents_helios(
                out,
                noise_mode_prob=drift.noise_mode_prob,
                corrupt_ratio=drift.corrupt_ratio,
                clean_prob=drift.clean_prob,
                downsample_min=drift.downsample_min,
                downsample_max=drift.downsample_max,
                saturation_clean_prob=drift.saturation_clean_prob,
                is_frame_independent=False,
                is_keep_x0=drift.keep_x0,
            )

        if degradation_mode is not None and self.error_bank is not None and self.error_bank.is_warm() and random.random() >= self.cfg.anti_drift.error_bank.bank_skip_prob:
            flags = self.error_bank.roll_inject_flags(ignore_clean=True)
            inject_latent = bool(flags["latent"])

        if inject_history:
            delta = self._sample_history_error(out, sigma)
            if delta is not None:
                out = out.clone()
                out[:, :, -delta.shape[2]:] = out[:, :, -delta.shape[2]:] + delta

        return out, {
            "anti_drift": mode,
            "inject_latent": inject_latent,
            "inject_history": inject_history,
        }

    def _sample_history_error(self, history: torch.Tensor, sigma: float) -> torch.Tensor | None:
        if self.error_bank is None:
            return None
        X = self.error_bank.sample_X(p_decay=0.1, max_X=max(1, history.shape[2] - 1))
        delta = self.error_bank.sample_x_frames(
            X=X,
            target_shape=(history.shape[0], history.shape[1], X, history.shape[3], history.shape[4]),
            device=self.dist.device,
            timestep=sigma,
            dtype=history.dtype,
        )
        if delta is None:
            return None
        weights = torch.linspace(0.5, 1.0, X, device=self.dist.device, dtype=history.dtype).view(1, 1, X, 1, 1)
        return self.error_bank.gamma * weights * delta

    def _inject_target_error(self, target_clean: torch.Tensor, K: int, sigma: float) -> torch.Tensor:
        if self.error_bank is None:
            return target_clean
        delta = self.error_bank.sample_one(
            K=K,
            timestep=sigma,
            target_shape=target_clean.shape,
            device=self.dist.device,
            dtype=target_clean.dtype,
        )
        return target_clean if delta is None else target_clean + self.error_bank.gamma * delta

    def _push_error_bank(
        self,
        target_noisy: torch.Tensor,
        sigma_view: torch.Tensor,
        pred_velocity: torch.Tensor,
        target_clean: torch.Tensor,
        sigma: float,
    ) -> None:
        if self.error_bank is None:
            return
        with torch.no_grad():
            x_hat_clean = target_noisy - sigma_view * pred_velocity.detach()
            residual = (x_hat_clean - target_clean).to(dtype=self.error_bank.dtype)
            self.error_bank.push(residual, timestep=sigma)

    def _build_error_bank(self) -> ErrorBank | None:
        cfg = self.cfg.anti_drift.error_bank
        if not cfg.enabled:
            return None
        return ErrorBank(
            buffer_k=cfg.buffer_k,
            num_grids=cfg.num_grids,
            warmup_iter=cfg.warmup_iter,
            latent_prob=cfg.target_latent_prob,
            clean_prob=cfg.bank_skip_prob,
            clean_buffer_update_prob=cfg.clean_buffer_update_prob,
            replacement_strategy=cfg.replacement_strategy,
            gamma=cfg.gamma,
            history_prob=cfg.history_prob,
            spatial_prob=cfg.spatial_prob,
            nearby_prob=cfg.nearby_prob,
            error_modulate_factor=cfg.modulate_factor,
            device=self.dist.device,
            dtype=self.dtype,
        )

    def _required_latents_for_max_K(self) -> int:
        max_gap = int(self.cfg.layout.max_gap_sec * self.cfg.sample.fps / self.cfg.sample.temporal_stride)
        max_k = max(self._active_output_latent_frames())
        next_extra = max_k if self.cfg.next_forcing.enabled else 0
        _sr = self.cfg.dmd.self_rollout if self.cfg.dmd.enabled else None
        gt_extra = (
            int(_sr.max_chunks) * max_k
            if _sr is not None and _sr.enabled and bool(getattr(_sr, "score_gt_context", False))
            else 0
        )
        if self._uses_fixed_no_history_condition_window():
            return self.cfg.layout.sink_latent_frames + max_gap + max_k + 1 + next_extra + gt_extra
        return (
            self.cfg.layout.sink_latent_frames
            + max_gap
            + self.cfg.layout.history_latent_frames
            + self._max_explicit_condition_latents()
            + max_k
            + next_extra
            + gt_extra
        )

    def _active_output_latent_frames(self) -> list[int]:
        pairs = [
            (int(k), float(p))
            for k, p in zip(self.cfg.layout.output.latent_frames, self.cfg.layout.output.probs)
            if float(p) > 0.0
        ]
        if not pairs:
            raise ValueError("layout.output.probs must contain at least one positive value")
        return [k for k, _ in pairs]

    def _max_explicit_condition_latents(self) -> int:
        if self.cfg.layout.history_latent_frames > 0:
            return 0
        if self.cfg.layout.condition.i2v_prob <= 0 and self.cfg.layout.condition.v2v_prob <= 0:
            return 0
        max_k = max(self.cfg.layout.output.latent_frames)
        max_cond = 1 if self.cfg.layout.condition.i2v_prob > 0 else 0
        if self.cfg.layout.condition.v2v_prob > 0:
            max_cond = max(max_cond, max(1, int(max_k * self.cfg.layout.condition.v2v_ratio_max)))
        return max_cond

    def _uses_fixed_no_history_condition_window(self) -> bool:
        return (
            self.cfg.layout.history_latent_frames == 0
            and (self.cfg.layout.condition.i2v_prob > 0 or self.cfg.layout.condition.v2v_prob > 0)
        )

    def _uses_vigeo_prefix_last_frame(self) -> bool:
        return (
            bool(self.cfg.spatial_memory.enabled)
            and str(self.cfg.spatial_memory.context_mode)
            == "vigeo_prefix_last_frame"
            and str(self.cfg.spatial_memory.depth_backend) == "vigeo"
        )

    def _vigeo_prefix_pixel_frames(self) -> int:
        configured = self.cfg.spatial_memory.vigeo_prefix_frames
        return int(
            self.cfg.spatial_memory.num_context_frames
            if configured is None
            else configured
        )

    def _vigeo_target_prefix_pixel_frames(
        self,
        *,
        history_latent_frames: int | None = None,
    ) -> int:
        history_latents = (
            int(self.cfg.layout.history_latent_frames)
            if history_latent_frames is None
            else int(history_latent_frames)
        )
        if history_latents <= 0:
            return self._vigeo_prefix_pixel_frames()
        return 1 + (history_latents - 1) * int(self.cfg.sample.temporal_stride)

    def _vigeo_scale_context_pixel_indices(
        self,
        *,
        history_latent_frames: int | None = None,
    ) -> list[int]:
        context_frames = self._vigeo_prefix_pixel_frames()
        target_prefix_frames = self._vigeo_target_prefix_pixel_frames(
            history_latent_frames=history_latent_frames
        )
        start = target_prefix_frames - context_frames
        if start < 0:
            raise ValueError(
                "ViGeo scale context does not fit before the target: "
                f"context_frames={context_frames}, "
                f"target_prefix_frames={target_prefix_frames}"
            )
        return list(range(start, target_prefix_frames))

    def _vigeo_motion_pixel_frames(self) -> int:
        return int(self.cfg.sample.temporal_stride) + 1

    def _validation_history_latents(self, mode_cfg: ValidationModeConfig) -> int:
        if mode_cfg.layout.history_latent_frames is None:
            return self.cfg.layout.history_latent_frames
        return int(mode_cfg.layout.history_latent_frames)

    def _local_sink_t_offset(self, history_latent_frames: int) -> int:
        return 0

    def _local_memory_t_offset(self, history_latent_frames: int, condition_latent_frames: int) -> int:
        return 1

    def _local_nearby_t_offset(
        self, history_latent_frames: int, condition_latent_frames: int, gap_steps: int = 0
    ) -> int:
        history_latents = max(0, int(history_latent_frames))
        condition_latents = max(0, int(condition_latent_frames))
        if history_latents > 0:
            # With temporal memory enabled, nearby is copied from the tail of
            # history rather than inserted as an extra frame before target.
            return 1 + max(0, history_latents - condition_latents)
        return 1

    def _local_target_t_indices(
        self,
        frames: int,
        *,
        history_latent_frames: int,
        condition_latent_frames: int,
        gap_steps: int = 0,
    ) -> torch.Tensor:
        history_latents = max(0, int(history_latent_frames))
        condition_latents = max(0, int(condition_latent_frames))
        if history_latents > 0:
            start = 1 + history_latents
        else:
            start = 1 + condition_latents
        return torch.arange(start, start + int(frames), device=self.dist.device, dtype=torch.float32)

    def _indices_grid(self, batch: int, frames: int, height: int, width: int, *, t_offset: int) -> torch.Tensor:
        patchifier = VideoLatentPatchifier(patch_size=1)
        shape = VideoLatentShape(batch=batch, channels=1, frames=frames, height=height, width=width)
        coords = patchifier.get_patch_grid_bounds(shape, device=self.dist.device).clone().to(torch.float32)
        coords[:, 0, :, :] += t_offset
        return coords

    def _indices_grid_for_t_indices(
        self,
        batch: int,
        t_indices: list[int | float] | torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        frames = int(t_indices.numel()) if torch.is_tensor(t_indices) else len(t_indices)
        coords = self._indices_grid(batch, frames, height, width, t_offset=0)
        if torch.is_tensor(t_indices):
            t = t_indices.to(device=self.dist.device, dtype=coords.dtype)
        else:
            t = torch.tensor(t_indices, device=self.dist.device, dtype=coords.dtype)
        per_token = t.view(frames, 1, 1).expand(frames, height, width).reshape(-1)
        bounds = torch.stack([per_token, per_token + 1], dim=-1)
        coords[:, 0, :, :] = bounds.unsqueeze(0)
        return coords


def _first_scalar(value: Any) -> Any:
    while isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
    return value


def _select_depth_by_latent_index(
    depth: torch.Tensor,
    latent_indices: list[int],
    temporal_stride: int,
) -> dict[int, torch.Tensor]:
    """Select source-frame depth maps from metadata depth."""
    if not torch.is_tensor(depth):
        raise TypeError(f"metadata depth must be a tensor, got {type(depth)!r}")
    d = depth
    if d.dim() == 3:
        d = d.unsqueeze(0)
    if d.dim() == 4:
        d = d.unsqueeze(2)
    if d.dim() != 5:
        raise ValueError(f"metadata depth must be [B,T,H,W] or [B,T,1,H,W], got {tuple(depth.shape)}")
    out: dict[int, torch.Tensor] = {}
    for latent_idx in sorted({int(i) for i in latent_indices}):
        frame_idx = min(max(0, latent_idx * int(temporal_stride)), d.shape[1] - 1)
        out[int(latent_idx)] = d[:, frame_idx].contiguous()
    return out


def _select_depth_by_frame_index(
    depth: torch.Tensor,
    frame_indices: list[int],
) -> dict[int, torch.Tensor]:
    """Select source-frame depth maps from metadata depth by raw pixel frame."""
    if not torch.is_tensor(depth):
        raise TypeError(f"metadata depth must be a tensor, got {type(depth)!r}")
    d = depth
    if d.dim() == 3:
        d = d.unsqueeze(0)
    if d.dim() == 4:
        d = d.unsqueeze(2)
    if d.dim() != 5:
        raise ValueError(f"metadata depth must be [B,T,H,W] or [B,T,1,H,W], got {tuple(depth.shape)}")
    out: dict[int, torch.Tensor] = {}
    for frame_idx in sorted({int(i) for i in frame_indices}):
        idx = min(max(0, int(frame_idx)), d.shape[1] - 1)
        out[int(frame_idx)] = d[:, idx].contiguous()
    return out


def _maybe_int_scalar(value: Any) -> int | None:
    value = _first_scalar(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        if not value.dtype.is_floating_point and not value.dtype.is_complex:
            return int(value.flatten()[0].item())
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    return None


def _as_bool(value: Any) -> bool:
    value = _first_scalar(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(value.flatten()[0].item())
    return bool(value)


def _to_numpy(value: Any) -> np.ndarray:
    value = _first_scalar(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _is_action_adaln_param(name: str) -> bool:
    return "action_adaln_embedder" in name or "action_adaln_projection" in name
