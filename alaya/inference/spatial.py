"""Spatial memory (geometric warp) subsystem for FlashAlaya.

Standalone extraction of RolloutTrainer's validation-rollout spatial bank:
past pixel frames + DA3 depths are stored in a bank; for each new chunk the
bank sources are selected by coverage retrieval, forward-warped to the target
camera poses, VAE-encoded, and returned as the DiT's spatial condition.

Scope (deliberate): only the `target_prefix_pixels` bank path used at
inference. The offline retrieval/non-bank paths from the trainer are not
carried over.

Depends only on library modules (alaya.memory.*, ltx2.modules); no trainer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from ltx2.modules.patchifier import VideoLatentPatchifier

from alaya.memory.da3_depth import DA3DepthEstimator
from alaya.memory.spatial_cache import (
    Sparse3DCache,
    pixel_intrinsics,
)
from alaya.inference.warp_da3 import forward_warp_indexed_pixel_sources_to_pixel_targets


@dataclass
class SpatialBank:
    pixels: list[torch.Tensor]
    frame_indices: list[int]
    depths: list[torch.Tensor | None]
    # Lazily-populated cache of per-frame downsampled world points (keyed by the
    # frame's local index in this bank). A frame's points never change once it is
    # in the bank, so retrieval recomputes only newly-appended frames each chunk
    # instead of the whole O(N) candidate set. See SpatialMemory._select_bank_sources.
    world_points: dict[int, torch.Tensor] = field(default_factory=dict)


def _as_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.flatten()[0].item()) if value.numel() else False
    if isinstance(value, (list, tuple)):
        return _as_bool(value[0]) if value else False
    return bool(value)


class SpatialMemory:
    """Bank-based spatial conditioning (init / build_context / append)."""

    def __init__(
        self,
        *,
        spatial_cfg,            # cfg.spatial_memory
        sample_cfg,             # cfg.sample (height/width/fps/temporal_stride)
        sink_latent_frames: int,
        vae_encoder,            # StreamingVAEEncoder
        vae_decoder,            # VideoDecoder (bank pixels for DA3)
        vae_chunk_size: int,
        device: torch.device,
        dtype: torch.dtype,
        da3_repo: str | None = None,
        da3_model: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        da3_cache: str | None = None,
        da3: DA3DepthEstimator | None = None,
    ) -> None:
        self.cfg = spatial_cfg
        self.sample = sample_cfg
        self.sink_latent_frames = int(sink_latent_frames)
        # DA3 path-type settings sourced from cfg.paths.da3_* (see engine.setup).
        self.da3_repo = da3_repo
        self.da3_model = da3_model
        self.da3_cache = da3_cache
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.vae_chunk_size = int(vae_chunk_size)
        self.device = device
        self.dtype = dtype
        self.da3 = da3
        # Optional tiny streaming decoder (taehv) for the bank pixels. When set,
        # the bank decode (which feeds DA3 depth + the forward warp) uses taehv
        # instead of the full LTX VAE — much cheaper, but LOSSY (approximate
        # pixels). Enable via enable_taehv_bank_decode(); see _decode_latent_to_bank_pixels.
        self._taehv = None
        self._taehv_dtype = torch.float16

    def enable_taehv_bank_decode(self, taehv, *, dtype: torch.dtype = torch.float16) -> None:
        """Use a TAEHV tiny decoder for the bank pixels (LOSSY). taeltx2_3_wide
        matches the LTX VAE's x8 temporal / x32 spatial, so the decoded frame
        count per chunk is identical (no bank frame-index drift)."""
        self._taehv = taehv
        self._taehv_dtype = dtype

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "enabled", False)) and (
            str(getattr(self.cfg, "context_mode", "retrieval")) == "target_prefix_pixels"
        )

    def ensure_da3(self) -> None:
        """Instantiate + load the DA3 model (call eagerly at setup)."""
        if not self.enabled or str(self.cfg.depth_backend) != "da3":
            return
        if self.da3 is None:
            device = self.device if str(self.cfg.da3_device) == "auto" else torch.device(str(self.cfg.da3_device))
            self.da3 = DA3DepthEstimator(
                repo_path=self.da3_repo,
                model_name=str(self.da3_model),
                cache_dir=self.da3_cache,
                device=device,
                process_res=int(self.cfg.da3_process_res),
                process_res_method=str(self.cfg.da3_process_res_method),
                align_to_input_scale=bool(self.cfg.da3_align_to_input_scale),
            )
        self.da3._load_model()

    # ------------------------------------------------------------- init bank
    @torch.no_grad()
    def init_bank(
        self, *, video_pixels: torch.Tensor, metadata: dict[str, Any], target_start: int
    ) -> SpatialBank | None:
        cfg = self.cfg
        if not self.enabled:
            return None
        if not _as_bool(metadata.get("has_camera", False)):
            return None
        cam_c2w = metadata.get("cam_c2w")
        intrinsic = metadata.get("intrinsic")
        if cam_c2w is None or intrinsic is None:
            return None

        stride = int(self.sample.temporal_stride)
        target_pixel_start = int(target_start) * stride
        history_pixels = max(1, int(cfg.num_context_frames))
        source_floor = 0
        if not bool(cfg.include_sink):
            source_floor = max(0, self.sink_latent_frames * stride)
        source_start = max(source_floor, target_pixel_start - history_pixels)
        source_indices = list(range(source_start, target_pixel_start))

        video_frames = self._video_pixel_frame_count(video_pixels)
        cam_frames = int(cam_c2w.shape[1] if cam_c2w.dim() == 4 else cam_c2w.shape[0])
        max_frames = min(video_frames, cam_frames)
        source_indices = [idx for idx in source_indices if 0 <= int(idx) < max_frames]
        if not source_indices:
            return None
        if bool(getattr(cfg, "require_full_context", True)) and len(source_indices) < history_pixels:
            return None

        pixels = self._select_video_pixel_frames(video_pixels, source_indices)
        depth_by_local = self._infer_bank_depths(pixels=pixels, metadata=metadata, frame_indices=source_indices)
        return SpatialBank(
            pixels=[pixels[:, :, i].detach().contiguous() for i in range(int(pixels.shape[2]))],
            frame_indices=[int(i) for i in source_indices],
            depths=[depth_by_local.get(i) if depth_by_local is not None else None for i in range(int(pixels.shape[2]))],
        )

    # --------------------------------------------------------- build context
    @torch.no_grad()
    def build_context(
        self,
        *,
        bank: SpatialBank,
        metadata: dict[str, Any],
        target_start: int,
        K: int,
        target_rope_t_indices: torch.Tensor | None,
    ) -> dict[str, Any] | None:
        cfg = self.cfg
        if not bank.pixels:
            return None
        cam_c2w = metadata.get("cam_c2w")
        intrinsic = metadata.get("intrinsic")
        if cam_c2w is None or intrinsic is None:
            return None
        if cam_c2w.dim() == 3:
            cam_c2w = cam_c2w.unsqueeze(0)
        cam_c2w = cam_c2w.to(device=self.device, dtype=torch.float32)
        intrinsic = intrinsic.to(device=self.device, dtype=torch.float32)

        stride = int(self.sample.temporal_stride)
        target_pixel_start = int(target_start) * stride
        target_pixel_count = 1 + max(0, int(K) - 1) * stride
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

        selected = self._select_bank_sources(
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

        source_video = torch.stack(bank.pixels, dim=2).to(device=self.device, dtype=self.dtype).contiguous()
        depth_by_source = {
            int(local_idx): bank.depths[local_idx] for local_idx in selected if bank.depths[local_idx] is not None
        }
        warp_result = forward_warp_indexed_pixel_sources_to_pixel_targets(
            source_pixels=source_video,
            source_pixel_indices=[int(i) for i in selected],
            source_camera_pixel_indices=[int(i) for i in bank.frame_indices],
            target_pixel_indices=target_pixel_indices,
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            depth_by_source_index=depth_by_source,
            height=int(self.sample.height),
            width=int(self.sample.width),
            constant_depth=float(cfg.constant_depth),
            depth_threshold=min(float(cfg.retrieval_depth_threshold), 1e-3),
            fill_value=None,
            return_coverage=True,
        )
        if warp_result is None:
            return None
        warped_pixels, coverage_pixels = warp_result

        spatial_latent = self._encode_context_video(warped_pixels, expected_latent_frames=int(K))
        mask_patch = self._build_mask_patch(coverage_pixels=coverage_pixels, spatial_latent=spatial_latent)
        if target_rope_t_indices is None:
            rope_t_indices: list[float] = list(range(int(target_start), int(target_start) + int(K)))
        else:
            rope_t_indices = [float(x) for x in target_rope_t_indices.detach().cpu().tolist()]
        return {
            "latent": spatial_latent,
            "mask_patch": mask_patch,
            "source_indices": [int(bank.frame_indices[i]) for i in selected],
            "target_indices": list(range(int(target_start), int(target_start) + int(K))),
            "source_pixel_indices": [int(bank.frame_indices[i]) for i in selected],
            "target_pixel_indices": target_pixel_indices,
            "rope_t_indices": rope_t_indices,
        }

    # ------------------------------------------------------- append new chunk
    @torch.no_grad()
    def append_prediction(
        self, *, bank: SpatialBank, pred_latent: torch.Tensor, metadata: dict[str, Any], target_start: int
    ) -> None:
        cam_c2w = metadata.get("cam_c2w")
        if cam_c2w is None:
            return
        pixels = self._decode_latent_to_bank_pixels(pred_latent)
        stride = int(self.sample.temporal_stride)
        frame_count = int(pixels.shape[2])
        frame_indices = list(range(int(target_start) * stride, int(target_start) * stride + frame_count))
        cam_frames = int(cam_c2w.shape[1] if cam_c2w.dim() == 4 else cam_c2w.shape[0])
        keep = [i for i, frame_idx in enumerate(frame_indices) if 0 <= int(frame_idx) < cam_frames]
        if not keep:
            return
        pixels = pixels[:, :, keep].contiguous()
        frame_indices = [frame_indices[i] for i in keep]
        depth_by_local = self._infer_bank_depths(pixels=pixels, metadata=metadata, frame_indices=frame_indices)
        for local_idx, frame_idx in enumerate(frame_indices):
            bank.pixels.append(pixels[:, :, local_idx].detach().contiguous())
            bank.frame_indices.append(int(frame_idx))
            bank.depths.append(depth_by_local.get(local_idx) if depth_by_local is not None else None)

    # ------------------------------------------------------------ internals
    def _decode_latent_to_bank_pixels(self, latent: torch.Tensor) -> torch.Tensor:
        chunk = latent.to(device=self.device, dtype=self.dtype)
        if self._taehv is not None:
            # LOSSY tiny decoder. taehv wants NTCHW; taeltx upscales x8 in time so
            # T=K latents -> (K-1)*8+1 frames, same as the full VAE. Output is
            # [0,1]; the bank convention (like video_pixels) is [-1,1].
            x = chunk.permute(0, 2, 1, 3, 4).to(dtype=self._taehv_dtype)
            with torch.no_grad():
                frames = self._taehv.decode_video(x, parallel=True, show_progress_bar=False)
            pixel = frames.permute(0, 2, 1, 3, 4) * 2.0 - 1.0
            return pixel.detach().to(device=self.device, dtype=self.dtype).contiguous()
        with torch.no_grad():
            pixel = self.vae_decoder(chunk)
        return pixel.detach().to(device=self.device, dtype=self.dtype).contiguous()

    def _select_bank_sources(
        self,
        *,
        bank: SpatialBank,
        candidate_indices: list[int],
        target_pixel_indices: list[int],
        cam_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> list[int]:
        cfg = self.cfg
        num_context = max(1, int(cfg.num_context_frames))
        ds = max(1, int(cfg.downsample))
        selected: list[int] = []
        try:
            cache = Sparse3DCache(downsample=ds)
            for local_idx in candidate_indices:
                frame_idx = int(bank.frame_indices[local_idx])
                # Per-frame world points are fixed once the frame is in the bank
                # (depth + camera never change); compute once and cache, so each
                # chunk only unprojects newly-appended frames. Bit-identical to
                # the previous full recompute (same compute_points math).
                points = bank.world_points.get(local_idx)
                if points is None:
                    depth = bank.depths[local_idx]
                    if depth is None:
                        depth = torch.full(
                            (cam_c2w.shape[0], 1, int(self.sample.height), int(self.sample.width)),
                            float(cfg.constant_depth),
                            device=self.device,
                            dtype=torch.float32,
                        )
                    else:
                        depth = depth.to(device=self.device, dtype=torch.float32)
                    points = Sparse3DCache.compute_points(
                        depth=depth,
                        w2c=torch.linalg.inv(cam_c2w[:, frame_idx]),
                        intrinsic=self._intrinsic_for_pixel_frame(intrinsic, frame_idx, batch=int(cam_c2w.shape[0])),
                        downsample=ds,
                    )
                    bank.world_points[local_idx] = points
                cache.add_precomputed(points=points, latent_index=int(local_idx), frame_id=frame_idx)

            retrieval_views = max(1, int(cfg.retrieval_views))
            if retrieval_views == 1:
                target_view_indices = [target_pixel_indices[-1]]
            else:
                offsets = torch.linspace(0, len(target_pixel_indices) - 1, retrieval_views)
                target_view_indices = [target_pixel_indices[int(round(float(x.item())))] for x in offsets]
            target_w2c = torch.stack([torch.linalg.inv(cam_c2w[:, idx]) for idx in target_view_indices], dim=1)
            target_K = torch.stack(
                [
                    self._intrinsic_for_pixel_frame(intrinsic, idx, batch=int(cam_c2w.shape[0]))
                    for idx in target_view_indices
                ],
                dim=1,
            )
            retrieved = cache.retrieve(
                target_w2c=target_w2c,
                target_intrinsic=target_K,
                target_hw=(int(self.sample.height), int(self.sample.width)),
                num_latents=num_context,
                max_coverage=bool(cfg.retrieval_max_coverage),
                depth_threshold=float(cfg.retrieval_depth_threshold),
            )
            selected = [int(local_idx) for local_idx, _frame_id in retrieved]
        except Exception as exc:
            print(f"[FlashAlaya:Spatial] coverage source selection failed: {type(exc).__name__}: {exc}", flush=True)
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

    def _infer_bank_depths(
        self, *, pixels: torch.Tensor, metadata: dict[str, Any], frame_indices: list[int]
    ) -> dict[int, torch.Tensor]:
        cfg = self.cfg
        backend = str(cfg.depth_backend)
        if backend == "constant":
            return {}
        if backend != "da3":
            raise ValueError(f"FlashAlaya supports depth_backend in ('da3','constant'), got {backend!r}")
        cam_c2w = metadata.get("cam_c2w")
        intrinsic = metadata.get("intrinsic")
        if cam_c2w is None or intrinsic is None:
            return {}
        self.ensure_da3()
        local_frame_indices = list(range(int(pixels.shape[2])))
        cam_subset = self._select_camera_frames(cam_c2w, frame_indices)
        intrinsic_subset = self._select_intrinsic_frames(intrinsic, frame_indices)
        return self.da3.infer_frame_depths(
            video_pixels=pixels,
            frame_indices=local_frame_indices,
            cam_c2w=cam_subset,
            intrinsic=intrinsic_subset,
            height=int(self.sample.height),
            width=int(self.sample.width),
        ) or {}

    def _encode_context_video(self, warped_pixels: torch.Tensor, *, expected_latent_frames: int) -> torch.Tensor:
        warped_pixels = warped_pixels.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            latent = self.vae_encoder.encode(warped_pixels, chunk_size=self.vae_chunk_size, verbose=False)
        latent = latent.to(device=self.device, dtype=self.dtype).contiguous()
        if int(latent.shape[2]) != int(expected_latent_frames):
            raise RuntimeError(
                f"spatial VAE produced {latent.shape[2]} latents, expected {expected_latent_frames}; "
                f"warped_pixels={tuple(warped_pixels.shape)}"
            )
        return latent

    def _build_mask_patch(self, *, coverage_pixels: torch.Tensor, spatial_latent: torch.Tensor) -> torch.Tensor:
        B, C, frames, height, width = spatial_latent.shape
        if coverage_pixels.dim() != 5 or int(coverage_pixels.shape[0]) != B or int(coverage_pixels.shape[1]) != 1:
            raise ValueError(f"coverage_pixels must be [B,1,T,H,W] with B={B}, got {tuple(coverage_pixels.shape)}")
        patchifier = VideoLatentPatchifier(patch_size=1)
        pt, ph, pw = patchifier.patch_size
        if frames % pt != 0 or height % ph != 0 or width % pw != 0:
            raise ValueError(f"spatial latent shape {(frames, height, width)} not divisible by {(pt, ph, pw)}")
        # A token is valid when its spatio-temporal bin is mostly covered (> 0.5).
        mask_grid = F.adaptive_avg_pool3d(
            coverage_pixels.to(device=self.device, dtype=torch.float32),
            output_size=(frames // pt, height // ph, width // pw),
        ).gt_(0.5).to(dtype=torch.float32)
        mask_flat = mask_grid[:, 0].reshape(B, -1, 1)
        channel_patch = int(C) * int(pt) * int(ph) * int(pw)
        return mask_flat.expand(B, -1, channel_patch).to(dtype=spatial_latent.dtype).contiguous()

    # ------------------------------------------------------- tensor helpers
    def _video_pixel_frame_count(self, video_pixels: torch.Tensor) -> int:
        if video_pixels.dim() == 5:
            if video_pixels.shape[2] == 3:
                return int(video_pixels.shape[1])
            if video_pixels.shape[1] == 3:
                return int(video_pixels.shape[2])
            raise ValueError(f"expected [B,F,C,H,W] or [B,C,F,H,W], got {tuple(video_pixels.shape)}")
        if video_pixels.dim() != 4:
            raise ValueError(f"expected [F,C,H,W] or [C,F,H,W], got {tuple(video_pixels.shape)}")
        return int(video_pixels.shape[1] if video_pixels.shape[0] == 3 else video_pixels.shape[0])

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
        raise ValueError(f"unexpected video shape {tuple(video.shape)}")

    def _select_video_pixel_frames(self, video_pixels: torch.Tensor, frame_indices: list[int]) -> torch.Tensor:
        video = self._video_pixels_to_bcfhw(video_pixels).to(device=self.device, dtype=self.dtype)
        idx = torch.tensor([int(i) for i in frame_indices], device=video.device, dtype=torch.long)
        return video.index_select(2, idx).contiguous()

    def _select_camera_frames(self, cam_c2w: torch.Tensor, frame_indices: list[int]) -> torch.Tensor:
        cam = cam_c2w.to(device=self.device, dtype=torch.float32)
        idx = torch.tensor([int(i) for i in frame_indices], device=cam.device, dtype=torch.long)
        if cam.dim() == 4:
            idx = idx.clamp(0, cam.shape[1] - 1)
            return cam.index_select(1, idx).contiguous()
        if cam.dim() == 3:
            idx = idx.clamp(0, cam.shape[0] - 1)
            return cam.index_select(0, idx).unsqueeze(0).contiguous()
        raise ValueError(f"unexpected cam_c2w shape {tuple(cam_c2w.shape)}")

    def _select_intrinsic_frames(self, intrinsic: torch.Tensor, frame_indices: list[int]) -> torch.Tensor:
        K = intrinsic.to(device=self.device, dtype=torch.float32)
        if K.dim() == 4:
            idx = torch.tensor([int(i) for i in frame_indices], device=K.device, dtype=torch.long).clamp(
                0, K.shape[1] - 1
            )
            return K.index_select(1, idx).contiguous()
        return K

    def _intrinsic_for_pixel_frame(self, intrinsic: torch.Tensor, frame_idx: int, *, batch: int) -> torch.Tensor:
        K = intrinsic.to(device=self.device, dtype=torch.float32)
        if K.dim() == 4:
            idx = max(0, min(int(frame_idx), int(K.shape[1]) - 1))
            K = K[:, idx]
        elif K.dim() == 2:
            K = K.unsqueeze(0)
        K = pixel_intrinsics(K, height=int(self.sample.height), width=int(self.sample.width))
        if K.shape[0] == 1 and int(batch) > 1:
            K = K.expand(int(batch), -1, -1)
        return K.contiguous()
