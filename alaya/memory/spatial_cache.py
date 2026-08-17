from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class SpatialContext:
    latent: torch.Tensor
    source_latent_indices: list[int]
    target_latent_indices: list[int]


def _invert_rigid_transform(transform: torch.Tensor) -> torch.Tensor:
    """Invert homogeneous camera poses without invoking a CUDA solver."""
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"camera transform must end in [4,4], got {tuple(transform.shape)}")
    rotation_inv = transform[..., :3, :3].transpose(-1, -2)
    translation = transform[..., :3, 3]
    inverse = torch.zeros_like(transform)
    inverse[..., :3, :3] = rotation_inv
    inverse[..., :3, 3] = -torch.matmul(rotation_inv, translation.unsqueeze(-1)).squeeze(-1)
    inverse[..., 3, 3] = 1
    return inverse


def pixel_intrinsics(intrinsic: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Return pixel-space intrinsics from either normalized or pixel K."""
    K = intrinsic.clone().to(dtype=torch.float32)
    if K.dim() == 2:
        K = K.unsqueeze(0)
    if float(K[..., 0, 2].abs().max()) <= 1.5 and float(K[..., 1, 2].abs().max()) <= 1.5:
        K[..., 0, 0] *= float(width)
        K[..., 1, 1] *= float(height)
        K[..., 0, 2] *= float(width)
        K[..., 1, 2] *= float(height)
    return K


def unproject_depth(
    depth: torch.Tensor,
    *,
    w2c: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    """Unproject a depth map to world points.

    Args:
        depth: [B, 1, H, W]
        w2c: [B, 4, 4]
        intrinsic: [B, 3, 3]

    Returns:
        [B, H, W, 3] world-space points.
    """
    if depth.dim() != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must be [B,1,H,W], got {tuple(depth.shape)}")
    B, _, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    z = depth[:, 0]
    fx = intrinsic[:, 0, 0].view(B, 1, 1).to(device=device, dtype=dtype)
    fy = intrinsic[:, 1, 1].view(B, 1, 1).to(device=device, dtype=dtype)
    cx = intrinsic[:, 0, 2].view(B, 1, 1).to(device=device, dtype=dtype)
    cy = intrinsic[:, 1, 2].view(B, 1, 1).to(device=device, dtype=dtype)
    x = (xs.view(1, H, W) - cx) / torch.clamp(fx, min=1e-6) * z
    y = (ys.view(1, H, W) - cy) / torch.clamp(fy, min=1e-6) * z
    cam = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)
    c2w = torch.linalg.inv(w2c.to(device=device, dtype=dtype))
    world = torch.matmul(c2w[:, None, None], cam.unsqueeze(-1))[..., :3, 0]
    return world


class Sparse3DCache:
    """Small Lyra-style geometry cache for frame retrieval.

    This stores per-frame world points and ranks candidate frames by how much
    visible area they project into target camera views.
    """

    def __init__(self, *, downsample: int = 4) -> None:
        self.downsample = int(downsample)
        self._world_points: list[torch.Tensor] = []
        self._latent_indices: list[int] = []
        self._frame_ids: list[int] = []

    @staticmethod
    def _scale_intrinsics(intrinsic: torch.Tensor, scale: float) -> torch.Tensor:
        K = intrinsic.clone()
        K[:, 0, :] *= scale
        K[:, 1, :] *= scale
        return K

    # ---- da3 inference additions (used only by alaya.inference; vigeo paths never call these) ----
    @staticmethod
    def compute_points(
        *,
        depth: torch.Tensor,
        w2c: torch.Tensor,
        intrinsic: torch.Tensor,
        downsample: int,
    ) -> torch.Tensor:
        """Downsample + unproject a frame's depth to world points (da3 bank precompute).

        Same math as `add`, split out so the da3 pipeline can cache per-frame points
        once and reuse them across chunks via `add_precomputed`."""
        ds = max(1, int(downsample))
        depth_ds = depth[:, :, ::ds, ::ds].to(dtype=torch.float32)
        K_ds = Sparse3DCache._scale_intrinsics(intrinsic.to(dtype=torch.float32), 1.0 / float(ds))
        return unproject_depth(depth_ds, w2c=w2c.to(dtype=torch.float32), intrinsic=K_ds)

    def add_precomputed(
        self,
        *,
        points: torch.Tensor,
        latent_index: int,
        frame_id: int | None = None,
    ) -> None:
        """Append world points produced by `compute_points` (skips recompute)."""
        self._world_points.append(points.detach())
        self._latent_indices.append(int(latent_index))
        self._frame_ids.append(int(latent_index) if frame_id is None else int(frame_id))

    def add(
        self,
        *,
        depth: torch.Tensor,
        w2c: torch.Tensor,
        intrinsic: torch.Tensor,
        latent_index: int,
        frame_id: int | None = None,
    ) -> None:
        ds = max(1, self.downsample)
        depth_ds = depth[:, :, ::ds, ::ds].to(dtype=torch.float32)
        K_ds = self._scale_intrinsics(intrinsic.to(dtype=torch.float32), 1.0 / float(ds))
        points = unproject_depth(depth_ds, w2c=w2c.to(dtype=torch.float32), intrinsic=K_ds)
        self._world_points.append(points.detach())
        self._latent_indices.append(int(latent_index))
        self._frame_ids.append(int(latent_index) if frame_id is None else int(frame_id))

    @torch.no_grad()
    def retrieve(
        self,
        *,
        target_w2c: torch.Tensor,
        target_intrinsic: torch.Tensor,
        target_hw: tuple[int, int],
        num_latents: int,
        skip_last_n: int = 0,
        max_coverage: bool = False,
        depth_threshold: float = 0.1,
    ) -> list[tuple[int, int]]:
        if not self._world_points or num_latents <= 0:
            return []
        device = target_w2c.device
        ds = max(1, self.downsample)
        Ht, Wt = target_hw
        Ht_ds = int((Ht + ds - 1) // ds)
        Wt_ds = int((Wt + ds - 1) // ds)

        if target_w2c.dim() == 4:
            views = int(target_w2c.shape[1])
            w2c_views = [target_w2c[:, i].to(device=device, dtype=torch.float32) for i in range(views)]
            K_views = [target_intrinsic[:, i].to(device=device, dtype=torch.float32) for i in range(views)]
        else:
            views = 1
            w2c_views = [target_w2c.to(device=device, dtype=torch.float32)]
            K_views = [target_intrinsic.to(device=device, dtype=torch.float32)]

        avail = max(0, len(self._world_points) - max(0, int(skip_last_n)))
        if avail == 0:
            return []

        pts = torch.stack([p.to(device=device, dtype=torch.float32) for p in self._world_points[:avail]], dim=0)
        C, B, H, W, _ = pts.shape
        homo = torch.cat([pts, torch.ones(C, B, H, W, 1, device=device)], dim=-1).unsqueeze(-1)
        w2c_stack = torch.stack(w2c_views, dim=0)
        K_ds_stack = torch.stack([self._scale_intrinsics(K, 1.0 / float(ds)) for K in K_views], dim=0)

        cam = torch.matmul(w2c_stack[:, None, :, None, None], homo[None])[..., :3, :]
        proj = torch.matmul(K_ds_stack[:, None, :, None, None], cam)[..., 0]
        z = cam[..., 2, 0]
        x = torch.round(proj[..., 0] / torch.clamp(proj[..., 2], min=1e-6)).long()
        y = torch.round(proj[..., 1] / torch.clamp(proj[..., 2], min=1e-6)).long()
        valid = (z > 0) & (x >= 0) & (x < Wt_ds) & (y >= 0) & (y < Ht_ds)
        if not valid.any():
            return []

        view_ids, cand_ids, b_ids, _ys, _xs = valid.nonzero(as_tuple=True)
        x_valid = x[valid]
        y_valid = y[valid]
        z_valid = z[valid].to(torch.float32)

        pixels_per_view = B * Ht_ds * Wt_ds
        n_keys = views * pixels_per_view
        keys = view_ids * pixels_per_view + b_ids * (Ht_ds * Wt_ds) + y_valid * Wt_ds + x_valid

        min_depth = torch.full((n_keys,), float("inf"), device=device, dtype=torch.float32)
        min_depth.scatter_reduce_(0, keys, z_valid, reduce="amin", include_self=True)
        visible = z_valid <= (min_depth[keys] + float(depth_threshold))
        if not visible.any():
            return []

        keys_vis = keys[visible]
        cand_vis = cand_ids[visible].to(torch.long)

        flat_idx = cand_vis * n_keys + keys_vis
        mask_flat = torch.zeros((avail * n_keys,), device=device, dtype=torch.bool)
        mask_flat.scatter_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.bool))
        mask = mask_flat.view(avail, n_keys)

        k = min(int(num_latents), avail)
        if k <= 0:
            return []

        if max_coverage:
            covered = torch.zeros((n_keys,), device=device, dtype=torch.bool)
            selected: list[int] = []
            for _ in range(k):
                additional = (mask & (~covered)).sum(dim=1)
                if selected:
                    additional[torch.tensor(selected, device=device)] = -1
                best = int(torch.argmax(additional).item())
                if int(additional[best].item()) <= 0:
                    break
                selected.append(best)
                covered |= mask[best]
            if not selected:
                return []
            top = selected
        else:
            scores = mask.sum(dim=1).to(torch.float32)
            if float(scores.max().item()) <= 0:
                return []
            top = torch.topk(scores, k=k).indices.tolist()

        top = top[::-1]
        return [(self._latent_indices[i], self._frame_ids[i]) for i in top]


def build_retrieved_latent_context(
    *,
    latent_full: torch.Tensor,
    cam_c2w: torch.Tensor,
    intrinsic: torch.Tensor,
    allowed_latent_indices: Iterable[int],
    target_latent_indices: Iterable[int],
    height: int,
    width: int,
    temporal_stride: int,
    num_context_frames: int,
    downsample: int,
    constant_depth: float,
    depth_by_latent_index: dict[int, torch.Tensor] | None = None,
    retrieval_max_coverage: bool = True,
    retrieval_depth_threshold: float = 0.1,
) -> SpatialContext | None:
    allowed = sorted({int(i) for i in allowed_latent_indices})
    targets = sorted({int(i) for i in target_latent_indices})
    if not allowed or not targets:
        return None

    if cam_c2w.dim() == 3:
        cam_c2w = cam_c2w.unsqueeze(0)
    if intrinsic.dim() == 2:
        intrinsic = intrinsic.unsqueeze(0)
    cam_c2w = cam_c2w.to(device=latent_full.device, dtype=torch.float32)
    intrinsic = pixel_intrinsics(
        intrinsic.to(device=latent_full.device, dtype=torch.float32),
        height=height,
        width=width,
    )
    if intrinsic.shape[0] == 1 and cam_c2w.shape[0] > 1:
        intrinsic = intrinsic.expand(cam_c2w.shape[0], -1, -1)
    B = int(latent_full.shape[0])
    if intrinsic.shape[0] == 1 and B > 1:
        intrinsic = intrinsic.expand(B, -1, -1)

    def _pixel_index(latent_idx: int) -> int:
        return max(0, min(int(latent_idx) * int(temporal_stride), int(cam_c2w.shape[1]) - 1))

    cache = Sparse3DCache(downsample=downsample)
    depth_shape = (B, 1, height, width)
    for latent_idx in allowed:
        pix_idx = _pixel_index(latent_idx)
        depth = None if depth_by_latent_index is None else depth_by_latent_index.get(int(latent_idx))
        if depth is None:
            depth = torch.full(depth_shape, float(constant_depth), device=latent_full.device, dtype=torch.float32)
        else:
            depth = depth.to(device=latent_full.device, dtype=torch.float32)
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            if depth.shape[-2:] != (height, width):
                depth = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
        w2c = torch.linalg.inv(cam_c2w[:, pix_idx])
        cache.add(
            depth=depth,
            w2c=w2c,
            intrinsic=intrinsic,
            latent_index=latent_idx,
            frame_id=latent_idx,
        )

    target_w2c = torch.stack([torch.linalg.inv(cam_c2w[:, _pixel_index(i)]) for i in targets], dim=1)
    target_K = intrinsic.unsqueeze(1).expand(-1, target_w2c.shape[1], -1, -1)
    retrieved = cache.retrieve(
        target_w2c=target_w2c,
        target_intrinsic=target_K,
        target_hw=(height, width),
        num_latents=num_context_frames,
        max_coverage=bool(retrieval_max_coverage),
        depth_threshold=float(retrieval_depth_threshold),
    )
    if not retrieved:
        return None
    selected = [int(latent_idx) for latent_idx, _frame_id in retrieved]
    latents = [latent_full[:, :, idx : idx + 1] for idx in selected if 0 <= idx < latent_full.shape[2]]
    if not latents:
        return None
    # The retrieved latent is a proxy for a target-view spatial context. Once
    # RGBD/latent warping is wired in, this tensor should contain the source
    # frame projected into these target slots instead of the raw source latent.
    target_slots = [targets[min(i, len(targets) - 1)] for i in range(len(latents))]
    return SpatialContext(
        latent=torch.cat(latents, dim=2).contiguous(),
        source_latent_indices=selected[: len(latents)],
        target_latent_indices=target_slots,
    )


@torch.no_grad()
def forward_warp_video_to_targets(
    *,
    video_pixels: torch.Tensor,
    source_latent_indices: list[int],
    target_latent_indices: list[int],
    cam_c2w: torch.Tensor,
    intrinsic: torch.Tensor,
    depth_by_latent_index: dict[int, torch.Tensor] | None,
    height: int,
    width: int,
    temporal_stride: int,
    constant_depth: float,
    depth_threshold: float = 1e-4,
) -> torch.Tensor | None:
    if not source_latent_indices or not target_latent_indices:
        return None
    video = _video_to_bcfhw(video_pixels).to(device=cam_c2w.device)
    if cam_c2w.dim() == 3:
        cam_c2w = cam_c2w.unsqueeze(0)
    if intrinsic.dim() == 2:
        intrinsic = intrinsic.unsqueeze(0)
    cam_c2w = cam_c2w.to(device=video.device, dtype=torch.float32)
    intrinsic = _prepare_intrinsics(intrinsic.to(device=video.device, dtype=torch.float32), height=height, width=width)

    B = int(video.shape[0])
    if B != int(cam_c2w.shape[0]):
        if cam_c2w.shape[0] == 1:
            cam_c2w = cam_c2w.expand(B, -1, -1, -1)
        else:
            raise ValueError(f"video batch {B} does not match camera batch {cam_c2w.shape[0]}")

    warped = []
    for src_idx, tgt_idx in zip(source_latent_indices, target_latent_indices):
        src_pix = _pixel_index(src_idx, temporal_stride, cam_c2w.shape[1])
        tgt_pix = _pixel_index(tgt_idx, temporal_stride, cam_c2w.shape[1])
        rgb = video[:, :, min(src_pix, video.shape[2] - 1)].to(dtype=torch.float32)
        depth = None if depth_by_latent_index is None else depth_by_latent_index.get(int(src_idx))
        if depth is None:
            depth = torch.full((B, 1, height, width), float(constant_depth), device=video.device, dtype=torch.float32)
        else:
            depth = depth.to(device=video.device, dtype=torch.float32)
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            if depth.shape[-2:] != (height, width):
                depth = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
        K_src = _select_intrinsic(intrinsic, src_pix)
        K_tgt = _select_intrinsic(intrinsic, tgt_pix)
        w2c_src = torch.linalg.inv(cam_c2w[:, src_pix])
        w2c_tgt = torch.linalg.inv(cam_c2w[:, tgt_pix])
        warped.append(
            _forward_warp_rgbd(
                rgb=rgb,
                depth=depth,
                source_w2c=w2c_src,
                target_w2c=w2c_tgt,
                source_K=K_src,
                target_K=K_tgt,
                depth_threshold=depth_threshold,
            )
        )
    out_dtype = video_pixels.dtype if video_pixels.dtype.is_floating_point else torch.float32
    return torch.stack(warped, dim=2).to(dtype=out_dtype)


@torch.no_grad()
def forward_warp_all_sources_to_targets(
    *,
    video_pixels: torch.Tensor,
    source_latent_indices: list[int],
    target_latent_indices: list[int],
    cam_c2w: torch.Tensor,
    intrinsic: torch.Tensor,
    depth_by_latent_index: dict[int, torch.Tensor] | None,
    height: int,
    width: int,
    temporal_stride: int,
    constant_depth: float,
    depth_threshold: float = 1e-4,
) -> torch.Tensor | None:
    if not source_latent_indices or not target_latent_indices:
        return None
    video = _video_to_bcfhw(video_pixels).to(device=cam_c2w.device)
    if cam_c2w.dim() == 3:
        cam_c2w = cam_c2w.unsqueeze(0)
    if intrinsic.dim() == 2:
        intrinsic = intrinsic.unsqueeze(0)
    cam_c2w = cam_c2w.to(device=video.device, dtype=torch.float32)
    intrinsic = _prepare_intrinsics(intrinsic.to(device=video.device, dtype=torch.float32), height=height, width=width)

    B = int(video.shape[0])
    if B != int(cam_c2w.shape[0]):
        if cam_c2w.shape[0] == 1:
            cam_c2w = cam_c2w.expand(B, -1, -1, -1)
        else:
            raise ValueError(f"video batch {B} does not match camera batch {cam_c2w.shape[0]}")

    source_payloads = []
    for src_idx in source_latent_indices:
        src_pix = _pixel_index(src_idx, temporal_stride, cam_c2w.shape[1])
        rgb = video[:, :, min(src_pix, video.shape[2] - 1)].to(dtype=torch.float32)
        depth = _depth_for_latent_index(
            src_idx=int(src_idx),
            depth_by_latent_index=depth_by_latent_index,
            batch=B,
            height=height,
            width=width,
            device=video.device,
            constant_depth=float(constant_depth),
        )
        K_src = _select_intrinsic(intrinsic, src_pix)
        w2c_src = torch.linalg.inv(cam_c2w[:, src_pix])
        source_payloads.append((rgb, depth, w2c_src, K_src))

    warped = []
    for tgt_idx in target_latent_indices:
        tgt_pix = _pixel_index(tgt_idx, temporal_stride, cam_c2w.shape[1])
        w2c_tgt = torch.linalg.inv(cam_c2w[:, tgt_pix])
        K_tgt = _select_intrinsic(intrinsic, tgt_pix)
        fused = torch.zeros((B, video.shape[1], height, width), device=video.device, dtype=torch.float32)
        fused_depth = torch.full((B * height * width,), float("inf"), device=video.device, dtype=torch.float32)
        fused_flat = fused.permute(0, 2, 3, 1).reshape(B * height * width, video.shape[1])

        for rgb, depth, w2c_src, K_src in source_payloads:
            candidate, candidate_depth = _forward_warp_rgbd_with_depth(
                rgb=rgb,
                depth=depth,
                source_w2c=w2c_src,
                target_w2c=w2c_tgt,
                source_K=K_src,
                target_K=K_tgt,
                depth_threshold=depth_threshold,
            )
            candidate_depth_flat = candidate_depth.reshape(-1)
            update = candidate_depth_flat < fused_depth
            if update.any():
                candidate_flat = candidate.permute(0, 2, 3, 1).reshape(B * height * width, video.shape[1])
                fused_flat[update] = candidate_flat[update]
                fused_depth[update] = candidate_depth_flat[update]

        warped.append(fused)

    out_dtype = video_pixels.dtype if video_pixels.dtype.is_floating_point else torch.float32
    return torch.stack(warped, dim=2).to(dtype=out_dtype)


@torch.no_grad()
def forward_warp_pixel_sources_to_pixel_targets(
    *,
    video_pixels: torch.Tensor,
    source_pixel_indices: list[int],
    target_pixel_indices: list[int],
    cam_c2w: torch.Tensor,
    intrinsic: torch.Tensor,
    depth_by_frame_index: dict[int, torch.Tensor] | None,
    height: int,
    width: int,
    constant_depth: float,
    depth_threshold: float = 1e-4,
    fill_value: float | None = None,
    return_coverage: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse arbitrary source pixel frames into arbitrary target pixel frames."""
    if not source_pixel_indices or not target_pixel_indices:
        return None
    video = _video_to_bcfhw(video_pixels).to(device=cam_c2w.device)
    if cam_c2w.dim() == 3:
        cam_c2w = cam_c2w.unsqueeze(0)
    if intrinsic.dim() == 2:
        intrinsic = intrinsic.unsqueeze(0)
    cam_c2w = cam_c2w.to(device=video.device, dtype=torch.float32)
    intrinsic = _prepare_intrinsics(intrinsic.to(device=video.device, dtype=torch.float32), height=height, width=width)

    B = int(video.shape[0])
    if B != int(cam_c2w.shape[0]):
        if cam_c2w.shape[0] == 1:
            cam_c2w = cam_c2w.expand(B, -1, -1, -1)
        else:
            raise ValueError(f"video batch {B} does not match camera batch {cam_c2w.shape[0]}")

    max_frames = min(int(video.shape[2]), int(cam_c2w.shape[1]))
    if max_frames <= 0:
        return None

    source_payloads = []
    for src_idx_raw in source_pixel_indices:
        src_idx = max(0, min(int(src_idx_raw), max_frames - 1))
        rgb = video[:, :, src_idx].to(dtype=torch.float32)
        depth = _depth_for_frame_index(
            frame_idx=src_idx,
            depth_by_frame_index=depth_by_frame_index,
            batch=B,
            height=height,
            width=width,
            device=video.device,
            constant_depth=float(constant_depth),
        )
        K_src = _select_intrinsic(intrinsic, src_idx)
        w2c_src = torch.linalg.inv(cam_c2w[:, src_idx])
        source_payloads.append((rgb, depth, w2c_src, K_src))

    if fill_value is None:
        fill = float(video.amin().item())
    else:
        fill = float(fill_value)

    warped = []
    coverages = []
    for tgt_idx_raw in target_pixel_indices:
        tgt_idx = max(0, min(int(tgt_idx_raw), max_frames - 1))
        w2c_tgt = torch.linalg.inv(cam_c2w[:, tgt_idx])
        K_tgt = _select_intrinsic(intrinsic, tgt_idx)
        fused = torch.full((B, video.shape[1], height, width), fill, device=video.device, dtype=torch.float32)
        fused_depth = torch.full((B * height * width,), float("inf"), device=video.device, dtype=torch.float32)
        fused_flat = fused.permute(0, 2, 3, 1).reshape(B * height * width, video.shape[1])

        for rgb, depth, w2c_src, K_src in source_payloads:
            candidate, candidate_depth = _forward_warp_rgbd_with_depth(
                rgb=rgb,
                depth=depth,
                source_w2c=w2c_src,
                target_w2c=w2c_tgt,
                source_K=K_src,
                target_K=K_tgt,
                depth_threshold=depth_threshold,
            )
            candidate_depth_flat = candidate_depth.reshape(-1)
            update = torch.isfinite(candidate_depth_flat) & (candidate_depth_flat < fused_depth)
            if update.any():
                candidate_flat = candidate.permute(0, 2, 3, 1).reshape(B * height * width, video.shape[1])
                fused_flat[update] = candidate_flat[update]
                fused_depth[update] = candidate_depth_flat[update]

        warped.append(fused)
        if return_coverage:
            coverages.append(torch.isfinite(fused_depth).view(B, 1, height, width).to(dtype=torch.float32))

    out_dtype = video_pixels.dtype if video_pixels.dtype.is_floating_point else torch.float32
    warped_tensor = torch.stack(warped, dim=2).to(dtype=out_dtype)
    if return_coverage:
        return warped_tensor, torch.stack(coverages, dim=2)
    return warped_tensor


@torch.no_grad()
def render_colored_pointmaps_to_camera_targets(
    *,
    source_pixels: torch.Tensor,
    source_pointmaps: torch.Tensor,
    source_valid_masks: torch.Tensor,
    source_c2w: torch.Tensor,
    target_c2w: torch.Tensor,
    target_intrinsic: torch.Tensor,
    height: int,
    width: int,
    depth_threshold: float = 1e-4,
    fill_value: float | None = None,
    return_coverage: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse colored camera-local pointmaps and render them into target views."""
    video = _video_to_bcfhw(source_pixels)
    device = video.device
    points = source_pointmaps.to(device=device, dtype=torch.float32)
    valid = source_valid_masks.to(device=device, dtype=torch.bool)
    if points.dim() == 4:
        points = points.unsqueeze(0)
    if valid.dim() == 3:
        valid = valid.unsqueeze(0)
    if source_c2w.dim() == 3:
        source_c2w = source_c2w.unsqueeze(0)
    if target_c2w.dim() == 3:
        target_c2w = target_c2w.unsqueeze(0)
    if target_intrinsic.dim() == 3:
        target_intrinsic = target_intrinsic.unsqueeze(0)

    B, C, source_count, source_height, source_width = video.shape
    expected_points = (B, source_count, source_height, source_width, 3)
    expected_valid = (B, source_count, source_height, source_width)
    if tuple(points.shape) != expected_points:
        raise ValueError(f"source pointmaps must be {expected_points}, got {tuple(points.shape)}")
    if tuple(valid.shape) != expected_valid:
        raise ValueError(f"source valid masks must be {expected_valid}, got {tuple(valid.shape)}")
    if (source_height, source_width) != (int(height), int(width)):
        raise ValueError(
            "source pointmap resolution must match render resolution: "
            f"{(source_height, source_width)} != {(height, width)}"
        )
    if tuple(source_c2w.shape[:2]) != (B, source_count):
        raise ValueError(
            f"source_c2w must start with {(B, source_count)}, got {tuple(source_c2w.shape)}"
        )
    if int(target_c2w.shape[0]) != B:
        raise ValueError(f"target camera batch {target_c2w.shape[0]} does not match source batch {B}")
    target_count = int(target_c2w.shape[1])
    if tuple(target_intrinsic.shape[:2]) != (B, target_count):
        raise ValueError(
            f"target_intrinsic must start with {(B, target_count)}, got {tuple(target_intrinsic.shape)}"
        )

    source_c2w = source_c2w.to(device=device, dtype=torch.float32)
    target_c2w = target_c2w.to(device=device, dtype=torch.float32)
    target_intrinsic = _prepare_intrinsics(
        target_intrinsic.to(device=device, dtype=torch.float32),
        height=height,
        width=width,
    )

    ones = torch.ones((*points.shape[:-1], 1), device=device, dtype=torch.float32)
    local_h = torch.cat([points, ones], dim=-1).unsqueeze(-1)
    world = torch.matmul(source_c2w[:, :, None, None], local_h)[..., :3, 0]
    world = world.reshape(B, -1, 3)
    colors = video.permute(0, 2, 3, 4, 1).reshape(B, -1, C).to(dtype=torch.float32)
    valid = (
        valid
        & torch.isfinite(points).all(dim=-1)
        & (points[..., 2] > 0)
    ).reshape(B, -1)

    fill = float(video.amin().item()) if fill_value is None else float(fill_value)
    warped = []
    coverages = []
    for target_index in range(target_count):
        candidate, candidate_depth = _rasterize_world_points_with_depth(
            world_points=world,
            colors=colors,
            valid=valid,
            target_w2c=_invert_rigid_transform(target_c2w[:, target_index]),
            target_K=target_intrinsic[:, target_index],
            height=height,
            width=width,
            fill_value=fill,
            depth_threshold=depth_threshold,
        )
        warped.append(candidate)
        if return_coverage:
            coverages.append(torch.isfinite(candidate_depth).unsqueeze(1).to(torch.float32))

    out_dtype = source_pixels.dtype if source_pixels.dtype.is_floating_point else torch.float32
    warped_tensor = torch.stack(warped, dim=2).to(dtype=out_dtype)
    if return_coverage:
        return warped_tensor, torch.stack(coverages, dim=2)
    return warped_tensor



@torch.no_grad()
def forward_warp_indexed_pixel_sources_to_pixel_targets(
    *,
    source_pixels: torch.Tensor,
    source_pixel_indices: list[int],
    source_camera_pixel_indices: list[int],
    target_pixel_indices: list[int],
    cam_c2w: torch.Tensor,
    intrinsic: torch.Tensor,
    depth_by_source_index: dict[int, torch.Tensor] | None,
    height: int,
    width: int,
    constant_depth: float,
    depth_threshold: float = 1e-4,
    fill_value: float | None = None,
    return_coverage: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse bank frames into target pixel frames.

    source_pixels is indexed by source_pixel_indices, while cameras are indexed
    in the original global pixel timeline via source_camera_pixel_indices.
    """
    if not source_pixel_indices or not target_pixel_indices:
        return None
    video = _video_to_bcfhw(source_pixels).to(device=cam_c2w.device)
    if cam_c2w.dim() == 3:
        cam_c2w = cam_c2w.unsqueeze(0)
    if intrinsic.dim() == 2:
        intrinsic = intrinsic.unsqueeze(0)
    cam_c2w = cam_c2w.to(device=video.device, dtype=torch.float32)
    intrinsic = _prepare_intrinsics(intrinsic.to(device=video.device, dtype=torch.float32), height=height, width=width)

    B = int(video.shape[0])
    if B != int(cam_c2w.shape[0]):
        if cam_c2w.shape[0] == 1:
            cam_c2w = cam_c2w.expand(B, -1, -1, -1)
        else:
            raise ValueError(f"video batch {B} does not match camera batch {cam_c2w.shape[0]}")

    max_source_frames = int(video.shape[2])
    max_camera_frames = int(cam_c2w.shape[1])
    if max_source_frames <= 0 or max_camera_frames <= 0:
        return None
    if len(source_camera_pixel_indices) < max_source_frames:
        raise ValueError(
            f"source_camera_pixel_indices has {len(source_camera_pixel_indices)} entries, "
            f"but source_pixels has {max_source_frames} frames"
        )

    source_payloads = []
    for src_idx_raw in source_pixel_indices:
        src_idx = max(0, min(int(src_idx_raw), max_source_frames - 1))
        src_cam_idx = max(0, min(int(source_camera_pixel_indices[src_idx]), max_camera_frames - 1))
        rgb = video[:, :, src_idx].to(dtype=torch.float32)
        depth = _depth_for_frame_index(
            frame_idx=src_idx,
            depth_by_frame_index=depth_by_source_index,
            batch=B,
            height=height,
            width=width,
            device=video.device,
            constant_depth=float(constant_depth),
        )
        K_src = _select_intrinsic(intrinsic, src_cam_idx)
        w2c_src = torch.linalg.inv(cam_c2w[:, src_cam_idx])
        source_payloads.append((rgb, depth, w2c_src, K_src))

    if fill_value is None:
        fill = float(video.amin().item())
    else:
        fill = float(fill_value)

    warped = []
    coverages = []
    for tgt_idx_raw in target_pixel_indices:
        tgt_idx = max(0, min(int(tgt_idx_raw), max_camera_frames - 1))
        w2c_tgt = torch.linalg.inv(cam_c2w[:, tgt_idx])
        K_tgt = _select_intrinsic(intrinsic, tgt_idx)
        fused = torch.full((B, video.shape[1], height, width), fill, device=video.device, dtype=torch.float32)
        fused_depth = torch.full((B * height * width,), float("inf"), device=video.device, dtype=torch.float32)
        fused_flat = fused.permute(0, 2, 3, 1).reshape(B * height * width, video.shape[1])

        for rgb, depth, w2c_src, K_src in source_payloads:
            candidate, candidate_depth = _forward_warp_rgbd_with_depth(
                rgb=rgb,
                depth=depth,
                source_w2c=w2c_src,
                target_w2c=w2c_tgt,
                source_K=K_src,
                target_K=K_tgt,
                depth_threshold=depth_threshold,
            )
            candidate_depth_flat = candidate_depth.reshape(-1)
            update = torch.isfinite(candidate_depth_flat) & (candidate_depth_flat < fused_depth)
            if update.any():
                candidate_flat = candidate.permute(0, 2, 3, 1).reshape(B * height * width, video.shape[1])
                fused_flat[update] = candidate_flat[update]
                fused_depth[update] = candidate_depth_flat[update]

        warped.append(fused)
        if return_coverage:
            coverages.append(torch.isfinite(fused_depth).view(B, 1, height, width).to(dtype=torch.float32))

    out_dtype = source_pixels.dtype if source_pixels.dtype.is_floating_point else torch.float32
    warped_tensor = torch.stack(warped, dim=2).to(dtype=out_dtype)
    if return_coverage:
        return warped_tensor, torch.stack(coverages, dim=2)
    return warped_tensor


def _rasterize_world_points_with_depth(
    *,
    world_points: torch.Tensor,
    colors: torch.Tensor,
    valid: torch.Tensor,
    target_w2c: torch.Tensor,
    target_K: torch.Tensor,
    height: int,
    width: int,
    fill_value: float,
    depth_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, point_count, _ = world_points.shape
    channels = int(colors.shape[-1])
    ones = torch.ones((B, point_count, 1), device=world_points.device, dtype=world_points.dtype)
    world_h = torch.cat([world_points, ones], dim=-1).unsqueeze(-1)
    camera = torch.matmul(target_w2c[:, None], world_h)[..., :3, 0]
    z = camera[..., 2]
    projected = torch.matmul(target_K[:, None], camera.unsqueeze(-1))[..., 0]
    x = torch.round(projected[..., 0] / torch.clamp(projected[..., 2], min=1e-6)).long()
    y = torch.round(projected[..., 1] / torch.clamp(projected[..., 2], min=1e-6)).long()
    visible = (
        valid
        & torch.isfinite(camera).all(dim=-1)
        & (z > 0)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )

    out = torch.full(
        (B, channels, height, width),
        float(fill_value),
        device=world_points.device,
        dtype=torch.float32,
    )
    out_depth = torch.full(
        (B * height * width,),
        float("inf"),
        device=world_points.device,
        dtype=torch.float32,
    )
    if not visible.any():
        return out, out_depth.view(B, height, width)

    batch_ids, point_ids = visible.nonzero(as_tuple=True)
    keys = batch_ids * (height * width) + y[visible] * width + x[visible]
    z_valid = z[visible].to(torch.float32)
    min_depth = torch.full_like(out_depth, float("inf"))
    min_depth.scatter_reduce_(0, keys, z_valid, reduce="amin", include_self=True)
    keep = z_valid <= min_depth[keys] + float(depth_threshold)
    if not keep.any():
        return out, out_depth.view(B, height, width)

    kept_ord = keep.nonzero(as_tuple=False).flatten()
    kept_keys = keys[kept_ord]
    owner = torch.full(
        (B * height * width,),
        torch.iinfo(torch.long).max,
        device=world_points.device,
        dtype=torch.long,
    )
    owner.scatter_reduce_(0, kept_keys, kept_ord.to(torch.long), reduce="amin", include_self=True)
    assigned = owner != torch.iinfo(torch.long).max
    if not assigned.any():
        return out, out_depth.view(B, height, width)

    color_ids = batch_ids * point_count + point_ids
    color_flat = colors.reshape(B * point_count, channels)
    out_flat = out.permute(0, 2, 3, 1).reshape(B * height * width, channels)
    winner_ord = owner[assigned]
    out_flat[assigned] = color_flat[color_ids[winner_ord]]
    out_depth[assigned] = z_valid[winner_ord]
    return out_flat.view(B, height, width, channels).permute(0, 3, 1, 2).contiguous(), out_depth.view(B, height, width)



def _forward_warp_rgbd(
    *,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    source_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    source_K: torch.Tensor,
    target_K: torch.Tensor,
    depth_threshold: float,
) -> torch.Tensor:
    B, C, H, W = rgb.shape
    points = unproject_depth(depth, w2c=source_w2c, intrinsic=source_K)
    homo = torch.cat([points, torch.ones(B, H, W, 1, device=rgb.device, dtype=points.dtype)], dim=-1).unsqueeze(-1)
    cam = torch.matmul(target_w2c[:, None, None], homo)[..., :3, 0]
    z = cam[..., 2]
    proj = torch.matmul(target_K[:, None, None], cam.unsqueeze(-1))[..., 0]
    x = torch.round(proj[..., 0] / torch.clamp(proj[..., 2], min=1e-6)).long()
    y = torch.round(proj[..., 1] / torch.clamp(proj[..., 2], min=1e-6)).long()
    valid = (depth[:, 0] > 0) & (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)

    out = torch.zeros_like(rgb)
    if not valid.any():
        return out

    b_ids, y_src, x_src = valid.nonzero(as_tuple=True)
    keys = b_ids * (H * W) + y[b_ids, y_src, x_src] * W + x[b_ids, y_src, x_src]
    z_valid = z[b_ids, y_src, x_src].to(torch.float32)
    n_keys = B * H * W
    min_depth = torch.full((n_keys,), float("inf"), device=rgb.device, dtype=torch.float32)
    min_depth.scatter_reduce_(0, keys, z_valid, reduce="amin", include_self=True)
    keep = z_valid <= (min_depth[keys] + float(depth_threshold))
    if not keep.any():
        return out

    kept_ord = keep.nonzero(as_tuple=False).flatten()
    kept_keys = keys[kept_ord]
    owner = torch.full((n_keys,), torch.iinfo(torch.long).max, device=rgb.device, dtype=torch.long)
    owner.scatter_reduce_(0, kept_keys, kept_ord.to(torch.long), reduce="amin", include_self=True)
    assigned = owner != torch.iinfo(torch.long).max
    if not assigned.any():
        return out

    src_flat = (b_ids * (H * W) + y_src * W + x_src).to(torch.long)
    rgb_flat = rgb.permute(0, 2, 3, 1).reshape(B * H * W, C)
    out_flat = out.permute(0, 2, 3, 1).reshape(B * H * W, C)
    out_flat[assigned] = rgb_flat[src_flat[owner[assigned]]]
    return out_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


def _forward_warp_rgbd_with_depth(
    *,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    source_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    source_K: torch.Tensor,
    target_K: torch.Tensor,
    depth_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, C, H, W = rgb.shape
    points = unproject_depth(depth, w2c=source_w2c, intrinsic=source_K)
    homo = torch.cat([points, torch.ones(B, H, W, 1, device=rgb.device, dtype=points.dtype)], dim=-1).unsqueeze(-1)
    cam = torch.matmul(target_w2c[:, None, None], homo)[..., :3, 0]
    z = cam[..., 2]
    proj = torch.matmul(target_K[:, None, None], cam.unsqueeze(-1))[..., 0]
    x = torch.round(proj[..., 0] / torch.clamp(proj[..., 2], min=1e-6)).long()
    y = torch.round(proj[..., 1] / torch.clamp(proj[..., 2], min=1e-6)).long()
    valid = (depth[:, 0] > 0) & (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)

    out = torch.zeros_like(rgb)
    out_depth = torch.full((B * H * W,), float("inf"), device=rgb.device, dtype=torch.float32)
    if not valid.any():
        return out, out_depth.view(B, H, W)

    b_ids, y_src, x_src = valid.nonzero(as_tuple=True)
    keys = b_ids * (H * W) + y[b_ids, y_src, x_src] * W + x[b_ids, y_src, x_src]
    z_valid = z[b_ids, y_src, x_src].to(torch.float32)
    n_keys = B * H * W
    min_depth = torch.full((n_keys,), float("inf"), device=rgb.device, dtype=torch.float32)
    min_depth.scatter_reduce_(0, keys, z_valid, reduce="amin", include_self=True)
    keep = z_valid <= (min_depth[keys] + float(depth_threshold))
    if not keep.any():
        return out, out_depth.view(B, H, W)

    kept_ord = keep.nonzero(as_tuple=False).flatten()
    kept_keys = keys[kept_ord]
    owner = torch.full((n_keys,), torch.iinfo(torch.long).max, device=rgb.device, dtype=torch.long)
    owner.scatter_reduce_(0, kept_keys, kept_ord.to(torch.long), reduce="amin", include_self=True)
    assigned = owner != torch.iinfo(torch.long).max
    if not assigned.any():
        return out, out_depth.view(B, H, W)

    src_flat = (b_ids * (H * W) + y_src * W + x_src).to(torch.long)
    rgb_flat = rgb.permute(0, 2, 3, 1).reshape(B * H * W, C)
    out_flat = out.permute(0, 2, 3, 1).reshape(B * H * W, C)
    winner_ord = owner[assigned]
    out_flat[assigned] = rgb_flat[src_flat[winner_ord]]
    out_depth[assigned] = z_valid[winner_ord]
    return out_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous(), out_depth.view(B, H, W)


def _depth_for_latent_index(
    *,
    src_idx: int,
    depth_by_latent_index: dict[int, torch.Tensor] | None,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    constant_depth: float,
) -> torch.Tensor:
    depth = None if depth_by_latent_index is None else depth_by_latent_index.get(int(src_idx))
    if depth is None:
        return torch.full((batch, 1, height, width), float(constant_depth), device=device, dtype=torch.float32)
    depth = depth.to(device=device, dtype=torch.float32)
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    if depth.shape[-2:] != (height, width):
        depth = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
    return depth


def _depth_for_frame_index(
    *,
    frame_idx: int,
    depth_by_frame_index: dict[int, torch.Tensor] | None,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    constant_depth: float,
) -> torch.Tensor:
    depth = None if depth_by_frame_index is None else depth_by_frame_index.get(int(frame_idx))
    if depth is None:
        return torch.full((batch, 1, height, width), float(constant_depth), device=device, dtype=torch.float32)
    depth = depth.to(device=device, dtype=torch.float32)
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    if depth.shape[-2:] != (height, width):
        depth = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
    return depth


def _video_to_bcfhw(video_pixels: torch.Tensor) -> torch.Tensor:
    video = video_pixels.detach()
    if video.dim() == 5:
        if video.shape[2] == 3:
            return video.permute(0, 2, 1, 3, 4).contiguous()
        if video.shape[1] == 3:
            return video
    if video.dim() == 4:
        if video.shape[0] == 3:
            return video.unsqueeze(0)
        if video.shape[1] == 3:
            return video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    raise ValueError(f"unexpected video shape {tuple(video.shape)}")


def _prepare_intrinsics(intrinsic: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if intrinsic.dim() == 3:
        return pixel_intrinsics(intrinsic, height=height, width=width)
    if intrinsic.dim() == 4:
        B, T, _, _ = intrinsic.shape
        return pixel_intrinsics(intrinsic.reshape(B * T, 3, 3), height=height, width=width).reshape(B, T, 3, 3)
    raise ValueError(f"unexpected intrinsic shape {tuple(intrinsic.shape)}")


def _select_intrinsic(intrinsic: torch.Tensor, pixel_idx: int) -> torch.Tensor:
    if intrinsic.dim() == 4:
        return intrinsic[:, min(max(0, int(pixel_idx)), intrinsic.shape[1] - 1)]
    return intrinsic


def _pixel_index(latent_idx: int, temporal_stride: int, num_pixel_frames: int) -> int:
    return max(0, min(int(latent_idx) * int(temporal_stride), int(num_pixel_frames) - 1))
