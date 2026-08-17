"""da3-only warp variants (verbatim from the flash inference release).

The da3 pipeline keeps exact copies of the functions whose implementations
diverged from alaya.memory.spatial_cache (the vigeo/oss versions), so vigeo
logic stays untouched. Only truly shared helpers are imported."""
from __future__ import annotations

import torch

from alaya.memory.spatial_cache import pixel_intrinsics, _prepare_intrinsics, _select_intrinsic, _video_to_bcfhw, _depth_for_frame_index

def _safe_inv(mat: torch.Tensor) -> torch.Tensor:
    """小矩阵(相机 4x4)求逆,在 CPU 上算后搬回原 device。

    GPU 的 torch.linalg.inv 走 cuSOLVER, 在显存吃紧时 cusolverDnCreate(handle) 会抛
    CUSOLVER_STATUS_INTERNAL_ERROR(DMD 同时驻留两份 13B + DA3 时常见)。相机矩阵极小,
    放 CPU 求逆代价可忽略, 且彻底绕开 cuSOLVER。float32 求逆更稳, 再转回原 dtype。
    """
    return torch.linalg.inv(mat.float().cpu()).to(device=mat.device, dtype=mat.dtype)

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
    c2w = _safe_inv(w2c.to(device=device, dtype=dtype))
    world = torch.matmul(c2w[:, None, None], cam.unsqueeze(-1))[..., :3, 0]
    return world

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

    # Unproject each source ONCE (target-independent); reuse across all targets.
    # Previously _forward_warp_rgbd_with_depth re-unprojected every source for
    # every target frame (sources x targets redundant unprojections + CPU-side
    # _safe_inv round-trips). Now: sources unprojections + sources*targets cheap
    # projections. Bit-identical output (same points reused).
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
        w2c_src = _safe_inv(cam_c2w[:, src_cam_idx])
        points, src_valid = _unproject_source(depth=depth, source_w2c=w2c_src, source_K=K_src)
        source_payloads.append((rgb, points, src_valid))

    if fill_value is None:
        fill = float(video.amin().item())
    else:
        fill = float(fill_value)

    # Stack sources once; each target fuses all sources in a single batched pass.
    rgb_stack = torch.stack([p[0] for p in source_payloads], dim=0)        # [S,B,C,H,W]
    points_stack = torch.stack([p[1] for p in source_payloads], dim=0)     # [S,B,H,W,3]
    src_valid_stack = torch.stack([p[2] for p in source_payloads], dim=0)  # [S,B,H,W]

    warped = []
    coverages = []
    for tgt_idx_raw in target_pixel_indices:
        tgt_idx = max(0, min(int(tgt_idx_raw), max_camera_frames - 1))
        w2c_tgt = _safe_inv(cam_c2w[:, tgt_idx])
        K_tgt = _select_intrinsic(intrinsic, tgt_idx)
        fused, covered = _warp_sources_to_target_batched(
            points=points_stack,
            src_valid=src_valid_stack,
            rgb=rgb_stack,
            target_w2c=w2c_tgt,
            target_K=K_tgt,
            depth_threshold=depth_threshold,
            fill=fill,
        )
        warped.append(fused)
        if return_coverage:
            coverages.append(covered.view(B, 1, height, width).to(dtype=torch.float32))

    out_dtype = source_pixels.dtype if source_pixels.dtype.is_floating_point else torch.float32
    warped_tensor = torch.stack(warped, dim=2).to(dtype=out_dtype)
    if return_coverage:
        return warped_tensor, torch.stack(coverages, dim=2)
    return warped_tensor

def _warp_sources_to_target_batched(
    *,
    points: torch.Tensor,      # [S,B,H,W,3] pre-unprojected source world points
    src_valid: torch.Tensor,   # [S,B,H,W]   source depth>0 mask
    rgb: torch.Tensor,         # [S,B,C,H,W] source colors
    target_w2c: torch.Tensor,  # [B,4,4]
    target_K: torch.Tensor,    # [B,3,3]
    depth_threshold: float,
    fill: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse ALL sources into one target view in a single batched pass.

    Bit-identical to looping `_warp_points_to_target` over sources and z-buffering
    by strict `<` (earliest source wins ties): per-source occlusion is keyed by
    s*N+pixel so each source keeps its own threshold/raster-order owner, then the
    cross-source winner is the earliest source at the minimal depth.
    Returns (fused_rgb [B,C,H,W], covered [B,H,W] bool)."""
    S, B, H, W, _ = points.shape
    C = rgb.shape[2]
    N = B * H * W
    device = rgb.device
    homo = torch.cat([points, torch.ones(S, B, H, W, 1, device=device, dtype=points.dtype)], dim=-1).unsqueeze(-1)
    cam = torch.matmul(target_w2c[None, :, None, None], homo)[..., :3, 0]
    z = cam[..., 2]
    proj = torch.matmul(target_K[None, :, None, None], cam.unsqueeze(-1))[..., 0]
    x = torch.round(proj[..., 0] / torch.clamp(proj[..., 2], min=1e-6)).long()
    y = torch.round(proj[..., 1] / torch.clamp(proj[..., 2], min=1e-6)).long()
    valid = src_valid & (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)

    fused = torch.full((N, C), float(fill), device=device, dtype=torch.float32)
    covered = torch.zeros((N,), device=device, dtype=torch.bool)
    if not valid.any():
        return fused.view(B, H, W, C).permute(0, 3, 1, 2).contiguous(), covered.view(B, H, W)

    s_ids, b_ids, y_src, x_src = valid.nonzero(as_tuple=True)
    tgt_pix = b_ids * (H * W) + y[valid] * W + x[valid]
    skey = s_ids * N + tgt_pix                     # per-source target key
    z_valid = z[valid].to(torch.float32)
    SN = S * N

    min_depth = torch.full((SN,), float("inf"), device=device, dtype=torch.float32)
    min_depth.scatter_reduce_(0, skey, z_valid, reduce="amin", include_self=True)
    keep = z_valid <= (min_depth[skey] + float(depth_threshold))
    if not keep.any():
        return fused.view(B, H, W, C).permute(0, 3, 1, 2).contiguous(), covered.view(B, H, W)

    kept_ord = keep.nonzero(as_tuple=False).flatten()
    owner = torch.full((SN,), torch.iinfo(torch.long).max, device=device, dtype=torch.long)
    owner.scatter_reduce_(0, skey[kept_ord], kept_ord.to(torch.long), reduce="amin", include_self=True)
    assigned_sk = owner != torch.iinfo(torch.long).max

    cand_depth = torch.full((SN,), float("inf"), device=device, dtype=torch.float32)
    cand_depth[assigned_sk] = z_valid[owner[assigned_sk]]
    rgb_gather = rgb.permute(0, 1, 3, 4, 2)[s_ids, b_ids, y_src, x_src]    # [M, C]
    cand_rgb = torch.full((SN, C), float(fill), device=device, dtype=torch.float32)
    cand_rgb[assigned_sk] = rgb_gather[owner[assigned_sk]]

    # cross-source: earliest source at the minimal candidate depth (torch.min
    # returns the first minimal index -> matches sequential strict-`<` fusion).
    best_depth, best_s = cand_depth.view(S, N).min(dim=0)
    covered = torch.isfinite(best_depth)
    sk_best = best_s * N + torch.arange(N, device=device)
    fused[covered] = cand_rgb[sk_best[covered]]
    return fused.view(B, H, W, C).permute(0, 3, 1, 2).contiguous(), covered.view(B, H, W)

def _unproject_source(
    *,
    depth: torch.Tensor,
    source_w2c: torch.Tensor,
    source_K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source-only half of the RGBD warp: unproject the source depth to world
    points once. Output is independent of the target camera, so it can be reused
    across all target frames (the previous per-(source,target) recomputation was
    the bulk of the warp cost). Returns (points [B,H,W,3], src_valid [B,H,W])."""
    points = unproject_depth(depth, w2c=source_w2c, intrinsic=source_K)
    src_valid = depth[:, 0] > 0
    return points, src_valid

def _warp_points_to_target(
    *,
    points: torch.Tensor,
    src_valid: torch.Tensor,
    rgb: torch.Tensor,
    target_w2c: torch.Tensor,
    target_K: torch.Tensor,
    depth_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Target-dependent half of the RGBD warp: project pre-unprojected source
    world points into one target view and z-buffer. Bit-identical to
    `_forward_warp_rgbd_with_depth` from the projection step onward."""
    B, C, H, W = rgb.shape
    homo = torch.cat([points, torch.ones(B, H, W, 1, device=rgb.device, dtype=points.dtype)], dim=-1).unsqueeze(-1)
    cam = torch.matmul(target_w2c[:, None, None], homo)[..., :3, 0]
    z = cam[..., 2]
    proj = torch.matmul(target_K[:, None, None], cam.unsqueeze(-1))[..., 0]
    x = torch.round(proj[..., 0] / torch.clamp(proj[..., 2], min=1e-6)).long()
    y = torch.round(proj[..., 1] / torch.clamp(proj[..., 2], min=1e-6)).long()
    valid = src_valid & (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)

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
