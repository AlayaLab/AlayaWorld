from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class ViGeoPrefixGeometry:
    source_depth: torch.Tensor
    source_intrinsic: torch.Tensor
    scale: float
    pairwise_scales: tuple[float, ...]
    predicted_poses: torch.Tensor


@dataclass(frozen=True)
class ViGeoStreamGeometry:
    pointmaps: torch.Tensor
    valid_masks: torch.Tensor
    predicted_poses: torch.Tensor
    intrinsics: torch.Tensor
    kv_caches: Any


class ViGeoGeometryEstimator:
    """Lazy ViGeo wrapper for scale-from-prefix, last-frame geometry."""

    def __init__(
        self,
        *,
        repo_path: str | None,
        checkpoint: str,
        device: str | torch.device,
        num_tokens: int = 1369,
    ) -> None:
        self.repo_path = repo_path
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.num_tokens = int(num_tokens)
        self._model = None

    @torch.no_grad()
    def infer_prefix_geometry(
        self,
        *,
        video_pixels: torch.Tensor,
        frame_indices: Iterable[int],
        cam_c2w: torch.Tensor,
    ) -> ViGeoPrefixGeometry:
        indices = [int(index) for index in frame_indices]
        if len(indices) < 2:
            raise ValueError("ViGeo scale estimation requires at least two prefix frames")

        video = _video_to_bfchw(video_pixels)
        if int(video.shape[0]) != 1:
            raise ValueError("ViGeo spatial geometry currently supports per-rank batch size 1")
        max_video_index = int(video.shape[1]) - 1
        if min(indices) < 0 or max(indices) > max_video_index:
            raise IndexError(
                f"ViGeo frame indices {indices[0]}..{indices[-1]} exceed video length {video.shape[1]}"
            )

        index_tensor = torch.tensor(indices, device=video.device, dtype=torch.long)
        images = video[0].index_select(0, index_tensor).detach().float()
        if float(images.min().item()) < -0.05:
            images = images * 0.5 + 0.5
        elif float(images.max().item()) > 2.0:
            images = images / 255.0
        images = images.clamp_(0.0, 1.0)

        prediction = self._load_model().infer(
            images,
            mode="offline",
            num_tokens=self.num_tokens,
            resize_output=True,
            reset_cache=True,
        )
        pointmaps = prediction["points_pred"].float()
        masks = prediction.get("mask_pred")
        if masks is None:
            valid = torch.isfinite(pointmaps).all(dim=-1) & (pointmaps[..., 2] > 0)
        else:
            valid = masks[:, 0].bool() & torch.isfinite(pointmaps).all(dim=-1) & (pointmaps[..., 2] > 0)
        predicted_poses = prediction["pose_pred"].float()
        if int(pointmaps.shape[0]) != len(indices) or int(predicted_poses.shape[0]) != len(indices):
            raise RuntimeError(
                "ViGeo returned an unexpected temporal length: "
                f"points={pointmaps.shape[0]} poses={predicted_poses.shape[0]} expected={len(indices)}"
            )

        gt_poses = _select_camera_frames(cam_c2w, indices)
        scale, pairwise_scales = _pairwise_translation_scale(predicted_poses, gt_poses)
        source_points = pointmaps[-1]
        source_valid = valid[-1]
        try:
            source_intrinsic = _fit_intrinsic_from_pointmap(source_points, source_valid)
        except RuntimeError:
            source_intrinsic = _default_intrinsic(source_points)
        source_depth = torch.where(
            source_valid,
            source_points[..., 2] * float(scale),
            torch.zeros_like(source_points[..., 2]),
        )

        output_device = video_pixels.device
        return ViGeoPrefixGeometry(
            source_depth=source_depth.unsqueeze(0).unsqueeze(0).to(
                device=output_device, dtype=torch.float32
            ),
            source_intrinsic=source_intrinsic.unsqueeze(0).to(
                device=output_device, dtype=torch.float32
            ),
            scale=float(scale),
            pairwise_scales=tuple(pairwise_scales),
            predicted_poses=predicted_poses.cpu(),
        )

    @torch.no_grad()
    def infer_stream_geometry(
        self,
        *,
        video_pixels: torch.Tensor,
        kv_caches: Any = None,
        shared_intrinsic: torch.Tensor | None = None,
        reset_cache: bool = False,
        chunk_size: int = 16,
        total_budget: int = 0,
    ) -> ViGeoStreamGeometry:
        """Incrementally reconstruct RGB frames with one fixed stream intrinsic."""
        video = _video_to_bfchw(video_pixels)
        if int(video.shape[0]) != 1:
            raise ValueError("ViGeo streaming geometry currently supports per-rank batch size 1")
        images = _normalize_images(video[0])
        prediction = self._load_model().infer(
            images,
            mode="chunk",
            chunk_size=max(1, int(chunk_size)),
            num_tokens=self.num_tokens,
            total_budget=max(0, int(total_budget)),
            resize_output=True,
            kv_caches=kv_caches,
            reset_cache=bool(reset_cache),
        )
        pointmaps = prediction["points_pred"].float().cpu()
        valid = _prediction_valid_mask(pointmaps, prediction.get("mask_pred"))
        predicted_poses = prediction["pose_pred"].float().cpu()
        if int(pointmaps.shape[0]) != int(video.shape[1]):
            raise RuntimeError(
                "ViGeo streaming output length mismatch: "
                f"points={pointmaps.shape[0]} input={video.shape[1]}"
            )

        if shared_intrinsic is None:
            try:
                intrinsic = _fit_intrinsic_from_pointmap(pointmaps[-1], valid[-1])
            except RuntimeError:
                intrinsic = _default_intrinsic(pointmaps[-1])
        else:
            intrinsic = shared_intrinsic.detach().float().cpu()
            if intrinsic.shape == (1, 3, 3):
                intrinsic = intrinsic[0]
            if intrinsic.shape != (3, 3):
                raise ValueError(
                    "shared ViGeo intrinsic must have shape [3,3] or [1,3,3], "
                    f"got {tuple(intrinsic.shape)}"
                )
        intrinsics = (
            intrinsic.unsqueeze(0)
            .expand(int(pointmaps.shape[0]), -1, -1)
            .clone()
        )

        return ViGeoStreamGeometry(
            pointmaps=pointmaps,
            valid_masks=valid.cpu(),
            predicted_poses=predicted_poses,
            intrinsics=intrinsics,
            kv_caches=prediction.get("kv_caches"),
        )

    def estimate_translation_scale(
        self,
        *,
        predicted_poses: torch.Tensor,
        cam_c2w: torch.Tensor,
        frame_indices: Iterable[int],
        fallback_scale: float,
    ) -> tuple[float, tuple[float, ...]]:
        indices = [int(index) for index in frame_indices]
        if len(indices) < 2:
            return float(fallback_scale), ()
        gt_poses = _select_camera_frames(cam_c2w, indices)
        scale, ratios = _pairwise_translation_scale(predicted_poses, gt_poses)
        if not ratios:
            return float(fallback_scale), ()
        return float(scale), tuple(ratios)

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self.repo_path:
            repo = Path(self.repo_path).expanduser().resolve()
            if not repo.exists():
                raise FileNotFoundError(f"ViGeo repository does not exist: {repo}")
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
        try:
            from vigeo import ViGeo  # type: ignore
        except Exception as exc:
            raise ImportError(
                "ViGeo is not importable; set spatial_memory.vigeo_repo_path to its repository"
            ) from exc

        checkpoint = Path(self.checkpoint).expanduser()
        checkpoint_arg = str(checkpoint.resolve()) if checkpoint.exists() else self.checkpoint
        self._model = ViGeo.from_pretrained(checkpoint_arg).to(self.device).eval()
        return self._model


def _video_to_bfchw(video_pixels: torch.Tensor) -> torch.Tensor:
    video = video_pixels.detach()
    if video.dim() == 5:
        if int(video.shape[2]) == 3:
            return video
        if int(video.shape[1]) == 3:
            return video.permute(0, 2, 1, 3, 4).contiguous()
    if video.dim() == 4:
        if int(video.shape[1]) == 3:
            return video.unsqueeze(0)
        if int(video.shape[0]) == 3:
            return video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    raise ValueError(
        "expected video [B,F,C,H,W], [B,C,F,H,W], [F,C,H,W], or [C,F,H,W], "
        f"got {tuple(video.shape)}"
    )


def _normalize_images(images: torch.Tensor) -> torch.Tensor:
    images = images.detach().float()
    if float(images.min().item()) < -0.05:
        images = images * 0.5 + 0.5
    elif float(images.max().item()) > 2.0:
        images = images / 255.0
    return images.clamp_(0.0, 1.0)


def _prediction_valid_mask(
    pointmaps: torch.Tensor,
    masks: torch.Tensor | None,
) -> torch.Tensor:
    finite = torch.isfinite(pointmaps).all(dim=-1) & (pointmaps[..., 2] > 0)
    if masks is None:
        return finite
    return masks[:, 0].bool().cpu() & finite.cpu()


def _select_camera_frames(cam_c2w: torch.Tensor, indices: list[int]) -> torch.Tensor:
    cameras = cam_c2w.detach().float().cpu()
    if cameras.dim() == 4:
        if int(cameras.shape[0]) != 1:
            raise ValueError("ViGeo geometry expects camera batch size 1")
        cameras = cameras[0]
    if cameras.dim() != 3:
        raise ValueError(f"expected camera poses [F,4,4] or [1,F,4,4], got {tuple(cameras.shape)}")
    index_tensor = torch.tensor(indices, dtype=torch.long).clamp_(0, int(cameras.shape[0]) - 1)
    return cameras.index_select(0, index_tensor)


def _camera_centers(poses: torch.Tensor) -> np.ndarray:
    array = poses.detach().float().cpu().numpy()
    if array.shape[-2:] == (3, 4):
        return array[:, :3, 3].astype(np.float64)
    if array.shape[-2:] == (4, 4):
        return array[:, :3, 3].astype(np.float64)
    raise ValueError(f"unsupported pose shape {array.shape}")


def _pairwise_translation_scale(
    predicted_poses: torch.Tensor,
    gt_poses: torch.Tensor,
) -> tuple[float, list[float]]:
    predicted = _camera_centers(predicted_poses)
    target = _camera_centers(gt_poses)
    ratios: list[float] = []
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            predicted_distance = float(np.linalg.norm(predicted[right] - predicted[left]))
            target_distance = float(np.linalg.norm(target[right] - target[left]))
            if predicted_distance > 1e-8 and target_distance > 1e-8:
                ratio = target_distance / predicted_distance
                if np.isfinite(ratio) and ratio > 0.0:
                    ratios.append(float(ratio))
    if not ratios:
        return 1.0, []
    return float(np.median(np.asarray(ratios, dtype=np.float64))), ratios


def _fit_intrinsic_from_pointmap(
    points: torch.Tensor,
    valid: torch.Tensor,
    *,
    stride: int = 4,
) -> torch.Tensor:
    points_np = points.detach().float().cpu().numpy()
    valid_np = valid.detach().bool().cpu().numpy()
    height, width = points_np.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    selected = valid_np[::stride, ::stride].reshape(-1)
    sampled = points_np[::stride, ::stride].reshape(-1, 3)[selected]
    sampled_x = xx[::stride, ::stride].reshape(-1)[selected].astype(np.float64)
    sampled_y = yy[::stride, ::stride].reshape(-1)[selected].astype(np.float64)
    if len(sampled) < 16:
        raise RuntimeError(f"too few valid ViGeo points to recover intrinsics: {len(sampled)}")
    x_over_z = sampled[:, 0].astype(np.float64) / sampled[:, 2]
    y_over_z = sampled[:, 1].astype(np.float64) / sampled[:, 2]
    fx, cx = np.linalg.lstsq(
        np.stack([x_over_z, np.ones_like(x_over_z)], axis=1), sampled_x, rcond=None
    )[0]
    fy, cy = np.linalg.lstsq(
        np.stack([y_over_z, np.ones_like(y_over_z)], axis=1), sampled_y, rcond=None
    )[0]
    intrinsic = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    return torch.from_numpy(intrinsic)


def _default_intrinsic(points: torch.Tensor) -> torch.Tensor:
    height, width = points.shape[:2]
    focal = float(max(height, width))
    return torch.tensor(
        [
            [focal, 0.0, (float(width) - 1.0) * 0.5],
            [0.0, focal, (float(height) - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
