from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from alaya.memory.spatial_cache import pixel_intrinsics


class DA3DepthEstimator:
    """Lazy Depth Anything 3 wrapper used by spatial memory.

    The DA3 repository can live outside this repo.  We import it lazily so the
    normal trainer path still works when the optional dependency is absent.
    """

    def __init__(
        self,
        *,
        repo_path: str | None,
        model_name: str,
        cache_dir: str | None,
        device: str | torch.device,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        align_to_input_scale: bool = True,
    ) -> None:
        self.repo_path = repo_path
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = torch.device(device)
        self.process_res = int(process_res)
        self.process_res_method = str(process_res_method)
        self.align_to_input_scale = bool(align_to_input_scale)
        self._model = None

    def infer_latent_depths(
        self,
        *,
        video_pixels: torch.Tensor,
        latent_indices: Iterable[int],
        cam_c2w: torch.Tensor | None,
        intrinsic: torch.Tensor | None,
        height: int,
        width: int,
        temporal_stride: int,
    ) -> dict[int, torch.Tensor]:
        indices = sorted({int(i) for i in latent_indices})
        if not indices:
            return {}
        frame_by_latent = {
            int(latent_idx): int(latent_idx) * int(temporal_stride)
            for latent_idx in indices
        }
        by_frame = self.infer_frame_depths(
            video_pixels=video_pixels,
            frame_indices=frame_by_latent.values(),
            cam_c2w=cam_c2w,
            intrinsic=intrinsic,
            height=height,
            width=width,
        )
        return {
            int(latent_idx): by_frame[int(frame_idx)]
            for latent_idx, frame_idx in frame_by_latent.items()
            if int(frame_idx) in by_frame
        }

    def infer_frame_depths(
        self,
        *,
        video_pixels: torch.Tensor,
        frame_indices: Iterable[int],
        cam_c2w: torch.Tensor | None,
        intrinsic: torch.Tensor | None,
        height: int,
        width: int,
    ) -> dict[int, torch.Tensor]:
        indices = sorted({int(i) for i in frame_indices})
        if not indices:
            return {}
        video_bfchw = _video_to_bfchw(video_pixels)
        if video_bfchw.shape[0] != 1:
            raise ValueError("DA3 spatial depth currently supports per-rank batch size 1")

        images: list[np.ndarray] = []
        frame_ids: list[int] = []
        for frame_idx_raw in indices:
            frame_idx = min(max(0, int(frame_idx_raw)), video_bfchw.shape[1] - 1)
            images.append(_frame_to_uint8_hwc(video_bfchw[0, frame_idx]))
            frame_ids.append(int(frame_idx_raw))
        if not images:
            return {}

        extrinsics_np = None
        intrinsics_np = None
        if cam_c2w is not None and intrinsic is not None and len(frame_ids) >= 2:
            extrinsics_np, intrinsics_np = _select_camera_arrays_for_frame_indices(
                cam_c2w=cam_c2w,
                intrinsic=intrinsic,
                frame_indices=frame_ids,
                height=height,
                width=width,
            )

        model = self._load_model()
        with torch.inference_mode():
            try:
                prediction = model.inference(
                    image=images,
                    extrinsics=extrinsics_np,
                    intrinsics=intrinsics_np,
                    align_to_input_ext_scale=self.align_to_input_scale and extrinsics_np is not None,
                    infer_gs=False,
                    process_res=self.process_res,
                    process_res_method=self.process_res_method,
                    export_dir=None,
                    export_format="mini_npz",
                )
            except Exception as exc:
                if extrinsics_np is None:
                    raise
                print(
                    "[DA3DepthEstimator] camera-conditioned inference failed; "
                    f"retrying without camera conditions: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                prediction = model.inference(
                    image=images,
                    extrinsics=None,
                    intrinsics=None,
                    align_to_input_ext_scale=False,
                    infer_gs=False,
                    process_res=self.process_res,
                    process_res_method=self.process_res_method,
                    export_dir=None,
                    export_format="mini_npz",
                )

        depths_np = np.asarray(prediction.depth, dtype=np.float32)
        if depths_np.ndim == 2:
            depths_np = depths_np[None]
        if depths_np.shape[0] != len(frame_ids):
            raise RuntimeError(f"DA3 returned {depths_np.shape[0]} depth maps for {len(frame_ids)} frames")

        out: dict[int, torch.Tensor] = {}
        for frame_idx, depth_np in zip(frame_ids, depths_np):
            depth = torch.from_numpy(depth_np).to(device=video_pixels.device, dtype=torch.float32)
            if depth.dim() == 2:
                depth = depth.unsqueeze(0).unsqueeze(0)
            elif depth.dim() == 3:
                depth = depth.unsqueeze(0)
            if depth.shape[-2:] != (height, width):
                depth = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
            depth = torch.nan_to_num(depth, nan=1e4, posinf=1e4, neginf=0.0).clamp(min=0.0, max=1e4)
            out[int(frame_idx)] = depth
        return out

    def _load_model(self):
        if self._model is not None:
            return self._model

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        if self.cache_dir:
            os.environ.setdefault("HF_HOME", str(Path(self.cache_dir).expanduser()))
        if self.repo_path:
            src = Path(self.repo_path).expanduser().resolve() / "src"
            if src.exists() and str(src) not in sys.path:
                sys.path.insert(0, str(src))
        _install_da3_export_dependency_stubs()
        try:
            from depth_anything_3.api import DepthAnything3  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Depth Anything 3 is not importable. Clone it under "
                "`spatial_memory.da3_repo_path` and install its required dependencies "
                "(at minimum the imports needed by depth_anything_3.api)."
            ) from exc

        model_name_or_path = _resolve_local_hf_model_snapshot(self.model_name, self.cache_dir)
        kwargs = {"local_files_only": True}
        if self.cache_dir:
            kwargs["cache_dir"] = str(Path(self.cache_dir).expanduser())
        self._model = DepthAnything3.from_pretrained(model_name_or_path, **kwargs).to(self.device)
        self._model.eval()
        return self._model


def _resolve_local_hf_model_snapshot(model_name: str, cache_dir: str | None) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)
    if not cache_dir or "/" not in model_name:
        return model_name

    cache_root = Path(cache_dir).expanduser()
    hub_root = cache_root / "hub"
    if not hub_root.exists():
        hub_root = cache_root
    repo_cache = hub_root / f"models--{model_name.replace('/', '--')}"
    refs_main = repo_cache / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = repo_cache / "snapshots" / revision
        if (snapshot / "config.json").exists() and (snapshot / "model.safetensors").exists():
            return str(snapshot)

    snapshots_root = repo_cache / "snapshots"
    if snapshots_root.exists():
        for snapshot in sorted(snapshots_root.iterdir(), reverse=True):
            if (snapshot / "config.json").exists() and (snapshot / "model.safetensors").exists():
                return str(snapshot)
    return model_name


def _install_da3_export_dependency_stubs() -> None:
    """Let DA3 inference import when optional DA3 extras are absent."""
    if "plyfile" not in sys.modules:
        try:
            import plyfile  # noqa: F401
        except Exception:
            plyfile_stub = types.ModuleType("plyfile")

            class _UnavailablePly:
                def __init__(self, *args, **kwargs):
                    raise ImportError("plyfile is required only for DA3 PLY/3DGS export")

                @classmethod
                def describe(cls, *args, **kwargs):
                    raise ImportError("plyfile is required only for DA3 PLY/3DGS export")

            plyfile_stub.PlyData = _UnavailablePly
            plyfile_stub.PlyElement = _UnavailablePly
            plyfile_stub.__file__ = f"<optional dependency stub: {plyfile_stub.__name__}>"
            sys.modules["plyfile"] = plyfile_stub

    for module_name, feature in (
        ("pycolmap", "COLMAP"),
        ("trimesh", "GLB"),
    ):
        if module_name in sys.modules:
            continue
        try:
            __import__(module_name)
        except Exception:
            stub = types.ModuleType(module_name)
            stub.__file__ = f"<optional dependency stub: {module_name}>"

            def _missing_optional_export_dep(*args, _module_name=module_name, _feature=feature, **kwargs):
                raise ImportError(f"{_module_name} is required only for DA3 {_feature} export")

            stub.__getattr__ = lambda name, _missing=_missing_optional_export_dep: _missing
            sys.modules[module_name] = stub

    if "evo.core.trajectory" not in sys.modules:
        try:
            from evo.core.trajectory import PosePath3D  # noqa: F401
        except Exception:
            evo_stub = types.ModuleType("evo")
            evo_stub.__file__ = "<optional dependency stub: evo>"
            evo_stub.__path__ = []
            core_stub = types.ModuleType("evo.core")
            core_stub.__file__ = "<optional dependency stub: evo.core>"
            core_stub.__path__ = []
            trajectory_stub = types.ModuleType("evo.core.trajectory")
            trajectory_stub.__file__ = "<optional dependency stub: evo.core.trajectory>"

            class PosePath3D:
                def __init__(self, poses_se3):
                    self.poses_se3 = [np.asarray(p, dtype=np.float64).copy() for p in poses_se3]

                def align(self, other, correct_scale=True):
                    source = np.stack(self.poses_se3)
                    target = np.stack(other.poses_se3)
                    rot, trans, scale = _umeyama_sim3(source[:, :3, 3], target[:, :3, 3], correct_scale)
                    aligned = source.copy()
                    aligned[:, :3, :3] = rot @ source[:, :3, :3]
                    aligned[:, :3, 3] = (rot @ (scale * source[:, :3, 3]).T).T + trans
                    self.poses_se3 = [pose for pose in aligned]
                    return rot, trans, scale

            trajectory_stub.PosePath3D = PosePath3D
            core_stub.trajectory = trajectory_stub
            evo_stub.core = core_stub
            sys.modules["evo"] = evo_stub
            sys.modules["evo.core"] = core_stub
            sys.modules["evo.core.trajectory"] = trajectory_stub


def _umeyama_sim3(source_xyz: np.ndarray, target_xyz: np.ndarray, correct_scale: bool) -> tuple[np.ndarray, np.ndarray, float]:
    source = np.asarray(source_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"expected matching Nx3 trajectories, got {source.shape} and {target.shape}")
    if source.shape[0] < 3:
        raise ValueError("at least three poses are required for Sim(3) alignment")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_var = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if source_var < 1e-12:
        raise ValueError("degenerate source trajectory for Sim(3) alignment")

    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    if singular_values[1] < 1e-12:
        raise ValueError("degenerate covariance for Sim(3) alignment")

    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        sign[-1] = -1.0
    rot = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular_values * sign) / source_var) if correct_scale else 1.0
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        raise ValueError("invalid Sim(3) scale")
    trans = target_mean - scale * (rot @ source_mean)
    return rot.astype(np.float64), trans.astype(np.float64), scale


def _video_to_bfchw(video_pixels: torch.Tensor) -> torch.Tensor:
    video = video_pixels.detach()
    if video.dim() == 5:
        # [B,F,C,H,W] from the dataloader.
        if video.shape[2] == 3:
            return video
        # [B,C,F,H,W]
        if video.shape[1] == 3:
            return video.permute(0, 2, 1, 3, 4).contiguous()
    if video.dim() == 4:
        # [F,C,H,W]
        if video.shape[1] == 3:
            return video.unsqueeze(0)
        # [C,F,H,W]
        if video.shape[0] == 3:
            return video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    raise ValueError(f"expected video tensor [B,F,C,H,W], [B,C,F,H,W], [F,C,H,W], or [C,F,H,W], got {tuple(video.shape)}")


def _frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    frame = frame_chw.detach().float().cpu()
    if frame.min().item() < -0.05:
        frame = frame * 0.5 + 0.5
    elif frame.max().item() > 2.0:
        frame = frame / 255.0
    frame = frame.clamp(0.0, 1.0)
    hwc = frame.permute(1, 2, 0).numpy()
    return np.clip(hwc * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _select_camera_arrays(
    *,
    cam_c2w: torch.Tensor,
    intrinsic: torch.Tensor,
    latent_indices: list[int],
    height: int,
    width: int,
    temporal_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame_indices = [int(i) * int(temporal_stride) for i in latent_indices]
    return _select_camera_arrays_for_frame_indices(
        cam_c2w=cam_c2w,
        intrinsic=intrinsic,
        frame_indices=frame_indices,
        height=height,
        width=width,
    )


def _select_camera_arrays_for_frame_indices(
    *,
    cam_c2w: torch.Tensor,
    intrinsic: torch.Tensor,
    frame_indices: list[int],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cam_c2w.dim() == 3:
        cam_c2w = cam_c2w.unsqueeze(0)
    if intrinsic.dim() == 2:
        intrinsic = intrinsic.unsqueeze(0)
    if intrinsic.dim() == 3:
        intrinsic_px = pixel_intrinsics(intrinsic, height=height, width=width)
    elif intrinsic.dim() == 4:
        B, T, _, _ = intrinsic.shape
        intrinsic_px = pixel_intrinsics(
            intrinsic.reshape(B * T, 3, 3),
            height=height,
            width=width,
        ).reshape(B, T, 3, 3)
    else:
        raise ValueError(f"unexpected intrinsic shape {tuple(intrinsic.shape)}")

    c2w = cam_c2w.detach().cpu().to(torch.float32)
    k_all = intrinsic_px.detach().cpu().to(torch.float32)
    exts: list[np.ndarray] = []
    ixts: list[np.ndarray] = []
    for frame_idx_raw in frame_indices:
        frame_idx = min(max(0, int(frame_idx_raw)), c2w.shape[1] - 1)
        w2c = torch.linalg.inv(c2w[0, frame_idx])
        exts.append(w2c.numpy().astype(np.float32))
        if k_all.dim() == 4:
            k_idx = min(frame_idx, k_all.shape[1] - 1)
            k = k_all[0, k_idx]
        else:
            k = k_all[0]
        ixts.append(k.numpy().astype(np.float32))
    return np.stack(exts, axis=0), np.stack(ixts, axis=0)
