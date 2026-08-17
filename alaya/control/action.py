from __future__ import annotations

import numpy as np
import torch


def build_action_vectors(
    cam_c2w: torch.Tensor,
    target_latent_indices: torch.Tensor,
    *,
    action_scale: str,
    temporal_stride: int = 8,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert normalized camera poses into latent-rate 6D AdaLN actions."""
    full = c2w_to_adaln_action_vectors(
        cam_c2w,
        action_scale=action_scale,
        vae_temporal_stride=temporal_stride,
    ).to(device=device, dtype=dtype)
    indices = target_latent_indices.to(device=device, dtype=torch.long)
    indices = indices.clamp(min=0, max=full.shape[1] - 1)
    return full.index_select(1, indices)


def build_action_vectors_from_pixel_indices(
    cam_c2w: torch.Tensor,
    pixel_indices: torch.Tensor | list[int],
    *,
    previous_pixel_index: int | None,
    action_scale: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build actions on an explicit, possibly non-stride-aligned pixel timeline."""
    from scipy.spatial.transform import Rotation as R

    cameras = cam_c2w.detach().cpu().numpy()
    if cameras.ndim == 3:
        cameras = cameras[None]
    if cameras.ndim != 4:
        raise ValueError(f"cam_c2w must be [F,4,4] or [B,F,4,4], got {cameras.shape}")
    if isinstance(pixel_indices, torch.Tensor):
        indices = [int(value) for value in pixel_indices.detach().cpu().tolist()]
    else:
        indices = [int(value) for value in pixel_indices]
    if not indices:
        return torch.empty(
            cameras.shape[0], 0, 6, device=device, dtype=dtype
        )

    scale = _parse_action_scale(action_scale)
    batch_out = []
    for camera_sequence in cameras:
        max_index = int(camera_sequence.shape[0]) - 1
        selected = [camera_sequence[max(0, min(index, max_index))] for index in indices]
        previous = (
            camera_sequence[max(0, min(int(previous_pixel_index), max_index))]
            if previous_pixel_index is not None
            else None
        )
        vectors = np.zeros((len(selected), 6), dtype=np.float32)
        for index, pose in enumerate(selected):
            reference = previous if index == 0 else selected[index - 1]
            if reference is None:
                continue
            relative = np.linalg.inv(reference) @ pose
            translation = relative[:3, 3].astype(np.float32)
            rotation = R.from_matrix(relative[:3, :3]).as_euler(
                "xyz", degrees=False
            ).astype(np.float32)
            vectors[index] = np.concatenate([translation, rotation], axis=0) / scale
        batch_out.append(torch.from_numpy(vectors))
    return torch.stack(batch_out, dim=0).to(device=device, dtype=dtype)


def c2w_to_adaln_action_vectors(
    c2w_normalized: torch.Tensor | np.ndarray,
    *,
    action_scale: str,
    vae_temporal_stride: int = 8,
) -> torch.Tensor:
    """Return scaled consecutive-latent pose actions as [B, T_lat, 6].

    This follows the original alaya-world Camera AdaLN path: first sample the
    camera trajectory at latent-rate pixel indices, then use the relative pose
    from latent i-1 to latent i. Latent 0 has no previous latent and is zero.
    """
    from scipy.spatial.transform import Rotation as R

    c2w_np = c2w_normalized.detach().cpu().numpy() if isinstance(c2w_normalized, torch.Tensor) else np.asarray(c2w_normalized)
    if c2w_np.ndim == 3:
        c2w_np = c2w_np[None]
    if c2w_np.ndim != 4:
        raise ValueError(f"cam_c2w must be [F,4,4] or [B,F,4,4], got {c2w_np.shape}")

    scale = _parse_action_scale(action_scale)
    batch_out = []
    for c2w_b in c2w_np:
        num_frames = c2w_b.shape[0]
        latent_num = (num_frames - 1) // vae_temporal_stride + 1
        latent_frame_indices = [min(i * vae_temporal_stride, num_frames - 1) for i in range(latent_num)]
        c2ws = np.asarray([c2w_b[i] for i in latent_frame_indices], dtype=np.float64)

        action_vecs = np.zeros((latent_num, 6), dtype=np.float32)
        for i in range(1, latent_num):
            rel = np.linalg.inv(c2ws[i - 1]) @ c2ws[i]
            t_rel = rel[:3, 3].astype(np.float32)
            r_rel = R.from_matrix(rel[:3, :3]).as_euler("xyz", degrees=False).astype(np.float32)
            action_vecs[i] = np.concatenate([t_rel, r_rel], axis=0)

        action_vecs = action_vecs / scale[None, :]
        batch_out.append(torch.tensor(action_vecs, dtype=torch.float32))
    return torch.stack(batch_out, dim=0)


def _parse_action_scale(raw: str) -> np.ndarray:
    try:
        values = [float(x.strip()) for x in raw.split(",")]
        if len(values) != 6 or any(v <= 0 for v in values):
            raise ValueError
        return np.asarray(values, dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"control.action_scale must contain six positive floats, got {raw!r}") from exc
