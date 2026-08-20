"""Provide media, cache, camera-tensor, and model import helpers."""

from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from reactor_runtime import CommandError, UploadedFile

_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
_UPLOAD_MAX_PIXELS = 100_000_000
_UPLOAD_MIME_FORMATS = {
    "image/bmp": "BMP",
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


def compact_rollout_cache(
    cache: Any,
    *,
    max_spatial_frames: int,
    recent_spatial_frames: int,
) -> None:
    """Bound generated latents and spatial memory while retaining old keyframes."""
    preds = getattr(cache, "preds", None)
    if isinstance(preds, list) and len(preds) > 1:
        preds[:] = preds[-1:]

    bank = getattr(cache, "spatial_bank", None)
    if bank is None:
        return
    pixels = bank.pixels
    frame_indices = bank.frame_indices
    depths = bank.depths
    total = len(frame_indices)
    if len(pixels) != total or len(depths) != total:
        raise RuntimeError("AlayaWorld spatial bank members have different lengths")
    if total <= max_spatial_frames:
        return

    recent_count = min(recent_spatial_frames, max_spatial_frames)
    historical_count = total - recent_count
    historical_budget = max_spatial_frames - recent_count
    if historical_budget == 1:
        historical = [0]
    elif historical_budget > 1:
        historical = [
            index * (historical_count - 1) // (historical_budget - 1)
            for index in range(historical_budget)
        ]
    else:
        historical = []
    keep = historical + list(range(historical_count, total))

    bank.pixels = [pixels[index] for index in keep]
    bank.frame_indices = [frame_indices[index] for index in keep]
    bank.depths = [depths[index] for index in keep]
    world_points = getattr(bank, "world_points", None)
    if isinstance(world_points, dict):
        bank.world_points = {
            new_index: world_points[old_index]
            for new_index, old_index in enumerate(keep)
            if old_index in world_points
        }


def validate_uploaded_image(image: UploadedFile) -> None:
    """Reject oversized, mislabeled, or undecodable uploaded image bytes."""
    expected_format = _UPLOAD_MIME_FORMATS.get(image.mime_type.lower())
    if expected_format is None:
        raise CommandError(
            "unsupported_media",
            f"{image.name} must declare image/jpeg, image/png, image/webp, or image/bmp.",
        )
    if not image.data:
        raise CommandError("invalid_image", f"{image.name} is empty.")
    if image.size > _UPLOAD_MAX_BYTES:
        raise CommandError(
            "image_too_large",
            f"{image.name} exceeds the {_UPLOAD_MAX_BYTES // (1024 * 1024)} MiB limit.",
        )
    try:
        with Image.open(io.BytesIO(image.data)) as decoded:
            image_format = decoded.format or ""
            width, height = decoded.size
            if image_format != expected_format:
                raise CommandError(
                    "unsupported_media",
                    f"{image.name} contains {image_format or 'unknown'} data but declares "
                    f"{image.mime_type}.",
                )
            if width <= 0 or height <= 0 or width * height > _UPLOAD_MAX_PIXELS:
                raise CommandError(
                    "image_too_large",
                    f"{image.name} exceeds the {_UPLOAD_MAX_PIXELS}-pixel limit.",
                )
            decoded.verify()
    except CommandError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise CommandError("invalid_image", f"{image.name} cannot be decoded.") from error


def uploaded_image_video(
    image: UploadedFile,
    metadata: dict[str, Any],
    *,
    target_hw: tuple[int, int],
    torch_module: Any,
) -> Any:
    """Decode, center-crop, and repeat an upload over the camera template."""
    target_height, target_width = target_hw
    with Image.open(io.BytesIO(image.data)) as decoded:
        oriented = ImageOps.exif_transpose(decoded).convert("RGB")
        fitted = ImageOps.fit(
            oriented,
            (target_width, target_height),
            method=Image.Resampling.LANCZOS,
        )
        pixels = np.array(fitted, dtype=np.uint8, copy=True)
    frame = torch_module.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1.0
    frame_count = int(camera_frames(metadata["cam_c2w"]).shape[0])
    return frame.unsqueeze(0).expand(frame_count, -1, -1, -1).contiguous()


def load_model_modules(repo_root: Path) -> dict[str, Any]:
    """Import AlayaWorld's own inference surface from the repository.

    The adapter calls the same entry points as ``inference/run.py``: the engine
    builder, the rollout planner, and ``FlashAlayaPipeline``. Importing them
    lazily keeps the schema renderable without CUDA present.

    Args:
        repo_root: Repository root, placed on ``sys.path`` so ``alaya``,
            ``ltx2``, and ``inference`` import as they do from a shell there.
    """
    root = str(repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    rollout = importlib.import_module("alaya.inference.rollout_utils")
    return {
        "torch": importlib.import_module("torch"),
        "load_config": importlib.import_module("alaya.config.loader").load_config,
        "pipeline_type": importlib.import_module("alaya.inference.pipeline").FlashAlayaPipeline,
        "load_input_sample": rollout.load_input_sample,
        "check_input_resolution": rollout.check_input_resolution,
        "plan_rollout": rollout.plan_rollout,
        "build_engine": rollout.build_engine,
        "apply_da3_robust_scale": importlib.import_module(
            "inference.da3_patch"
        ).apply_da3_robust_scale,
        "pytorch_attention": importlib.import_module(
            "ltx2.modules.attention"
        ).AttentionFunction.PYTORCH,
    }


def set_attention_backend(engine: Any, attention_function: Any) -> int:
    """Select an attention callable on every loaded attention module."""
    changed = 0
    for root in (engine.transformer, engine.text_encoder):
        if root is None:
            continue
        for module in root.modules():
            if hasattr(module, "attention_function"):
                module.attention_function = attention_function
                changed += 1
    if changed == 0:
        raise RuntimeError("AlayaWorld exposed no configurable attention modules")
    return changed


class FlashAttention4:
    """Route attention through FlashAttention 4, keeping masked blocks on PyTorch.

    AlayaWorld calls its attention hook both with and without an additive or
    boolean mask. FlashAttention 4 covers the unmasked calls, including the
    sliding window it accepts natively, and the masked calls fall through to the
    PyTorch implementation that builds the equivalent banded mask.
    """

    def __init__(self, flash_attention: Any, masked_fallback: Any, torch_module: Any) -> None:
        self._flash_attention = flash_attention
        self._masked_fallback = masked_fallback
        self._torch = torch_module

    def __call__(
        self,
        q: Any,
        k: Any,
        v: Any,
        heads: int,
        mask: Any | None = None,
        window_size: tuple[int, int] | None = None,
    ) -> Any:
        """Return attention over ``[batch, tokens, heads * head_dim]`` inputs."""
        if mask is not None:
            return self._masked_fallback(q, k, v, heads, mask, window_size=window_size)
        batch, _, fused = q.shape
        head_dim = fused // heads
        query, key, value = (t.view(batch, -1, heads, head_dim) for t in (q, k, v))
        bfloat16 = self._torch.bfloat16
        out = self._flash_attention(
            query.to(bfloat16),
            key.to(bfloat16),
            value.to(bfloat16),
            # (-1, -1) is full attention; anything else is the token window the
            # caller asked for, in (left, right) form.
            window_size=window_size if window_size is not None else (-1, -1),
        )
        # The kernel returns the output alongside its log-sum-exp.
        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(batch, -1, heads * head_dim).to(value.dtype)


def resolve_attention_backend(
    backend: str,
    *,
    pytorch_attention: Any,
    torch_module: Any,
) -> Any | None:
    """Return the attention callable for *backend*, or ``None`` to leave AlayaWorld's.

    Raises:
        RuntimeError: FlashAttention 4 was asked for but cannot be imported.
    """
    if backend == "upstream":
        return None
    if backend == "pytorch":
        return pytorch_attention
    try:
        from flash_attn.cute import flash_attn_func
    except ImportError as error:
        raise RuntimeError(
            "inference.attention_backend is flash_attention_4 but flash-attn-4 is "
            "not importable; install it or set the backend to pytorch"
        ) from error
    return FlashAttention4(flash_attn_func, pytorch_attention, torch_module)


def camera_frames(camera: Any) -> Any:
    """Return camera poses as ``[F, 4, 4]`` from batched or unbatched input."""
    if camera.dim() == 3:
        return camera
    if camera.dim() == 4:
        return camera[0]
    raise ValueError(f"cam_c2w must be [F,4,4] or [B,F,4,4], got {tuple(camera.shape)}")


def ensure_camera_capacity(camera: Any, frame_count: int, torch_module: Any) -> Any:
    """Extend a camera tensor by repeating its final pose when needed."""
    current = int(camera_frames(camera).shape[0])
    if current >= frame_count:
        return camera
    missing = frame_count - current
    time_axis = 0 if camera.dim() == 3 else 1
    tail = camera[-1:] if camera.dim() == 3 else camera[:, -1:]
    repeats = [1] * camera.dim()
    repeats[time_axis] = missing
    return torch_module.cat([camera, tail.repeat(*repeats)], dim=time_axis)
