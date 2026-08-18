"""On-disk whole-clip VAE latent cache. Training encodes the head of a window fresh and slices
the tail from this cache, reproducing a full fresh window encode exactly.

A causal VAE carries roughly 16 latents of temporal memory:
the relative difference between a sliced whole-clip encode and a fresh window encode is ~100% at
latent 0, ~2% at 8, ~0.5% at 12 and exactly 0 from 16 onwards.
So a training window encodes only the first FRESH_HEAD_LAT latents (129 pixel frames) fresh and
takes the rest from the cache, which is bit-identical to encoding the whole window fresh.

Storage: latent [128,T,17,30] bf16 stored as a uint16 view in .npy (memmap-friendly read slices),
with a sidecar {h}.json holding {ratio, n_lat, height, width}. Key = sha1("{source}|{video_id}|{H}x{W}|fps{fps}").
Alignment requirement: the window start (in source frames) must equal m * 8 * ratio and that value
must be an integer. Unaligned or uncached lookups return None and the caller encodes the full window fresh.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

GRID_LAT = 8               # 1 latent = 8 pixel frames (temporal stride)
FRESH_HEAD_LAT = 17        # latents encoded fresh at the window head (>=16 makes the sliced tail exact)


def _key(source: str, video_id: str, height: int, width: int, fps: float) -> str:
    return hashlib.sha1(f"{source}|{video_id}|{height}x{width}|fps{fps:g}".encode()).hexdigest()


def entry_paths(cache_dir: str, source: str, video_id: str, height: int, width: int, fps: float) -> tuple[str, str]:
    h = _key(source, video_id, height, width, fps)
    base = os.path.join(cache_dir, h[:2], h)
    return base + ".npy", base + ".json"


def save_entry(cache_dir: str, source: str, video_id: str, height: int, width: int, fps: float,
               latent: torch.Tensor, ratio: float) -> None:
    """Write latent [C,T,H_lat,W_lat] bf16 atomically."""
    npy_p, json_p = entry_paths(cache_dir, source, video_id, height, width, fps)
    os.makedirs(os.path.dirname(npy_p), exist_ok=True)
    arr = latent.detach().to("cpu", torch.bfloat16).contiguous().view(torch.uint16).numpy()
    tmp = npy_p + f".tmp{os.getpid()}.npy"
    np.save(tmp, arr)
    os.replace(tmp, npy_p)
    meta = {"ratio": float(ratio), "n_lat": int(latent.shape[1]), "shape": list(arr.shape)}
    tmp_j = json_p + f".tmp{os.getpid()}"
    with open(tmp_j, "w") as f:
        json.dump(meta, f)
    os.replace(tmp_j, json_p)


def load_meta(cache_dir: str, source: str, video_id: str, height: int, width: int, fps: float) -> dict | None:
    _npy, json_p = entry_paths(cache_dir, source, video_id, height, width, fps)
    if not os.path.exists(json_p):
        return None
    try:
        with open(json_p) as f:
            return json.load(f)
    except Exception:
        return None


def load_slice(cache_dir: str, source: str, video_id: str, height: int, width: int, fps: float,
               m_start: int, n_lat: int) -> torch.Tensor | None:
    """Read-only memmap slice [C, n_lat, H_lat, W_lat] bf16; returns None if missing or out of range."""
    npy_p, json_p = entry_paths(cache_dir, source, video_id, height, width, fps)
    if not (os.path.exists(npy_p) and os.path.exists(json_p)):
        return None
    try:
        arr = np.load(npy_p, mmap_mode="r")
        if m_start < 0 or m_start + n_lat > arr.shape[1]:
            return None
        sl = np.ascontiguousarray(arr[:, m_start : m_start + n_lat])
        return torch.from_numpy(sl).view(torch.bfloat16)
    except Exception:
        return None


def aligned_m_start(frame_start: int, ratio: float) -> int | None:
    """Map a source-frame start to the whole-clip latent grid index m; returns None when unaligned.

    A 1e-2 tolerance absorbs timebase drift in the decoder's average fps (e.g. 30.0003 -> q=10.0001):
    the dataset quantizes starts to an integer q, so m = frame_start/q is exact after rounding.
    """
    q = GRID_LAT * ratio
    if q <= 0:
        return None
    m = frame_start / q
    if abs(m - round(m)) > 1e-2:
        return None
    return int(round(m))
