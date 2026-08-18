from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Custom image-to-video: start from a single image and drive autoregressive generation with a
#   real camera trajectory (cam_c2w). Images come from a plain directory and the camera poses are
#   supplied by the caller rather than synthesized. Intrinsics default to a placeholder because the
#   motion is determined by the inter-frame translation of cam_c2w and not by the intrinsics.
_GENERIC_CAPTION = "A first-person walking tour moving forward through the scene."
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Placeholder normalized intrinsics (fx/W = fy/H = 0.5, principal point centred)
_DEFAULT_INTRINSIC = torch.tensor(
    [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]], dtype=torch.float32
)


class CustomI2VDataset(Dataset):
    """Pair each image with one camera trajectory (deterministic image_i <-> pose_i; poses are tiled cyclically)."""

    def __init__(
        self,
        *,
        image_dir: str,
        pose_jsonl: str,
        annotation_base_dir: str | None,
        width: int,
        height: int,
        frames: int,
        traj_frames: int = 8192,
        caption: str | None = None,
        pose_offset: int = 0,
        poses_per_image: int = 1,
        pose_stride: int = 40,
        captions_json: str | None = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.pose_offset = int(pose_offset)      # image i is paired with pose[(i + offset) % N]
        self.poses_per_image = max(1, int(poses_per_image))  # trajectories per image (group g uses +g*pose_stride)
        self.pose_stride = int(pose_stride)      # pose index gap between groups of the same image
        self.frames = int(frames)               # number of seed frames the single image is repeated to
        self.traj_frames = int(traj_frames)      # trajectory length after cyclic tiling
        self.caption = str(caption or _GENERIC_CAPTION)
        # Per-image captions keyed by file name with or without extension; falls back to a generic prompt.。
        self.captions: dict[str, str] = {}
        if captions_json:
            with open(captions_json, "r", encoding="utf-8") as f:
                self.captions = {str(k): str(v) for k, v in json.load(f).items()}
        self._to_tensor = transforms.ToTensor()

        self.images = sorted(
            p for p in Path(image_dir).iterdir() if p.suffix.lower() in _IMG_EXTS
        )
        if not self.images:
            raise FileNotFoundError(f"no images ({_IMG_EXTS}) under {image_dir}")

        self.pose_entries: list[dict[str, Any]] = []
        with open(pose_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.pose_entries.append(json.loads(line))
        if not self.pose_entries:
            raise ValueError(f"empty pose_jsonl: {pose_jsonl}")

        self._ann_roots = [
            r for r in (
                annotation_base_dir,
                "data/Annotation",
                "data/Annotation",
            ) if r
        ]

    def __len__(self) -> int:
        return len(self.images) * self.poses_per_image

    def _map(self, index: int) -> tuple[int, int]:
        """index -> (image index, group). The first num_images indices are group 0, then group 1, ..."""
        n = len(self.images)
        return int(index) % n, int(index) // n

    def _pose_index(self, index: int) -> int:
        img_idx, group = self._map(index)
        return (img_idx + self.pose_offset + group * self.pose_stride) % len(self.pose_entries)

    def _resolve_pose(self, rel: str) -> str | None:
        if not rel:
            return None
        if os.path.isabs(rel) and os.path.exists(rel):
            return rel
        for root in self._ann_roots:
            p = os.path.join(root, rel)
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def _tile_axis0(t: torch.Tensor, n: int) -> torch.Tensor:
        """Cyclically repeat along axis 0 up to length n."""
        if int(t.shape[0]) >= n:
            return t[:n].contiguous()
        reps = (n + int(t.shape[0]) - 1) // int(t.shape[0])
        return t.repeat(reps, *([1] * (t.dim() - 1)))[:n].contiguous()

    def _load_cam(self, idx: int) -> torch.Tensor:
        # Deterministic pairing: pose = (image index + offset + group * stride) % N
        ent = self.pose_entries[self._pose_index(idx)]
        pp = self._resolve_pose(str(ent.get("pose_path", "")))
        if pp is None:
            raise FileNotFoundError(f"pose npz not found for entry: {ent.get('pose_path')}")
        z = np.load(pp)
        cam = torch.from_numpy(np.asarray(z["cam_c2w"], dtype=np.float32))   # [N,4,4]
        # Optional real normalized intrinsics: "intrinsic" in pose.jsonl (3x3 or [fx,fy,cx,cy]) wins, then the npz key
        K = ent.get("intrinsic")
        if K is None and "intrinsic" in getattr(z, "files", []):
            K = np.asarray(z["intrinsic"], dtype=np.float32)
        if K is not None:
            K = np.asarray(K, dtype=np.float32)
            if K.shape == (4,):
                fx, fy, cx, cy = [float(v) for v in K]
                K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            if K.shape != (3, 3):
                raise ValueError(f"intrinsic must be 3x3 or [fx,fy,cx,cy], got {K.shape}")
            self._entry_intrinsic = torch.from_numpy(K)
        else:
            self._entry_intrinsic = None
        return self._tile_axis0(cam, self.traj_frames)                       # tile cyclically

    def __getitem__(self, index: int) -> dict[str, Any]:
        img_idx, group = self._map(index)
        img_path = self.images[img_idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self._to_tensor(image)
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=(self.height, self.width),
            mode="bicubic", align_corners=False, antialias=True,
        ).squeeze(0).contiguous()
        tensor = tensor.sub_(0.5).div_(0.5)                                  # [-1,1]
        video_pixels = tensor.unsqueeze(0).repeat(self.frames, 1, 1, 1).contiguous()

        cam_c2w = self._load_cam(int(index))                                 # extrinsics trajectory (tiled)
        _real_K = getattr(self, "_entry_intrinsic", None)
        intrinsic = _real_K.clone() if _real_K is not None else _DEFAULT_INTRINSIC.clone()
        metadata: dict[str, Any] = {
            "intrinsic": intrinsic,
            "cam_c2w": cam_c2w,
            "intrinsic_raw": intrinsic.clone(),
            "cam_c2w_raw": cam_c2w.clone(),
            "has_camera": True,
            # True = caller supplied real intrinsics (used directly by the warp); False = placeholder + ViGeo fit
            "has_real_intrinsic": bool(_real_K is not None),
            "video_id": img_path.stem if self.poses_per_image == 1 else f"{img_path.stem}_p{group}",
            "source": "custom_i2v",
            "caption_type": "custom",
            "pose_orig_w": float(self.width),
            "pose_orig_h": float(self.height),
            "frame_start": 0,
            "frame_end": int(self.traj_frames) - 1,
        }
        # Per-image caption (try stem then file name), falling back to the generic prompt
        caption = self.captions.get(img_path.stem) or self.captions.get(img_path.name) or self.caption
        return {"video_pixels": video_pixels, "caption": caption, "metadata": metadata}
