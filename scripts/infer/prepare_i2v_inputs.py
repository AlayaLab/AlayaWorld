#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turn CLI arguments into the three inputs custom_i2v expects:
images/ + captions.json + pose.jsonl (plus an extrinsics .npz).

Extrinsics, pick one:
  --extrinsics FILE      bring your own camera-to-world poses: .npz (key cam_c2w, [N,4,4])
                         or .npy/.txt ([N,16] or [N,4,4])
  --synth-frames N       synthesize a trajectory from --forward / --yaw / --pitch
Intrinsics are optional. Without them a placeholder fx=fy=0.5 is used and the warp falls back
to ViGeo-fitted intrinsics.
  --intrinsic fx fy cx cy   normalized intrinsics (fx/W, fy/H, cx/W, cy/H)

Example:
  python scripts/infer/prepare_i2v_inputs.py --image in.png --prompt "walking forward" \
      --synth-frames 256 --forward 0.0049 --yaw 0.15 --out outputs/infer_i2v_camera/inputs
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np


def synth_trajectory(n: int, forward: float, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Build c2w [n,4,4]: each frame advances `forward` along its own -z axis and accumulates yaw/pitch."""
    poses = np.zeros((n, 4, 4), dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    for i in range(n):
        poses[i, :3, :3] = R
        poses[i, :3, 3] = t
        poses[i, 3, 3] = 1.0
        ya, pa = math.radians(yaw_deg), math.radians(pitch_deg)
        Ry = np.array([[math.cos(ya), 0, math.sin(ya)],
                       [0, 1, 0],
                       [-math.sin(ya), 0, math.cos(ya)]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, math.cos(pa), -math.sin(pa)], [0, math.sin(pa), math.cos(pa)]], dtype=np.float32)
        R = R @ Ry @ Rx
        t = t + R @ np.array([0, 0, forward], dtype=np.float32)   # +z is forward (matches the training pose convention)
    return poses


def load_extrinsics(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        z = np.load(path)
        key = "cam_c2w" if "cam_c2w" in z.files else z.files[0]
        arr = np.asarray(z[key], dtype=np.float32)
    elif path.suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float32)
    else:
        arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 16:
        arr = arr.reshape(-1, 4, 4)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"extrinsics must be [N,4,4] or [N,16], got {arr.shape}")
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="single first frame image")
    ap.add_argument("--prompt", required=True, help="text prompt")
    ap.add_argument("--out", default="outputs/infer_i2v_camera/inputs")
    ap.add_argument("--extrinsics", default=None, help="file holding your own camera-to-world trajectory")
    ap.add_argument("--synth-frames", type=int, default=None,
                    help="length of the synthesized trajectory; derived from --rounds when omitted. "
                         "The trajectory is tiled cyclically, so a short one jumps back to its start.")
    ap.add_argument("--rounds", type=int, default=5,
                    help="only used to size the trajectory; the actual round count is written to the config by generate_video.sh")
    ap.add_argument("--forward", type=float, default=0.0049, help="per-frame translation (same scale as training poses)")
    ap.add_argument("--yaw", type=float, default=0.0, help="per-frame yaw in degrees")
    ap.add_argument("--pitch", type=float, default=0.0, help="per-frame pitch in degrees")
    ap.add_argument("--intrinsic", type=float, nargs=4, default=None,
                    metavar=("FX", "FY", "CX", "CY"), help="normalized intrinsics; omit to use the placeholder")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    img = Path(args.image)
    shutil.copy(img, out / "images" / img.name)

    if args.extrinsics:
        cam = load_extrinsics(Path(args.extrinsics))
        how = f"provided by {args.extrinsics}"
    else:
        # 4 latents per round x stride 8 = 32 pixel frames; 3x margin, at least 512
        n = args.synth_frames or max(512, args.rounds * 32 * 3)
        cam = synth_trajectory(n, args.forward, args.yaw, args.pitch)
        how = f"synthesized {n} frames (forward={args.forward}/frame, yaw={args.yaw} deg, pitch={args.pitch} deg)"

    npz = out / "cam_c2w.npz"
    save = {"cam_c2w": cam.astype(np.float32)}
    entry = {"pose_path": str(npz.resolve()), "assigned_image": img.name, "note": how}
    if args.intrinsic:
        fx, fy, cx, cy = args.intrinsic
        save["intrinsic"] = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        entry["intrinsic"] = [fx, fy, cx, cy]
    np.savez(npz, **save)

    (out / "captions.json").write_text(json.dumps({img.name: args.prompt}, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    with open(out / "pose.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    step = float(np.linalg.norm(np.diff(cam[:, :3, 3], axis=0), axis=1).mean()) if len(cam) > 1 else 0.0
    print(f"[prepare] image     : {out / 'images' / img.name}")
    print(f"[prepare] prompt    : {args.prompt[:70]}{'...' if len(args.prompt) > 70 else ''}")
    print(f"[prepare] extrinsics: {cam.shape} ({how}), mean inter-frame translation {step:.4f}")
    print(f"[prepare] intrinsic : {'real ' + str(args.intrinsic) if args.intrinsic else 'placeholder fx=fy=0.5 (ViGeo-fitted fallback)'}")
    print(f"[prepare] inputs written to {out}")


if __name__ == "__main__":
    main()
