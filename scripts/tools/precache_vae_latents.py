"""Offline prebuild of the whole-clip VAE latent cache.
Training encodes the first latents of a window fresh and slices the tail from this cache.

Usage (single or multi-GPU sharding, resumable):
  PYTHONPATH=. python scripts/tools/precache_vae_latents.py \
      --config configs/stage1_pretrain_bidir.yaml [--rank 0 --world 2] [--device cuda:0] [--limit N]

Pixel processing mirrors the dataset: uniform_sample on the 24fps grid -> ToTensor ->
bicubic(antialias) resize (H,W) → [-1,1] → StreamingVAEEncoder(chunk 33)。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from alaya.config.loader import load_config
from alaya.data.vae_latent_cache import entry_paths, save_entry
from alaya.model.loader import load_vae
from fastvideo.ltx2_streaming_vae import StreamingVAEEncoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    ap.add_argument("--world", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N videos (smoke test)")
    ap.add_argument("--batch_chunk", type=int, default=33)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_dir = cfg.runtime.vae_latent_cache_dir
    assert cache_dir, "runtime.vae_latent_cache_dir must be set in the config"
    dev = torch.device(args.device or f"cuda:{args.rank % max(1, torch.cuda.device_count())}")
    dtype = torch.bfloat16
    H, W = int(cfg.sample.height), int(cfg.sample.width)
    fps = float(cfg.sample.fps)
    stride = int(cfg.sample.temporal_stride)

    # Enumerate unique videos from the training dataset so source/video_id match training
    from alaya.data.dataloader import build_train_dataloader
    from alaya.utils.distributed import DistributedState

    dist_state = DistributedState(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))
    loader = build_train_dataloader(cfg, dist_state)
    ds = loader.dataset
    subsets = [ds] if hasattr(ds, "samples") else list(getattr(ds, "datasets", []))
    videos: dict[tuple[str, str], str] = {}
    for sub in subsets:
        for video_path, _cap, _pose, source, video_id in sub.samples:
            videos.setdefault((str(source), str(video_id)), str(video_path))
    items = sorted(videos.items())
    if args.limit:
        items = items[: args.limit]
    missing = [
        (k, v) for k, v in items
        if not os.path.exists(entry_paths(cache_dir, k[0], k[1], H, W, fps)[0])
    ]
    shard = missing[args.rank :: max(1, args.world)]
    print(
        f"[VAECache] unique videos {len(items)}, already cached {len(items) - len(missing)}, "
        f"this rank ({args.rank}/{args.world}) will encode {len(shard)} (dir={cache_dir})",
        flush=True,
    )
    if not shard:
        return

    vae_enc_raw, _dec = load_vae(cfg.paths.vae, device=dev, dtype=dtype)
    enc = StreamingVAEEncoder(vae_enc_raw, device=dev, dtype=dtype)
    import decord  # noqa: E402 - imported after torch to avoid a thread conflict

    t0 = time.time()
    done = 0
    total = len(shard)
    report_every = max(1, total // 100)
    for (source, video_id), video_path in shard:
        try:
            vr = decord.VideoReader(video_path, num_threads=2)
            video_fps = float(vr.get_avg_fps())  # raw value, matching the dataset fps ratio (no rounding)
            ratio = video_fps / fps
            total_src = len(vr)
            # Largest 24fps grid length n with round((n-1)*ratio) inside the clip and (n-1)%stride==0
            n = int((total_src - 1) / ratio) + 1
            n = ((n - 1) // stride) * stride + 1
            if n < stride + 1:
                done += 1
                continue
            ids = [min(int(j * ratio), total_src - 1) for j in range(n)]
            frames = vr.get_batch(ids).asnumpy()  # [T,H0,W0,C] uint8
            px = torch.from_numpy(frames).float().div_(255.0).permute(0, 3, 1, 2)  # [T,C,H0,W0]
            px = torch.nn.functional.interpolate(
                px, size=(H, W), mode="bicubic", align_corners=False, antialias=True
            )
            px = px.sub_(0.5).div_(0.5).permute(1, 0, 2, 3).to(dev, dtype)  # [C,T,H,W]
            with torch.no_grad():
                latent = enc.encode(px.unsqueeze(0), chunk_size=args.batch_chunk, verbose=False)[0]
            save_entry(cache_dir, source, video_id, H, W, fps, latent, ratio)
            del px, latent, frames
        except Exception as e:  # one failed video must not abort the shard
            print(f"[VAECache] SKIP {source}/{video_id}: {type(e).__name__} {e}", flush=True)
        done += 1
        if done % report_every == 0 or done == total:
            el = time.time() - t0
            ips = done / max(el, 1e-6)
            eta = (total - done) / max(ips, 1e-6)
            bar_n = int(30 * done / total)
            print(
                f"[VAECache] rank{args.rank} |{'█' * bar_n}{'.' * (30 - bar_n)}| "
                f"{done}/{total} ({100 * done / total:.1f}%) {ips:.2f} vid/s ETA {eta / 3600:.2f}h",
                flush=True,
            )
    print(f"[VAECache] rank{args.rank} done in {(time.time() - t0) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
