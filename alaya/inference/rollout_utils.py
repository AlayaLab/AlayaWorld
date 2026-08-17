"""Run/rollout helpers for the FlashAlaya CLI: Context-Parallel bootstrap, engine
build, rollout-length planning from the input camera trajectory, freeing the
generation models before decode, and an optional taehv preview decode.

All functions take explicit parameters (no argparse coupling) so they can be
reused / tested independently of run.py.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

import torch
import torch._dynamo
import torch.distributed as dist
import torchvision.transforms.functional as TF
from torchvision.io import read_image, read_video

from ltx2.utils.context_parallel import initialize_context_parallel
from alaya.inference.engine import InferenceEngine
from alaya.inference.taehv import TAEHV, StreamingTAEHV


def bootstrap_context_parallel(seed, compile_mode):
    """Init Context Parallel (Ulysses) when launched via torchrun (WORLD_SIZE>1).

    Every rank runs the full engine on identical conditions+noise; the DiT forward
    shards the sequence internally (pad->scatter->all-to-all->gather), so the
    pipeline is CP-agnostic and only rank0 saves. Returns
    (cp_size, is_rank0, seed, compile_mode) with seed/compile_mode adjusted for CP
    (seed forced so ranks draw identical noise; reduce-overhead downgraded since
    cudagraph degrades under per-layer all-to-all).
    """
    cp_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if cp_size <= 1:
        return cp_size, True, seed, compile_mode

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    # The platform image often exports NCCL_DEBUG=INFO, which floods the console
    # with per-channel topology logs from every rank. Quiet it (set before NCCL
    # initializes); opt back in with FLASH_NCCL_DEBUG=1.
    if os.environ.get("FLASH_NCCL_DEBUG") != "1":
        os.environ["NCCL_DEBUG"] = "WARN"
    dist.init_process_group("nccl")
    initialize_context_parallel(cp_size)
    if seed is None:
        seed = 1234
    if compile_mode == "reduce-overhead":
        compile_mode = "default"
    if rank == 0:
        print(f"[FlashAlaya:CP] cp_size={cp_size} seed={seed} compile={compile_mode}", flush=True)
    return cp_size, rank == 0, seed, compile_mode


def silence_worker_rank() -> None:
    """Mute non-rank0 ranks under Context Parallel so logs aren't printed N times.

    Redirects stdout to /dev/null and disables the HF / transformers progress bars
    (those go to stderr). stderr itself is kept so a crashing worker still shows a
    traceback. Only rank0 produces the user-facing output / saves the video.
    """
    import sys

    sys.stdout = open(os.devnull, "w")  # noqa: SIM115 (lives for process lifetime)
    for _disable in (
        lambda: __import__("transformers").utils.logging.disable_progress_bar(),
        lambda: __import__("huggingface_hub").utils.disable_progress_bars(),
    ):
        try:
            _disable()
        except Exception:
            pass


def build_engine(cfg, *, compile_mode, compile_aux, bank_taehv, verbose):
    """Load the standalone engine, optionally swap the bank decoder for taehv, and
    torch.compile the DiT."""
    engine = InferenceEngine(cfg)
    engine.setup(compile_aux=compile_aux, verbose=verbose)

    if bank_taehv:
        taehv_weights = cfg.paths.taehv
        if not taehv_weights:
            raise SystemExit("--bank-taehv set but paths.taehv is empty; set paths.taehv in the config")
        # LOSSY: bank pixels (DA3 depth + forward-warp sources) via the tiny taehv
        # decoder instead of the full LTX VAE. taeltx2_3_wide matches the VAE's
        # x8 temporal / x32 spatial so per-chunk frame counts are unchanged.
        tae = TAEHV(taehv_weights).to(engine.device, torch.float16).eval().requires_grad_(False)
        engine.spatial.enable_taehv_bank_decode(tae, dtype=torch.float16)
        if verbose:
            print(f"[FlashAlaya] bank decode = taehv (LOSSY) {taehv_weights}", flush=True)

    if compile_mode != "none":
        torch._dynamo.config.cache_size_limit = 64
        # Silence the benign graph-break log flood: the DiT forward has one
        # data-dependent branch on spatial coverage (`invalid_spatial.any().item()`
        # in model_ltx_2_3.py) that dynamo can't trace — it partitions cudagraphs
        # but is correct. Mute only that `.item()` emitter logger; other dynamo
        # warnings (recompiles / eager fallback) and other graph breaks still show.
        logging.getLogger("torch._dynamo.variables.tensor").setLevel(logging.ERROR)
        mode = None if compile_mode == "default" else compile_mode
        engine.transformer = torch.compile(engine.transformer, mode=mode, dynamic=False)
        # if verbose:
        #     print(f"[FlashAlaya] torch.compile(transformer, mode={compile_mode}) — first chunk compiles once", flush=True)
    return engine


def plan_rollout(cfg, video_pixels, metadata, *, rounds_cap, K, N, gap_steps, cond_end):
    """Decide the rollout length: follow the input camera trajectory to its end,
    capped at ``rounds_cap``. Length is bounded by min(video frames, camera poses).
    Trims ``video_pixels`` and per-frame metadata (cam_c2w*) to the exact window.

    Returns (video_pixels, metadata, rounds, max_rounds, needed_latents).
    """
    stride = int(cfg.sample.temporal_stride)
    prefix = cfg.layout.sink_latent_frames + gap_steps + N + (cond_end if N == 0 else 0)

    frames_in = int(video_pixels.shape[0])
    cam = metadata.get("cam_c2w")
    cam_frames = int(cam.shape[0]) if cam is not None and getattr(cam, "ndim", 0) >= 1 else frames_in
    avail_latents = (min(frames_in, cam_frames) - 1) // stride + 1
    max_rounds = (avail_latents - 1 - prefix) // K
    if max_rounds < 1:
        raise SystemExit(
            f"input too short: video={frames_in} cam={cam_frames} give 0 rounds "
            f"(need >= {(prefix + 1) * stride + 1} frames)"
        )
    rounds = min(int(rounds_cap), int(max_rounds))

    needed_latents = prefix + rounds * K + 1
    needed_pixels = (needed_latents - 1) * stride + 1
    if frames_in > needed_pixels:
        video_pixels = video_pixels[:needed_pixels]
    for key in ("cam_c2w", "cam_c2w_raw"):  # keep per-frame metadata aligned to the trim
        v = metadata.get(key)
        if v is not None and getattr(v, "ndim", 0) >= 1 and v.shape[0] > needed_pixels:
            metadata[key] = v[:needed_pixels]
    return video_pixels, metadata, rounds, max_rounds, needed_latents


def free_generation_models(engine) -> None:
    """Drop DiT / Gemma / history / DA3 after the rollout so the final VAE decode
    has VRAM headroom (decode needs only the VAE decoder + the cached latents)."""
    engine.transformer = engine.text_encoder = engine.encode_text = engine.history_encoder = None
    if engine.spatial is not None:
        engine.spatial.da3 = None
        engine.spatial._taehv = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_taehv_preview(engine, cfg, cache, *, taehv_weights, out_path, crf=28, N=None, gap_steps=0):
    """[Backup, not wired into the CLI] Decode a rollout with the streaming taehv
    tiny decoder (O(1) memory, no seam frames) and save an mp4. The history prefix
    is fed for streaming context then stripped, so the video starts at the first
    generated frame (same prefix-strip as the main decode)."""
    if N is None:
        N = int(cfg.layout.history_latent_frames)
    tae = TAEHV(taehv_weights).to(engine.device, torch.float16).eval().requires_grad_(False)
    stream = StreamingTAEHV(tae)
    frames: list[torch.Tensor] = []

    def push(latent_bcthw):
        x = latent_bcthw.permute(0, 2, 1, 3, 4).to(device=engine.device, dtype=torch.float16)
        for t in range(x.shape[1]):
            f = stream.decode(x[:, t : t + 1])
            while f is not None:
                frames.append((f[0, 0].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu())
                f = stream.decode()

    keep = max(1, min(int(cfg.validation.video_history_latent_frames or N), N))
    hist_end = cfg.layout.sink_latent_frames + gap_steps + N
    push(cache.latent_full[:, :, hist_end - keep : hist_end])  # context (stripped below)
    for pred in cache.preds:
        push(pred)
    prefix_pixel_len = (keep - 1) * int(cfg.sample.temporal_stride) + 1
    if 0 < prefix_pixel_len < len(frames):
        frames = frames[prefix_pixel_len:]
    video = torch.stack(frames, dim=0)
    engine.write_video(out_path, video, crf=crf)
    print(f"[FlashAlaya] saved {out_path} frames={tuple(video.shape)}", flush=True)


# ----------------------------------------------------------------- input I/O
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _fit_image_to(img: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Resize a [C,H,W] image so the short side covers the target, then center-crop
    to exactly target_hw. Used only for image input (videos must already match)."""
    th, tw = int(target_hw[0]), int(target_hw[1])
    _, h, w = img.shape
    if (h, w) == (th, tw):
        return img
    scale = max(th / h, tw / w)  # cover both dims, then crop the excess
    img = TF.resize(img, [round(h * scale), round(w * scale)], antialias=True)
    img = TF.center_crop(img, [th, tw])
    print(f"[FlashAlaya] image input: resized+center-cropped {h}x{w} -> {th}x{tw}", flush=True)
    return img


def load_input_sample(path: str, *, image_target_hw: tuple[int, int] | None = None):
    """Load a user-provided sample. Accepted forms:

    1. scene triplet (video OR first-frame image) + camera + prompt:
       <prefix>_video.mp4   conditioning video (h264, viewable), OR
       <prefix>_image.<ext> a single first frame (png/jpg/jpeg/webp/bmp)
       <prefix>_camera.pt   metadata dict (cam_c2w [F,4,4], intrinsic [3,3], ...)
       <prefix>_prompt.txt  final text prompt
       --input may be the prefix or any one of the member files.
       A video is used as-is; an image is replicated to the camera-trajectory
       length to seed the history window (the model needs ~5.4s of history to
       start; see README), then generation follows the camera.
    2. single .pt dict {"video_pixels": Tensor[F,C,H,W] in [-1,1],
       "caption": str, "metadata": dict}.

    Returns (video_pixels [F,C,H,W] float in [-1,1], caption, metadata).
    """
    p = Path(path)
    # resolve scene-triplet prefix from any of its member files
    stem = str(p)
    for suffix in ("_video.mp4", "_camera.pt", "_prompt.txt", *(f"_image{e}" for e in _IMAGE_EXTS)):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    video_p, camera_p, prompt_p = (Path(stem + s) for s in ("_video.mp4", "_camera.pt", "_prompt.txt"))
    image_p = next((Path(stem + f"_image{e}") for e in _IMAGE_EXTS if Path(stem + f"_image{e}").exists()), None)

    if camera_p.exists() and prompt_p.exists() and (video_p.exists() or image_p is not None):
        metadata = dict(torch.load(camera_p, map_location="cpu", weights_only=False))
        caption = prompt_p.read_text(encoding="utf-8").strip()
        if video_p.exists():
            frames, _, _ = read_video(str(video_p), pts_unit="sec", output_format="TCHW")  # uint8 [F,C,H,W]
            video_pixels = frames.float() / 127.5 - 1.0
        else:
            img = read_image(str(image_p))[:3].float() / 127.5 - 1.0  # [C,H,W] in [-1,1] (drop alpha)
            if image_target_hw is not None:  # auto-fit images (videos must already match)
                img = _fit_image_to(img, image_target_hw)
            cam = metadata.get("cam_c2w")
            n_frames = int(cam.shape[1] if getattr(cam, "ndim", 0) == 4 else cam.shape[0]) if cam is not None else 1
            video_pixels = img.unsqueeze(0).expand(max(1, n_frames), -1, -1, -1).contiguous()  # [F,C,H,W]
            print(f"[FlashAlaya] image input: replicated first frame x{video_pixels.shape[0]} to seed history", flush=True)
        return video_pixels, caption, metadata

    if not p.exists():
        raise SystemExit(
            f"--input {path}: not found (and no {stem}_video.mp4 or {stem}_image.<ext> "
            f"+ {stem}_camera.pt + {stem}_prompt.txt triplet)"
        )
    sample = torch.load(p, map_location="cpu", weights_only=False)
    try:
        video_pixels = sample["video_pixels"]
        caption = str(sample["caption"])
        metadata = dict(sample["metadata"])
    except (TypeError, KeyError) as exc:
        raise SystemExit(
            f"--input {path}: expected a scene triplet (<prefix>_video.mp4/_camera.pt/_prompt.txt) "
            f"or a .pt dict with video_pixels/caption/metadata; got {type(sample).__name__} ({exc})"
        )
    if video_pixels.ndim != 4:
        raise SystemExit(f"--input video_pixels must be [F,C,H,W], got {tuple(video_pixels.shape)}")
    return video_pixels, caption, metadata


def check_input_resolution(video_pixels, cfg) -> None:
    """Fail fast (before loading any model) if the input video resolution does not
    match cfg.sample.height/width. The spatial-memory warp builds depth/points at
    the configured size but samples RGB from the raw video, so a mismatch otherwise
    crashes deep in the rollout (CUDA index out of bounds) after a long model load."""
    h, w = int(video_pixels.shape[-2]), int(video_pixels.shape[-1])
    exp_h, exp_w = int(cfg.sample.height), int(cfg.sample.width)
    if (h, w) != (exp_h, exp_w):
        raise SystemExit(
            f"input video resolution {h}x{w} != config sample {exp_h}x{exp_w}. "
            f"Provide a {exp_h}x{exp_w} video or set "
            f"sample.height/width in the config to match the input."
        )


def apply_joystick_overlay(cfg, cache, frames):
    """Draw the dual-joystick (Move/Rotate) HUD on the decoded frames and return them.

    The joystick directions are derived from the SAME camera trajectory the model
    was conditioned on: the generated output starts at latent ``target_base_start``
    (pixel index target_base_start*stride), so we slice cam_c2w from there and let
    c2w_to_action_labels / add_joystick_overlay map latent labels back to frames.
    Returns ``frames`` unchanged if there is no camera trajectory.
    """
    import numpy as np
    from ltx2.modules.camera_control import add_joystick_overlay, c2w_to_action_labels

    cam = cache.metadata.get("cam_c2w")
    if cam is None:
        print("[FlashAlaya] no cam_c2w in metadata; joystick overlay skipped", flush=True)
        return frames
    if cam.dim() == 4:  # [B,F,4,4] -> [F,4,4]
        cam = cam[0]
    stride = int(cfg.sample.temporal_stride)
    start = int(cache.target_base_start) * stride
    n = int(frames.shape[0])
    cam_window = cam[start : start + n].to(dtype=torch.float32).cpu().numpy()
    labels = c2w_to_action_labels(cam_window, vae_temporal_stride=stride)
    frame_list = [frames[i].cpu().numpy() for i in range(n)]
    frame_list = add_joystick_overlay(frame_list, labels, vae_temporal_stride=stride)
    return torch.from_numpy(np.stack(frame_list, axis=0))
