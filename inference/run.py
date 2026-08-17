"""Alaya World — official image-to-video (i2v) inference entry.

This is the outward-facing CLI for external users. It is a thin wrapper over the
`alaya.inference` engine: it reuses the engine's rollout helpers and streaming
pipeline unchanged, and only wires up an i2v-oriented command line + sensible
defaults. No engine logic is copied here.

Given a single first-frame image, a camera/action trajectory, and a text prompt,
the model autoregressively rolls out a video that follows the camera path. One
chunk = 4 latent frames = 32 pixel frames ≈ 1.33s @ 24fps, so a ~1-minute clip
is roughly 45 chunks and needs a camera trajectory of ≥ ~1450 frames.

Input (a "case" — the layout used by playground/case1):

    <prefix>_image.<png|jpg|jpeg|webp|bmp>   first frame (seeds the history)
    <prefix>_camera.pt                       metadata dict (cam_c2w [F,4,4], ...)
    <prefix>_prompt.txt                      text prompt
    <prefix>_skill.txt        (optional)     end-of-clip skill prompt (see --skill-sec)

`--input` may be the prefix or any one of those files.

Usage:
    # single GPU
    PYTORCH_ALLOC_CONF=expandable_segments:True \
        python -m inference.run --input playground/case1/case1

    # multi-GPU (Ulysses Context Parallel; e.g. 2 or 4 GPUs)
    OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
        torchrun --nproc_per_node=4 -m inference.run --input playground/case1/case1
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist

from alaya.config.loader import load_config
from alaya.inference.pipeline import FlashAlayaPipeline
from alaya.inference.rollout_utils import (
    apply_joystick_overlay,
    bootstrap_context_parallel,
    build_engine,
    check_input_resolution,
    free_generation_models,
    load_input_sample,
    plan_rollout,
    silence_worker_rank,
)


_INPUT_SUFFIXES = ("_video.mp4", "_camera.pt", "_prompt.txt", "_skill.txt",
                   "_image.png", "_image.jpg", "_image.jpeg", "_image.webp", "_image.bmp")


def resolve_skill_prompt(input_path: str) -> str | None:
    """Auto-detect a per-case end-skill caption at ``<prefix>_skill.txt`` (same
    prefix resolution as load_input_sample). Returns its stripped text, or None if
    the case ships no skill file."""
    stem = str(input_path)
    for suffix in _INPUT_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    skill_p = Path(stem + "_skill.txt")
    if skill_p.exists():
        text = skill_p.read_text(encoding="utf-8").strip()
        return text or None
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="inference.run",
        description=(
            "Alaya World official image-to-video inference. "
            "Single GPU: python -m inference.run --input <case>; "
            "multi-GPU Context Parallel: torchrun --nproc_per_node=N (e.g. 2 or 4) -m inference.run --input <case>."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help="case prefix (or any member file) with <prefix>_image.<ext> + _camera.pt + _prompt.txt",
    )
    p.add_argument("--cfg", default="configs/infer.yaml", help="inference config (yaml); model paths live under paths:")
    p.add_argument(
        "--output-dir",
        default=None,
        help="where to write the mp4 (overrides run.output_dir in the config); saved as <output-dir>/<case>_rounds-N.mp4",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=1000,
        help="max autoregressive chunks (~1.33s each); actual = min(this, camera-trajectory length). ~45 => ~1 min",
    )
    p.add_argument("--seed", type=int, default=None, help="fix per-chunk noise seed for reproducible runs")
    p.add_argument(
        "--compile",
        default="reduce-overhead",
        choices=["reduce-overhead", "default", "max-autotune", "none"],
        help="torch.compile mode for the DiT (under CP, reduce-overhead auto-downgrades to default)",
    )
    p.add_argument("--flex-attn", action=argparse.BooleanOptionalAction, default=True, help="fused flex_attention")
    p.add_argument(
        "--joystick",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="overlay the Move/Rotate joystick HUD (default: follow config validation.save_joystick)",
    )
    p.add_argument(
        "--ttc",
        action="store_true",
        help="Pathwise Test-Time Correction (arXiv:2602.05871): re-anchor each chunk to the first frame to "
        "curb appearance/style drift over long (~1 min) rollouts. OFF by default; knobs in config validation.ttc.",
    )
    p.add_argument("--video-crf", type=int, default=28, help="h264 crf for saved mp4s (18 near-lossless, 28 small)")
    p.add_argument(
        "--skill-sec",
        type=float,
        default=4.0,
        help="cast a one-off end effect: switch the text prompt for the final N seconds so a summoned "
        "creature / energy burst erupts at the very end (a single caption can't time this). 0 disables. "
        "Prompt comes from --skill-prompt or the case's <prefix>_skill.txt.",
    )
    p.add_argument(
        "--skill-prompt",
        default=None,
        help="inline skill caption for the final --skill-sec seconds (overrides the case's <prefix>_skill.txt)",
    )
    p.add_argument(
        "--skill-keep-wrap",
        action="store_true",
        help="keep the spatial-memory (forward-warp) conditioning during the skill window. By default it is "
        "DROPPED at skill onset so the summoned effect is not suppressed by the old scene's geometry; "
        "set this to keep the geometry locked (skill effects will be much weaker).",
    )
    return p.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    # args=None: CLI mode (parse argv). A pre-built namespace is passed by the
    # unified config entry (alaya.train dispatch on da3_infer.enabled).
    if args is None:
        args = parse_args()
    flex_attn = args.flex_attn and args.compile != "none"
    if args.flex_attn and not flex_attn:
        print("[AlayaWorld] --compile none: flex-attn off (eager math fallback)", flush=True)

    cp_size, is_rank0, args.seed, args.compile = bootstrap_context_parallel(args.seed, args.compile)
    if not is_rank0:
        silence_worker_rank()  # only rank0 prints / saves; workers stay quiet (stderr kept for tracebacks)

    cfg = load_config(args.cfg)
    if args.output_dir is not None:
        cfg.run.output_dir = args.output_dir
    if args.joystick is not None:
        cfg.validation.save_joystick = args.joystick

    # Read + validate the input BEFORE loading any model, so a bad input fails
    # immediately instead of after the (~40s) engine load. An image is auto-fit to
    # the config resolution and replicated to the camera length to seed history.
    video_pixels, caption, metadata = load_input_sample(
        args.input, image_target_hw=(int(cfg.sample.height), int(cfg.sample.width))
    )
    check_input_resolution(video_pixels, cfg)

    engine = build_engine(
        cfg, compile_mode=args.compile, compile_aux=False, bank_taehv=False, verbose=is_rank0
    )

    # DA3 has now been loaded by the engine, so depth_anything_3 is importable.
    # Make its camera-to-depth scale alignment robust to (near-)colinear trajectories
    # (straight forward/lateral moves) — otherwise Umeyama degenerates, camera
    # conditioning is dropped, and forward motion renders as a zoom. See inference/da3_patch.py.
    from inference.da3_patch import apply_da3_robust_scale
    if apply_da3_robust_scale() and is_rank0:
        print("[AlayaWorld] DA3 robust scale-only fallback enabled (colinear-safe depth scaling)", flush=True)

    # Rollout layout is derived from the config's (single) validation mode.
    mode_cfg = next(iter(cfg.validation.modes.values()))
    K = int(mode_cfg.layout.output_latent_frames)
    N = int(
        cfg.layout.history_latent_frames
        if mode_cfg.layout.history_latent_frames is None
        else mode_cfg.layout.history_latent_frames
    )
    gap_steps = int(float(mode_cfg.layout.max_gap_sec or 0.0) * cfg.sample.fps / cfg.sample.temporal_stride)
    cond_end = int(mode_cfg.layout.condition_latent_frames)
    video_pixels, metadata, rounds, max_rounds, needed_latents = plan_rollout(
        cfg, video_pixels, metadata, rounds_cap=args.rounds, K=K, N=N, gap_steps=gap_steps, cond_end=cond_end
    )

    if is_rank0:
        approx_sec = rounds * K * int(cfg.sample.temporal_stride) / float(cfg.sample.fps)
        print(
            f"[AlayaWorld] input={args.input} rounds={rounds}/{max_rounds} (cap {args.rounds}) "
            f"~{approx_sec:.0f}s @ {cfg.sample.fps:g}fps  caption={caption[:60]!r}",
            flush=True,
        )

    pipe = FlashAlayaPipeline(
        engine,
        control_modes=list(mode_cfg.control),
        use_memory=bool(mode_cfg.use_memory),
        action_cfg_scale=float(mode_cfg.action_cfg_scale),
        flex_attn=flex_attn,
        seed=args.seed,
        ttc=args.ttc,
        ttc_levels=tuple(int(x) for x in cfg.validation.ttc.levels),
        ttc_strength=float(cfg.validation.ttc.strength),
        ttc_ref_action=bool(cfg.validation.ttc.ref_action),
    )
    if is_rank0:
        ttc_str = (
            f"ON levels={','.join(map(str, cfg.validation.ttc.levels))} strength={cfg.validation.ttc.strength}"
            if args.ttc
            else "off"
        )
        print(
            f"[AlayaWorld] compile={args.compile} flex_attn={'on' if flex_attn else 'off'} "
            f"seed={args.seed} ttc={ttc_str}",
            flush=True,
        )

    cache = pipe.initialize_cache(
        video_pixels, caption, metadata, rounds=rounds, K=K, cond_end=cond_end, needed_latents=needed_latents
    )

    # End-of-clip skill: the caption is encoded once and reused for the whole
    # rollout, so a single prompt can't reliably time a one-off effect to the end.
    # Instead we encode a second "skill" caption and swap cache.context in-place for
    # the final --skill-sec seconds, so a summoned creature / energy burst erupts
    # right at the end with precise timing. Engine untouched (context is a plain,
    # per-step-read field of the cache).
    skill_prompt = args.skill_prompt or resolve_skill_prompt(args.input)
    skill_context = None
    skill_start = rounds
    if skill_prompt and args.skill_sec > 0:
        frames_per_chunk = K * int(cfg.sample.temporal_stride)
        skill_chunks = min(rounds, max(1, math.ceil(args.skill_sec * float(cfg.sample.fps) / frames_per_chunk)))
        skill_start = rounds - skill_chunks
        skill_context = engine.encode_caption(skill_prompt)
        if is_rank0:
            print(
                f"[AlayaWorld] end-skill: final {skill_chunks} chunk(s) "
                f"(~{skill_chunks * frames_per_chunk / float(cfg.sample.fps):.1f}s) switch context -> "
                f"{skill_prompt[:60]!r}"
                f"{'' if args.skill_keep_wrap else '  (+drop spatial wrap)'}",
                flush=True,
            )

    for i in range(rounds):
        if skill_context is not None and i == skill_start:
            cache.context = skill_context  # one-shot switch at skill onset (held to the end)
            # Drop the spatial-memory (forward-warp) conditioning so the summoned
            # effect is not suppressed by the old scene's re-projected geometry —
            # the caption swap alone can't overcome the geometric prior. History /
            # nearby context stay, so the scene's appearance still carries over; only
            # the hard geometric lock is released. These are the final chunk(s), so
            # skipping bank append/build for the rest is harmless (nothing reads it later).
            if not args.skill_keep_wrap:
                cache.spatial_bank = None
            if is_rank0:
                print(
                    f"[AlayaWorld] skill onset at chunk {i + 1}/{rounds}"
                    f"{' (wrap kept)' if args.skill_keep_wrap else ' (wrap dropped)'}",
                    flush=True,
                )
        t0 = time.perf_counter()
        pred = pipe.generate(i, cache)
        t1 = time.perf_counter()
        pipe.finalize(i, cache, pred)
        torch.cuda.synchronize()
        if is_rank0:
            print(
                f"[AlayaWorld] chunk {i + 1}/{rounds}: generate={t1 - t0:.2f}s finalize={time.perf_counter() - t1:.2f}s",
                flush=True,
            )

    if is_rank0:  # only rank0 decodes + saves
        free_generation_models(engine)
        t0 = time.perf_counter()
        frames = pipe.decode(cache)
        if cfg.validation.save_joystick:
            frames = apply_joystick_overlay(cfg, cache, frames)
        out_dir = Path(cfg.run.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(args.input).stem}_rounds-{rounds}.mp4"
        engine.write_video(out_path, frames, crf=args.video_crf)
        print(
            f"[AlayaWorld] decode {time.perf_counter() - t0:.1f}s frames={tuple(frames.shape)} -> saved {out_path}",
            flush=True,
        )

    if cp_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
