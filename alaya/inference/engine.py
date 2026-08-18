"""Self-contained inference engine for FlashAlaya.

Loads everything inference needs — DiT (meta-init, weights streamed directly to
GPU from the merged one-file checkpoint), VAE encoder/decoder, Gemma text
encoder, history encoder, and (eagerly) the DA3 depth model — without touching
alaya.trainer. Expects an inference config such as configs/infer.yaml
(single merged safetensors; resume_checkpoint empty; lora merged into weights).

Library deps only: alaya.model.loader / alaya.memory / alaya.config,
ltx2.modules. The streaming VAE encoder is vendored at
ltx2/utils/ltx2_streaming_vae.py (an LTX2 adaptation; the alaya training
path still uses independently).
"""
from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Callable

import safetensors.torch
import torch

import alaya.model.loader as L
from alaya.memory.builder import build_history_encoder
from alaya.utils.dtype import resolve_dtype
from alaya.inference.spatial import SpatialMemory
from ltx2.modules.model_ltx_2_3 import LTX23Model


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def fast_load_transformer(checkpoint_path: str, cfg, device, dtype, state_dict=None) -> LTX23Model:
    """meta-device build (skips the ~70s random init) + assign=True weight load.
    Requires a checkpoint that covers every model param/buffer (the merged one-file
    weights do: missing=0). state_dict: a preloaded merged state to filter from
    (shared across transformer/vae/text encoder so the file is read once); when
    None the transformer subset is read directly to `device`."""
    L._configure_control_env(cfg)
    config = L._read_transformer_config(checkpoint_path)
    config.update(L._runtime_transformer_overrides(cfg))
    valid = set(inspect.signature(LTX23Model.__init__).parameters.keys())
    filtered = {k: v for k, v in config.items() if k in valid}

    with torch.device("meta"):
        model = LTX23Model(**filtered)

    if state_dict is None and not (checkpoint_path and Path(checkpoint_path).exists()):
        raise RuntimeError(f"fast loader needs an existing checkpoint, got {checkpoint_path!r}")
    state = state_dict if state_dict is not None else safetensors.torch.load_file(checkpoint_path, device=str(device))
    model_keys = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    converted = L.convert_transformer_state_dict(state, model_keys)
    missing, unexpected = model.load_state_dict(converted, strict=False, assign=True)
    leftover = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if leftover:
        raise RuntimeError(
            f"fast loader: {len(leftover)} tensors stayed on meta (e.g. {leftover[:4]}); "
            "checkpoint does not cover the full model"
        )
    model = model.to(device=device, dtype=dtype)  # no-op move if already on `device`
    model.eval()
    print(
        f"[FlashAlaya:DiT] loaded={len(converted)} missing={len(missing)} unexpected={len(unexpected)} "
        f"(meta-init -> {device})",
        flush=True,
    )
    return model


class InferenceEngine:
    """Owns all models + basic encode/decode ops. No trainer, no optimizer."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # 跟随当前设备: 单进程默认 cuda:0; torchrun CP 下由 run.py 先 set_device(LOCAL_RANK)
        self.device = (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        self.dtype = resolve_dtype(cfg.runtime.dtype)
        self.transformer = None
        self.vae_encoder = None
        self.vae_decoder = None
        self.text_encoder = None
        self.encode_text: Callable | None = None
        self.history_encoder = None
        self.spatial: SpatialMemory | None = None
        self.load_times: dict[str, float] = {}

    # ----------------------------------------------------------------- setup
    def setup(
        self,
        *,
        eager_da3: bool = True,
        compile_aux: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        compile_aux: torch.compile the VAE decoder (mode='default'). Measured
            verdict (2026-06-10): NO steady-state gain on the big LTX decoder
            (single large-kernel calls; cuDNN already saturates) and ~10s extra
            one-time compile for the final decode shape. Default OFF; kept as a
            switch for future small/streaming decoders (taehv), where per-call
            overhead actually dominates. DA3 must stay eager (4.7x slower when
            compiled — dynamic Python internals).
        """
        cfg = self.cfg

        def phase(name):
            _sync()
            t0 = time.perf_counter()

            class _T:
                def __enter__(_s):
                    return _s

                def __exit__(_s, *a):
                    _sync()
                    self.load_times[name] = time.perf_counter() - t0

            return _T()

        ckpt = cfg.paths.effective_transformer
        vae_ckpt = cfg.paths.effective_vae
        print(f"[FlashAlaya:Engine] transformer checkpoint = {ckpt}", flush=True)

        # Read the merged one-file weights ONCE (to CPU) and share across DiT / VAE
        # / text encoder — each filters its own keys. Freed after the loads. The VAE
        # only shares when it lives in the same file (the common merged case).
        merged_state = safetensors.torch.load_file(ckpt, device="cpu") if Path(ckpt).exists() else None
        vae_state = merged_state if vae_ckpt == ckpt else None

        with phase("DiT (merged, meta-init)"):
            self.transformer = fast_load_transformer(ckpt, cfg, self.device, self.dtype, state_dict=merged_state)

        with phase("VAE"):
            from ltx2.utils.ltx2_streaming_vae import StreamingVAEEncoder

            vae_encoder_raw, self.vae_decoder = L.load_vae(
                vae_ckpt, device=self.device, dtype=self.dtype, state_dict=vae_state
            )
            self.vae_encoder = StreamingVAEEncoder(vae_encoder_raw, device=self.device, dtype=self.dtype)
            if compile_aux:
                # lossless kernel fusion; mode='default' (no cudagraphs) so the
                # stale_bank background thread can never collide with a capture.
                # Must happen BEFORE SpatialMemory grabs its decoder reference.
                self.vae_decoder = torch.compile(self.vae_decoder, mode="default", dynamic=False)

        with phase("Gemma text encoder"):
            self.text_encoder, self.encode_text = L.load_text_encoder(
                ckpt, cfg.paths.gemma, device=self.device, dtype=self.dtype, state_dict=merged_state
            )

        # history encoder weights are folded into the merged one-file checkpoint
        # under the `history_encoder.` prefix (see tools/merge_infer_weights.py).
        # Pull that subset out of the shared buffer before it is freed.
        HE_PREFIX = "history_encoder."
        he_state = (
            {k[len(HE_PREFIX):]: v for k, v in merged_state.items() if k.startswith(HE_PREFIX)}
            if merged_state is not None
            else None
        ) or None

        del merged_state, vae_state  # free the shared CPU buffer (~one merged file)

        with phase("history encoder"):
            if cfg.layout.history_latent_frames > 0:
                if he_state is None:
                    raise RuntimeError(
                        f"history encoder weights ({HE_PREFIX}*) not found in {ckpt}; "
                        "re-run tools/merge_infer_weights.py to fold them into the merged checkpoint"
                    )
                patchify_proj = self.transformer.patchify_proj
                self.history_encoder = build_history_encoder(
                    cfg.memory,
                    in_channels=patchify_proj.in_features,
                    out_channels=patchify_proj.out_features,
                    device=self.device,
                    dtype=self.dtype,
                    state_dict=he_state,
                )
                if cfg.memory.use_lr_branch:
                    self.history_encoder.setup_lr_proj_from_patchify(patchify_proj)
                self.history_encoder.eval()

        self.spatial = SpatialMemory(
            spatial_cfg=cfg.spatial_memory,
            sample_cfg=cfg.sample,
            sink_latent_frames=int(cfg.layout.sink_latent_frames),
            vae_encoder=self.vae_encoder,
            vae_decoder=self.vae_decoder,
            vae_chunk_size=int(cfg.runtime.vae_chunk_size),
            device=self.device,
            dtype=self.dtype,
            da3_repo=cfg.paths.da3_repo,
            da3_model=cfg.paths.da3_model,
            da3_cache=cfg.paths.da3_cache,
        )
        if eager_da3:
            with phase("DA3 depth (eager)"):
                self.spatial.ensure_da3()

        # NOTE: DA3 is deliberately NOT compiled. Measured verdict (2026-06-10):
        # its forward is full of dynamic Python (nested-giant dispatch, camera
        # branches) — dynamo produced 84-104s compiles per input shape and a
        # 4.7x SLOWER steady state (0.46s -> 2.19s). Keep it eager.

        if verbose:
            total = sum(self.load_times.values())
            peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            print("\n================ FlashAlaya Engine LOAD TIME ================", flush=True)
            for k, v in self.load_times.items():
                print(f"  {k:<28}: {v:6.1f}s", flush=True)
            print(f"  {'TOTAL':<28}: {total:6.1f}s   peak VRAM = {peak:.1f} GB", flush=True)
            print("==============================================================", flush=True)

    # ------------------------------------------------------------- basic ops
    @torch.no_grad()
    def encode_video(self, video_pixels: torch.Tensor, *, needed_latents: int) -> torch.Tensor:
        video_pixels = video_pixels.to(device=self.device, dtype=self.dtype)
        if video_pixels.dim() == 5:
            if video_pixels.shape[0] != 1:
                raise ValueError("FlashAlaya supports batch_size=1")
            video_pixels = video_pixels[0]
        if video_pixels.dim() != 4:
            raise ValueError(f"expected [F,C,H,W] or [B,F,C,H,W], got {tuple(video_pixels.shape)}")
        if video_pixels.shape[0] != 3:
            video_pixels = video_pixels.permute(1, 0, 2, 3).contiguous()
        stride = self.cfg.sample.temporal_stride
        needed_pixels = (needed_latents - 1) * stride + 1
        use_pixels = min(video_pixels.shape[1], needed_pixels)
        use_pixels = 1 + stride * ((use_pixels - 1) // stride)
        video_pixels = video_pixels[:, :use_pixels]
        latent = self.vae_encoder.encode(
            video_pixels.unsqueeze(0), chunk_size=self.cfg.runtime.vae_chunk_size, verbose=False
        )
        return latent.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def encode_caption(self, caption: str) -> torch.Tensor:
        output = self.encode_text(self.text_encoder, [caption])
        context = output[0][0] if isinstance(output, list) and output[0].dim() == 3 else output[0]
        return context.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def decode_latent_to_video_frames(self, latent: torch.Tensor) -> torch.Tensor:
        """latent [B,C,T,H,W] -> uint8 frames [T,H,W,C], via SEAMLESS overlap-tiling.

        Each core chunk [s:e) is decoded with `vae_decode_overlap_latents` neighbor
        latents on each side as context, then only the core's output frames are
        kept. The non-causal LTX decoder's cross-chunk dependency is supplied by
        that context, so the concatenation is seamless and frame-exact with a whole
        decode (exact when overlap >= the decoder's temporal receptive field ~6
        latents; memory-bounded vs whole decode's ~60GB). Set decode_chunk >= total
        latents for a single-tile whole decode. Ported from the trainer's decoder."""
        decode_chunk = self.cfg.runtime.vae_decode_chunk_latents
        chunk_latents = max(1, int(decode_chunk if decode_chunk is not None else self.cfg.runtime.vae_chunk_size))
        # overlap >= 1 is required for frame-complete tiling (a sub-decode's first
        # latent only yields 1 frame, so the core must have a context latent before
        # it); >= ~6 for byte-exactness. (overlap is irrelevant for a single tile.)
        overlap = max(1, int(getattr(self.cfg.runtime, "vae_decode_overlap_latents", 6) or 6))
        total_latents = int(latent.shape[2])
        r = int(self.cfg.sample.temporal_stride)
        frames = []

        # latent->pixel temporal map (8x upsample, first latent special):
        # latent0 -> 1 frame; latent j>=1 -> r frames at global [r(j-1)+1, rj].
        # A sub-decode [a:b) treats latent a as its "first" (1 frame, local 0);
        # latent a+k (k>=1) -> local [r(k-1)+1, rk].
        for s in range(0, total_latents, chunk_latents):
            e = min(total_latents, s + chunk_latents)
            a = max(0, s - overlap)
            b = min(total_latents, e + overlap)
            ctx = latent[:, :, a:b].to(device=self.device, dtype=self.dtype)
            pixel = self.vae_decoder(ctx)
            pixel = (pixel * 0.5 + 0.5).clamp(0, 1)
            tile = pixel.squeeze(0).permute(1, 2, 3, 0).contiguous()  # [F_tile,H,W,C]
            k_s = s - a
            lo = 0 if s == 0 else r * (k_s - 1) + 1   # core's first output frame within the tile
            hi = r * (e - 1 - a) + 1                   # core's last output frame + 1
            frames.append((tile[lo:hi] * 255.0).to(torch.uint8).cpu())
            del ctx, pixel, tile
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(frames, dim=0)

    def write_video(self, path: Path, frames: torch.Tensor, *, crf: int = 18) -> None:
        """crf: h264 quality (18 ~ visually lossless, 28 ~ small preview, +6 ≈ half size).
        Always saved at full resolution."""
        from torchvision.io import write_video

        path.parent.mkdir(parents=True, exist_ok=True)
        write_video(str(path), frames, fps=int(self.cfg.sample.fps), options={"crf": str(int(crf)), "preset": "veryfast"})
