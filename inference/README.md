<p align="right">
  <kbd><b>English</b></kbd>
  <kbd><a href="README_zh.md">简体中文</a></kbd>
</p>

# Alaya World — Inference (Image-to-Video)

Official inference entry for **Alaya World**, an autoregressive interactive
world model built on LTX 2.3. Give it a **single first-frame image**, a
**camera/action trajectory**, and a **text prompt**; it rolls out a video that
follows the camera path, one chunk at a time.

> 1 chunk = 4 latent frames = 32 pixel frames ≈ **1.33s @ 24fps**.
> A ~1-minute clip ≈ **45 chunks**, which needs a camera trajectory of
> **≥ ~1450 frames**.

## Requirements

- A CUDA GPU. PyTorch **≥ 2.7.1** (the DiT uses `flex_attention`).
- Model weights under `./checkpoints/` (relative to the repo root), configured in
  `configs/infer.yaml` under `paths:`:

  ```
  checkpoints/
  ├── merged_infer.safetensors                    # DiT + VAE + text-encoder + history_encoder bundle
  ├── gemma-3-12b-it-qat-q4_0-unquantized/         # Gemma text encoder
  ├── Depth-Anything-3/                            # DA3 code repo (spatial-memory depth)
  ├── hf_cache/                                    # HF cache holding the DA3 weights
  └── taeltx2_3_wide.pth                           # optional tiny bank decoder (only with --bank-taehv), from github.com/madebyollin/taehv
  ```

  Point `paths:` elsewhere if your weights live in another location.

## Input layout (a "case")

```
<prefix>_image.<png|jpg|jpeg|webp|bmp>   first frame — seeds the history
<prefix>_camera.pt                       metadata dict: cam_c2w [F,4,4], intrinsic, ...
<prefix>_prompt.txt                      the text prompt
```

`--input` may point at the prefix or at any one of these files. The image is
auto-resized + center-cropped to the config resolution (default **544×960**)
and replicated to the trajectory length to seed the model's history window (the
model needs ~5.4s of history to start). Ready-to-run cases live under
[`playground/`](../playground).

## Run

One command (single GPU) — renders the bundled **case1** (~1 min):

```bash
bash inference/run.sh
```

Multi-GPU (Ulysses Context Parallel; e.g. 2 or 4 GPUs):

```bash
GPUS=4 bash inference/run.sh
```

The launcher just forwards to `python -m inference.run` (defaulting to
`--input playground/case1/case1`); call the module directly to run any case:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m inference.run --input playground/case1/case1 --seed 1234
```

Output: `outputs/<input>_rounds-N.mp4` (override the location with
`--output-dir`). By default the Move/Rotate joystick HUD is drawn on top; turn
it off with `--no-joystick`. The final few seconds cast a one-off skill effect
when the case ships a `<prefix>_skill.txt` (disable with `--skill-sec 0`).

## Common options

| Flag | Default | Meaning |
|------|---------|---------|
| `--input` | *(required)* | case prefix or any member file |
| `--cfg` | `configs/infer.yaml` | inference config; model paths live under `paths:` |
| `--output-dir` | *(config)* | where to save the mp4 |
| `--rounds` | `1000` | max autoregressive chunks; actual = `min(this, trajectory length)`. ~45 ≈ 1 min |
| `--seed` | `None` | fix per-chunk noise for reproducible runs |
| `--compile` | `reduce-overhead` | `torch.compile` mode for the DiT (`none` to disable) |
| `--no-flex-attn` | *(on)* | disable fused `flex_attention` |
| `--no-joystick` | *(config)* | do not draw the joystick HUD |
| `--ttc` | *(off)* | Pathwise Test-Time Correction — curbs appearance drift over long rollouts |
| `--video-crf` | `28` | h264 quality (18 near-lossless, 28 small) |
| `--skill-sec` | `4.0` | switch to the case's `_skill.txt` prompt for the final N seconds (one-off end effect); `0` disables |
| `--skill-prompt` | *(file)* | inline skill caption, overrides `<prefix>_skill.txt` |

Run `python -m inference.run --help` for the full list.

## Notes

- This CLI is a thin wrapper over the `alaya.inference` engine — it reuses the
  engine's rollout helpers and streaming pipeline unchanged. The same run is
  also available in the unified config form: `CONFIG_PATH=configs/infer.yaml
  bash scripts/finetune/train.sh` (knobs under `da3_infer:`).
- For long (~1 min) rollouts, `--ttc` re-anchors each chunk to the first frame
  to reduce appearance/style drift; tune its knobs under `validation.ttc` in the
  config.
