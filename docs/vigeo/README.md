<p align="right">
  <kbd><b>English</b></kbd>
  <kbd><a href="README_zh.md">简体中文</a></kbd>
</p>

# Alaya-World — training code for an autoregressive video world model

Config-driven training and inference code for an **LTX-2.3-based (~13B) video DiT** turned into an
autoregressive world model: given a first frame, a text prompt and a camera trajectory, it rolls out
video chunk by chunk while keeping long-range memory and 3D consistency.

What ships here:

- **four training stages**, each one YAML + one launcher — bidirectional pretrain, history pretrain,
  autoregressive SFT with geometry conditioning, and few-step distillation down to a 4-step student;
- **a cache prebuild stage** (stage0) for the text-embedding and whole-clip VAE-latent caches;
- **inference scripts** that take a single image + camera intrinsics/extrinsics + a prompt and write an mp4.

```bash
pip install -r requirements.txt

bash scripts/finetune/stage0_precache.sh                                   # build caches first
CONFIG_PATH=configs/stage1_pretrain_bidir.yaml bash scripts/finetune/train.sh
```

---

## 1. Setup

### 1.1 Python environment

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126   # match your CUDA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation                                   # FA2; needed by the attention path
```

### 1.2 Model weights

Three pieces: the LTX-2.3 base (from its own release — not redistributed here) plus the two
checkpoints published with this repository.

```bash
pip install -U "huggingface_hub[cli]"
export HF_HUB_ENABLE_HF_TRANSFER=1        # much faster for the 26GB transformer

# 1) LTX-2.3 base: transformer + VAE in one safetensors, plus the text encoder
#    obtain it from the LTX-2.3 release and place it as below

# 2) autoregressive teacher (stage2b): full fine-tuned transformer, 26GB
hf download AlayaLab/AlayaWorld-v1.1-stage2b --local-dir weights/alaya-world-ar

# 3) few-step student (stage3): LoRA on top of the teacher, 2.6GB
hf download AlayaLab/AlayaWorld-v1.1-stage3 --local-dir weights/alaya-world-dmd
```

What each repository contains and which config field consumes it:

| file | size | config field |
|---|---|---|
| `weights/ltx-2.3/ltx-2.3-22b-dev.safetensors` | ~50GB | `paths.base_transformer`, `paths.vae` (transformer and VAE live in one file) |
| `weights/ltx-2.3/google/gemma-3-12b-it-qat-q4_0-unquantized/` | text encoder dir | `paths.gemma` |
| `weights/alaya-world-ar/transformer.pt` | 26.2GB | `paths.resume_checkpoint` (point it at the directory) |
| `weights/alaya-world-ar/history_encoder.pt` | 34MB | `paths.history_encoder` |
| `weights/alaya-world-dmd/lora.safetensors` | 2.6GB | `paths.dmd_resume` (point it at the directory) |
| *(critic / GAN discriminator states)* | — | not part of the released checkpoint; only produced when you run stage3 training yourself |

Inference uses the teacher for both configs; `paths.dmd_resume` is what switches it from the 30-step
teacher to the 4-step student (see [Inference](#8-inference-single-image--camera--prompt--video)).

### 1.3 ViGeo (required for stage2b and stage3)

The spatial condition of the autoregressive stages calls an external ViGeo pointmap estimator — it is
not vendored here. Clone it and put its checkpoint where the configs expect:

```bash
git clone https://github.com/aigc3d/ViGeo third_party/ViGeo
pip install -r third_party/ViGeo/requirements.txt   # pulls in xformers - see the note below
hf download pkqbajng/ViGeo1.1 --local-dir third_party/ViGeo/checkpoints/ViGeo1.1   # checkpoint from https://huggingface.co/pkqbajng/ViGeo1.1
```

Config keys: `spatial_memory.vigeo_repo_path`, `spatial_memory.vigeo_checkpoint`,
with `spatial_memory.enabled: true` and `context_mode: vigeo_prefix_last_frame`.

> **ViGeo imports xformers, which cannot coexist with a locally built flash-attn-3.** xformers ships
> its own compiled flash-attn-3 (`xformers/flash_attn_3/_C.so`), and PyTorch allows only one
> `TORCH_LIBRARY` block per operator namespace. Loading both aborts the process at the C++ level
> (`Only a single TORCH_LIBRARY can be used to register the namespace flash_attn_3`) — it is not a
> catchable Python exception, so there is no fallback. Run every ViGeo/DA3 stage with
> `ALAYA_USE_FA3=0`; FA2 is unaffected.

Depth-Anything-3 (`spatial_memory.depth_backend: da3`) is an alternative depth backend and is
optional — clone it to `third_party/Depth-Anything-3` only if you want that variant. It pulls in
xformers through dinov2, so the same FA3 restriction applies.

### 1.4 Dataset

The dataset has two parts obtained from two places:

- **Annotations** (manifest + captions + camera poses, ~1.5GB packed) — released by us as the
  **gated** Hugging Face dataset [AlayaLab/AlayaWorld-v1.1-data](https://huggingface.co/datasets/AlayaLab/AlayaWorld-v1.1-data):
  open the page, accept the terms, then download with a token that has access.
- **Videos** — **not distributed by us.** Download the Sekai-Real-Walking source clips with the
  upstream tooling: [Lixsp11/sekai-codebase](https://github.com/Lixsp11/sekai-codebase)
  (follow its README to fetch the walking subset), and place them under
  `data/Video/sekai_real_walking/`. Budget ~0.6TB for all 18,208 clips.

```bash
hf auth login                              # or export HF_TOKEN=<your token>

# annotations: three files (manifest jsonl + two tarballs)
hf download AlayaLab/AlayaWorld-v1.1-data --repo-type dataset \
    --local-dir data/Annotation/sekai_real_hq

# unpack captions/poses into the layout the configs expect
cd data/Annotation/sekai_real_hq && tar -xzf caption.tar.gz && tar -xzf pose.tar.gz && cd -

# videos: via sekai-codebase (see its README), into data/Video/sekai_real_walking/
```

A partial video download is fine to train on — the loader skips clips whose file is missing.

After unpacking, the layout the configs expect is:

```
data/Annotation/sekai_real_hq/sekai_real_hq.jsonl      18208 lines, one clip each
data/Annotation/sekai_real_hq/caption/                 overall + segment captions
data/Annotation/sekai_real_hq/pose/*.npz               camera-to-world extrinsics (estimated)
data/Video/sekai_real_walking/<video-id>/*.mp4         clips, 3879 source videos (from sekai-codebase)
```

Point `paths.annotation_base_dir` at `data/Annotation` and `paths.video_base_dir` at `data/Video`.
The clip fields and the caption/pose schema are described in [Data format](#4-data-format).

Use of the annotations is bound to the terms you accept on the dataset page. **By using this data
you are deemed to have agreed to the data usage agreement of
[sekai-codebase](https://github.com/Lixsp11/sekai-codebase)**, from which the footage is derived.
The camera poses are our own estimates, not source annotations.

### 1.5 Check before you launch

```bash
DESCRIBE=1 CONFIG_PATH=configs/stage1_pretrain_bidir.yaml bash scripts/finetune/train.sh
```

This resolves the config, prints the summary and exits without touching a GPU — it fails loudly on a
missing weight, dataset or ViGeo path.

## 2. Requirements

| | |
|---|---|
| GPUs | the launcher defaults to 8 GPUs per node (`NPROC_PER_NODE`). The full fine-tune stages (stage1 / stage2b) need >=80GB per GPU and shard optimizer state with FSDP; the LoRA stages (stage2a / stage3) also run with `runtime.fsdp: false` |
| Python / CUDA | Python 3.10, CUDA 12.x |
| PyTorch | >= 2.7.1 (`requirements.txt` records lower bounds that are known to work, with no upper pins) |
| Attention | flash-attn 2 is enough. flash-attn-3 (Hopper) is optional, must be built locally, and cannot be combined with the ViGeo/DA3 stages — see [Setup](#1-setup) |
| Disk | model weights ~80GB (LTX-2.3 base + the released AR/DMD checkpoints); the dataset ~0.65TB; the whole-clip VAE latent cache another ~0.4TB at 544×960 |

`ltx2/` and `fastvideo/` are vendored third-party stacks — see `THIRD_PARTY.md` for provenance.

## 3. Repository layout for weights and data

Every external path in the shipped configs is **relative to the repository root**. Put the files
there or symlink them:

```
weights/ltx-2.3/                LTX-2.3 transformer .safetensors, VAE, and the text-encoder directory
third_party/ViGeo/              external ViGeo geometry estimator (+ checkpoints/ViGeo1.1)
third_party/Depth-Anything-3/   only needed for the depth-warp spatial variant
data/Video/                     training clips (mp4)
data/Annotation/                per-clip caption json + camera pose npz + the dataset jsonl
cache/                          text-embedding and VAE-latent caches (written by stage0)
outputs/, logs/                 run artifacts (git-ignored)
```

`ALAYA_DATASET_CACHE_DIR` overrides where the sample-list cache goes (default `.cache/dataset`;
point it at `/dev/shm/dataset_cache` for a RAM-backed cache).

## 4. Data format

The shipped configs train on a single source, `sekai_real_hq` (`data.sources: {sekai_real_hq: 1.0}`);
the dataset itself is not redistributed here. The loader
(`fastvideo/dataset/t2v_datasets.py`) expects three things per clip.

**One jsonl line per clip.** Relative paths join `paths.video_base_dir` / `paths.annotation_base_dir`;
absolute paths are used as-is.

```json
{"video": "sekai_real_hq/clip_0001.mp4",
 "prompt": "sekai_real_hq/clip_0001.json",
 "pose": "sekai_real_hq/clip_0001.npz",
 "num_frames": 1800,
 "valid_k8_starts": [0, 8, 16]}
```

`num_frames` lets the sampler pick windows without opening the video; `valid_k{K}_starts` is optional
and only read when `layout.k8_use_valid_starts` / `k4_use_valid_starts` is on.

**Caption json** — an overall caption plus optional timed segments. Training draws the whole-clip
caption with probability `data.overall_caption_prob` and a segment caption otherwise:

```json
{"overall":  {"short_prompt": "...", "full_prompt": "..."},
 "segments": [{"time_range_s": [0.0, 4.0], "full_prompt": "..."}]}
```

**Camera pose npz** — camera-to-world extrinsics, and intrinsics if you have them:

| key | shape | notes |
|---|---|---|
| `cam_c2w` (or `extrinsic`, `data`) | `[N, 4, 4]` | camera-to-world, one entry per video frame |
| `intrinsics` / `intrinsic` / `K` | `[3, 3]` or `[N, 3, 3]` | optional; a sibling intrinsics npz is picked up automatically, otherwise pixel-unit values are normalised by the frame size |

Extrinsics are normalised before use (`data.camera_norm_mode`, `camera_post_relic_scale`) so that
translations stay in a range that is safe for positional encodings in bf16.

## 5. Training stages

| config | stage | trainer / mode | mechanism | starts from |
|---|---|---|---|---|
| `configs/stage0_precache.yaml` | **cache prebuild (no training)** | `scripts/finetune/stage0_precache.sh` | whole-clip VAE latent cache (sharded over all GPUs, resumable) + text-embedding cache | LTX-2.3 VAE + text encoder |
| `configs/stage1_pretrain_bidir.yaml` | bidirectional pretrain | `RolloutTrainer` / sft | whole 20s clip denoised in one shot; in-clip clean-mask conditioning (i2v 0.7 / v2v 0.2 / t2v 0.1, condition frames get sigma=0 inside the model); variable-length training | LTX-2.3 base weights |
| `configs/stage2a_histpretrain.yaml` | history pretrain | `FrameQueryTrainer` / lora | masked-history reconstruction: non-Omega frames noised, Omega kept clean -> HistoryEncoder -> reconstruct Omega; trains the HistoryEncoder + LoRA (rank 256) | stage1 checkpoint |
| `configs/stage2b_arsft_vigeo.yaml` | autoregressive SFT | `RolloutTrainer` / sft | sink (remote) + history memory + nearby motion latents + **ViGeo** pointmap-prefix geometry + action AdaLN; anti-drift and next-forcing | stage1 transformer + stage2a HistoryEncoder |
| `configs/stage3_dmd_vigeo.yaml` | few-step distillation | `DmdTrainer` / lora | DMD (TTUR 1:10) + rCM consistency regularization + Self-Forcing++ self-rollout; yields a 4-step student | stage2b checkpoint (teacher and critic share it) |

The stages chain: **stage0 → stage1 → stage2a → stage2b → stage3**. Each later stage reads the
previous stage's output, so fill these fields in with your own run directories:

| config field | points at |
|---|---|
| `stage2a: paths.resume_checkpoint` | stage1 output checkpoint |
| `stage2b: paths.resume_checkpoint` + `paths.history_encoder` | stage2a checkpoint, LoRA merged back into the base (`scripts/tools/merge_lora_for_rollout.py`) |
| `stage3: paths.resume_checkpoint` + `paths.history_encoder` | stage2b checkpoint |
| `stage3: paths.dmd_resume` | stage3's own checkpoint, when resuming the student |

Training-time validation writes videos to `outputs/<run>/validation/step-XXXXXX/` at
`validation.interval`; `validation.before_train: true` emits a baseline before step 1.

## 6. Launcher knobs

`scripts/finetune/train.sh` reads `CONFIG_PATH` and dispatches on which `*.enabled` flag the config
sets (see `alaya/train.py`):

| var | effect |
|---|---|
| `VALIDATE_ONLY=1` | run validation only, no training |
| `DESCRIBE=1` | print the resolved config summary and exit |
| `LOG_FILTER=all` | keep full stdout (the default keeps only `[Train] step=` lines) |
| `NPROC_PER_NODE` / `MASTER_PORT` | override the torchrun topology |
| `ALAYA_USE_FA3=1` + `FA3_HOPPER=<path>` | use a locally built flash-attn-3 (Hopper); loss is bit-identical to FA2. **Mutually exclusive with the ViGeo / DA3 spatial path**, which pulls in xformers and its own bundled flash-attn-3 — use `ALAYA_USE_FA3=0` there (see [Setup](#1-setup)) |
| `LOG_ROOT` / `LOG_NAME` | override where the run log goes (default is derived from `run.log_dir` or the config name) |

`runtime.fsdp: false` is for the LoRA stages only — the SFT stages need sharding because the
optimizer state of a full fine-tune does not fit otherwise.

## 7. Caches

Two disk caches are shared across stages and restarts:

| cache | config key | behaviour |
|---|---|---|
| text embeddings | `runtime.text_embed_cache_dir` | lazy: a miss is encoded and written back during training. `runtime.precache_text_embeds: true` (stage0) front-loads every reachable prompt |
| whole-clip VAE latents | `runtime.vae_latent_cache_dir` | **read-only during training** — training never writes it, so it must be prebuilt by stage0 (`scripts/tools/precache_vae_latents.py`) |

Without the VAE prebuild, every step pays a fresh whole-window VAE encode; with it, the window tail
is sliced from cache and only the head is encoded fresh (roughly a 60% cut in VAE time on the
long-window stages).

**Reading the `[Perf]` line** (printed every 10 steps):

```
[Perf] last10: text_encode=0.39s vae_encode=5.86s text_cache_hit=10/10 vae_cache_hit=10/10 cache_size=10
```

`vae_cache_hit=0/0` — a zero denominator, meaning the cache was never queried — is **expected on the
short-window stages** (stage2b and stage3 use K=4). The cache can only serve latents beyond the first
17 of a window, because a causal VAE needs about 16 latents of context before a sliced value is
bit-identical to a fresh encode; a 4-latent window has no sliceable tail, so the lookup is skipped.
Nothing is lost — a 4-latent fresh encode is cheaper than a 61-latent one that hits the cache.

## 8. Inference: single image + camera + prompt → video

```bash
bash scripts/infer/generate_video.sh \
    --image first_frame.png \
    --prompt "a first-person walk down a misty forest trail" \
    --synth-frames 256 --forward 0.0049 --yaw 0.15 \
    --rounds 5
```

The script prepares the inputs (`scripts/infer/prepare_i2v_inputs.py`) and runs the model in
validation-only mode. Point `paths.resume_checkpoint` / `paths.history_encoder` at your stage2b
checkpoint, and `paths.dmd_resume` at stage3.

| input | how |
|---|---|
| first frame | `--image` (any resolution; resized to the config's 544×960) |
| prompt | `--prompt` |
| **extrinsics** | `--extrinsics my_c2w.npz` (`cam_c2w` `[N,4,4]`, camera-to-world) **or** synthesize a trajectory with `--synth-frames/--forward/--yaw/--pitch` |
| **intrinsics** | `--intrinsic fx fy cx cy` (normalized). When omitted, the geometry backend fits intrinsics itself instead of trusting the placeholder values |
| length | `--rounds N` — 4 latents (~1.3s of video) per round |

Two configs, same script:

| config | model | sampling |
|---|---|---|
| `configs/infer_i2v_camera.yaml` | stage3 4-step DMD student (`paths.dmd_resume`) | 4 steps, uniform, CFG 1.0 — fast |
| `configs/infer_i2v_camera_ar.yaml` | stage2b AR-SFT teacher (no student LoRA) | 30 steps, shift schedule, CFG 3.0 — slow, quality reference |

```bash
CONFIG_PATH=configs/infer_i2v_camera_ar.yaml \
    bash scripts/infer/generate_video.sh --image a.png --prompt "..." --rounds 5
```

## 9. Code layout

```
alaya/            first-party training code
  train.py        entry point + trainer dispatch
  config/         dataclass schema + YAML loader
  trainer/        RolloutTrainer / FrameQueryTrainer / DmdTrainer / dmd_self_rollout (SF++)
  memory/         HistoryEncoder, ViGeo geometry, DA3 depth, spatial cache
  dmd/            DMD losses, rCM consistency, next-forcing, discriminator
  model/          LTX loading, LoRA, FSDP wrapping
  data/           dataloaders, text-embedding and VAE-latent caches
configs/          stage0-stage3 training configs + the two inference configs
scripts/finetune/ train.sh launcher, stage0_precache.sh
scripts/infer/    single-clip inference entry point
scripts/tools/    VAE latent prebuild, LoRA merge for rollout
ltx2/, fastvideo/ vendored third-party stacks (see THIRD_PARTY.md)
```

## 10. Known behaviours

| symptom | explanation |
|---|---|
| `vae_cache_hit=0/0` on stage2b / stage3 | expected; see [Caches](#7-caches) |
| `RuntimeError: sample too short` retries during stage3 | the Self-Forcing++ window needs `81 + max_chunks*K*stride` frames; clips shorter than that are skipped and retried, which is normal dataset behaviour rather than an error |
| the process aborts with `Only a single TORCH_LIBRARY can be used to register the namespace flash_attn_3` | a locally built flash-attn-3 and the copy bundled inside xformers (pulled in by ViGeo, and by DA3 through dinov2) register the same torch operator namespace. It aborts in C++, so it cannot be caught and retried — set `ALAYA_USE_FA3=0` on every spatial stage |
| OOM on stage1 / stage2b with `runtime.fsdp: false` | the SFT stages need FSDP sharding for optimizer state |

## 11. License and third-party code

See `LICENSE` for this repository, and `THIRD_PARTY.md` for the provenance and licenses of the
vendored `ltx2/` and `fastvideo/` trees, the Helios-derived anti-drift and GAN recipes, and the
external estimators (ViGeo, Depth-Anything-3) that are loaded at runtime but not bundled here.
