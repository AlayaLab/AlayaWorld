# AlayaWorld: Long-Horizon and Playable Video World Generation

<p align="center"><a href="https://alayalab.ai/"><b>Alaya Lab</b></a></p>

<p align="center">
  <a href="https://alaya-lab.github.io/AlayaWorld/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
  <a href="https://www.youtube.com/watch?v=n0jIEg7taTI"><img src="https://img.shields.io/badge/YouTube-Demo-red?logo=youtube&logoColor=white"></a>
  <a href="https://arxiv.org/abs/2607.06291"><img src="https://img.shields.io/badge/Intro-Report-red"></a>
  <a href="https://arxiv.org/abs/2607.18367"><img src="https://img.shields.io/badge/Full-Report-red"></a>
  <a href="https://github.com/AlayaLab/AlayaWorld"><img src="https://img.shields.io/badge/Code-Available-brightgreen?logo=github"></a>
  <a href="https://huggingface.co/AlayaLab/AlayaWorld"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HuggingFace-yellow"></a>
</p>

<p align="center">
  <img src="assets/fig1-AlayaWorld.png" width="100%">
</p>

> An interactive autoregressive world model with real-time camera control, prompt switching, and long-horizon memory consistency.

---

## 📰 News

- **[2026-07-21]** [Full Technical Report](https://arxiv.org/abs/2607.18367) released.
- **[2026-07-16]** Inference code released and pretrained weights available on 🤗 [Hugging Face](https://huggingface.co/AlayaLab/AlayaWorld). See [Quick Start](#-quick-start).
- **[2026-07-08]** Project page and [technical report](https://arxiv.org/abs/2607.06291) released.

## 🚀 Release Roadmap

- [x] Inference code
- [x] Pretrained weights — 🤗 [AlayaLab/AlayaWorld](https://huggingface.co/AlayaLab/AlayaWorld)
- [ ] Training code
- [ ] Training data (partial)

## ✨ Core Properties

AlayaWorld is built around four core properties — **interaction**, **consistency**, **stability**, and **runtime**.

### 🎮 Interaction
Two control channels: a rendered 3D cache with lightweight AdaLN camera modulation for grounded, trajectory-aware navigation, and chunk-level prompt switching to introduce new events mid-generation.

### 🧠 Consistency
Two forms of complementary memory: an explicit 3D cache reprojected to the queried view for spatial recall, plus a compressed frame-history embedding for temporal continuity, so revisited places stay recognizable.

### 🛡️ Stability
Long-horizon stability from training on drifted histories and an error bank that re-injects accumulated artifacts into both memory and target, preventing errors from compounding over minute-long rollouts.

### ⚡ Runtime
Real-time interaction via few-step DMD distillation and short temporal chunks, with prompt switching at chunk boundaries to minimize both visual and semantic latency.

## 🏃 Quick Start

Inference is image-to-video: give the model a **first-frame image**, a **camera trajectory**, and a **text prompt**, and it rolls out a video chunk by chunk along the camera path (1 chunk ≈ 1.33s @ 24fps; ~45 chunks ≈ 1 minute).

**1. Environment** — a CUDA GPU and PyTorch ≥ 2.6 (the DiT uses `flex_attention`).

**2. Weights** — the model runs on four pieces. Only `merged_infer.safetensors`
(AlayaWorld's own weights) is hosted on our 🤗 repo; the text encoder and the
depth model are third-party — get them from their original sources:

| Path under `checkpoints/` | Source |
|---|---|
| `merged_infer.safetensors` — DiT + VAE + text-encoder + history-encoder bundle | 🤗 [AlayaLab/AlayaWorld](https://huggingface.co/AlayaLab/AlayaWorld) |
| `gemma-3-12b-it-qat-q4_0-unquantized/` — Gemma text encoder | 🤗 [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) (gated — accept Google's license first) |
| `Depth-Anything-3/` — DA3 code repo | GitHub [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (see step 3) |
| `hf_cache/` — DA3 weights, in HF-cache layout | 🤗 [depth-anything/DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) (see step 3) |
| `taeltx2_3_wide.pth` — *optional* TAEHV bank decoder (`--bank-taehv`) | GitHub [madebyollin/taehv](https://github.com/madebyollin/taehv) |

From the repo root:

```bash
# AlayaWorld weights (this repo)
hf download AlayaLab/AlayaWorld merged_infer.safetensors --local-dir checkpoints

# Gemma text encoder (gated: log in + accept the license on its HF page first)
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
```

Target layout (`paths:` in `configs/infer.yaml` points here; repoint it if your
weights live elsewhere):

```
checkpoints/
├── merged_infer.safetensors
├── gemma-3-12b-it-qat-q4_0-unquantized/
├── Depth-Anything-3/          # step 3
├── hf_cache/                  # step 3
└── taeltx2_3_wide.pth         # optional
```

**3. Depth-Anything-3 (required)** — the spatial-memory branch depends on DA3, an
external repo (inference errors out without it):

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 checkpoints/Depth-Anything-3
pip install -e checkpoints/Depth-Anything-3      # run AFTER installing the torch stack (see requirements.txt)
```

Its weights (`depth-anything/DA3NESTED-GIANT-LARGE-1.1`) load from
`checkpoints/hf_cache/` — either let them download there on first run, or fetch
them ahead of time:

```bash
HF_HOME=checkpoints/hf_cache hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1
```

**4. Run a ready-made case** (cases live under [`playground/`](playground)).
The one-command launcher renders the bundled **case1** (~1 min):

```bash
# single GPU
bash inference/run.sh

# multi-GPU (Ulysses context parallel; e.g. 2 or 4 GPUs)
GPUS=4 bash inference/run.sh
```

`run.sh` just forwards to `python -m inference.run` (defaulting to
`--input playground/case1/case1`); call the module directly to run any case or
pass extra flags:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m inference.run --input playground/case1/case1 --seed 1234
```

The result mp4 is written under `outputs/`. See [`inference/README.md`](inference/README.md)
for the full option list.

**5. Use your own input** — a "case" is three files sharing a prefix:

```
<prefix>_image.png     first frame (seeds the history)
<prefix>_camera.pt     camera trajectory: cam_c2w [F,4,4] + intrinsics
<prefix>_prompt.txt    text prompt
```

Point `--input` at the prefix. For long (~1 min) rollouts, add `--ttc` to curb appearance drift. See [`inference/README.md`](inference/README.md) for the full option list.

## 👥 Team

- **Core Lead:** Kaipeng Zhang
- **Lead:** Chuanhao Li
- **Core Contributors:** Chuanhao Li, Kaipeng Zhang, Yifan Zhan, Yongtao Ge, Yuanyang Yin
- **Contributors:** Jiaming Tan, Kang He, Liaoyuan Fan, Mingliang Zhai, Ruicong Liu, Xiaojie Xu, Xuangeng Chu, Zhen Li, Zhengyuan Lin, Zhixiang Wang, Zian Meng, Zihui Gao

## 📬 Contact

For collaboration or business inquiries, contact **kaipeng.zhang@shanda.com**.

## 📝 Citation

If you find AlayaWorld useful for your research, please cite:

```bibtex
@article{team2026alayaworldintro,
  title={AlayaWorld: Long-Horizon and Playable Video World Generation},
  author={Team, AlayaWorld and Zhang, Kaipeng and Li, Chuanhao and Zhan, Yifan and Ge, Yongtao and Yin, Yuanyang and Tan, Jiaming and He, Kang and Fan, Liaoyuan and Liu, Ruicong and others},
  journal={arXiv preprint arXiv:2607.06291},
  year={2026}
}

@article{team2026alayaworldfull,
  title={AlayaWorld: Long-Horizon and Playable Video World Generation},
  author={Team, AlayaWorld and Zhang, Kaipeng and Li, Chuanhao and Zhan, Yifan and Ge, Yongtao and Yin, Yuanyang and Tan, Jiaming and He, Kang and Fan, Liaoyuan and Liu, Ruicong and Zhai, Mingliang and others},
  journal={arXiv preprint arXiv:2607.18367},
  year={2026}
}
```

## 📄 License

This project is based on LTX-2 by Lightricks Ltd. Portions of the original LTX-2
codebase (`flash_alaya/ltx2/`) have been modified by Alaya Lab for academic and research purposes only, and the released weights
(`merged_infer.safetensors`) are fine-tuned from LTX-2.3. Accordingly, this
project — code and weights — is released under the
[**LTX-2 Community License Agreement**](LICENSE). All original copyright, license,
patent, trademark, and attribution notices from LTX-2 are retained.

**For academic research and non-commercial use only.** For commercial use of
LTX-2 or its derivatives, contact Lightricks Ltd. (entities with ≥ $10M annual
revenue require a commercial license).

See [NOTICE](NOTICE) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for
full attribution. Third-party weights (Gemma-3 text encoder; Depth-Anything-3) are
**not redistributed** here — obtain them from their original sources under their
own licenses.
