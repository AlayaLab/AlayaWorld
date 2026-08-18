# Third-Party Code Attribution

This project is based on LTX-2 (Lightricks) and is released, as a whole, under the
**LTX-2 Community License Agreement** (see LICENSE). It also uses code and model
weights from the additional open-source projects listed below; all copyright
notices are preserved in the source files, and third-party components remain under
their original licenses.

---

## 1. LTX-Video / LTX-2 (Diffusion backbone, VAE, text-encoder)

**Directory:** `ltx2/`
**Source:** Lightricks
**Repository:** https://github.com/Lightricks/LTX-2 · https://huggingface.co/Lightricks/LTX-2.3
**License (code and model):** **LTX-2 Community License Agreement** (dated 2026-01-05) — https://github.com/Lightricks/LTX-2/blob/main/LICENSE . This is **not** Apache 2.0. Free for academic use and for commercial use by entities under $10M ARR; entities with ≥ $10M ARR need a paid commercial license (ltxv-licensing@lightricks.com). **Any Derivative of LTX-2 must be distributed exclusively under this same Agreement**, retaining copyright notices, shipping the full license + use restrictions, and marking modified files. Includes use-based restrictions (Attachment A).
**Usage:** Our DiT and streaming pipeline in `ltx2/` are refactored/adapted from LTX-2 (derivative code). Our released weights `merged_infer.safetensors` are **fine-tuned from the LTX-2.3-22B base**, and the VAE / text-encoder tensors in that bundle are taken from the original LTX-2.3 weights — i.e. both the `ltx2/` code and the released weights are **derivatives of LTX-2.3** and are governed by the LTX-2 Community License Agreement, **not** this repo's Apache 2.0.

---

## 2. Wan / Wan2.1 (VAE and architecture style)

**Directory:** `ltx2/` (VAE and "WAN-style" module structure; see `ltx2/utils/ltx2_streaming_vae.py`, `ltx2/configs/`)
**Source:** Wan-Video (Alibaba)
**Repository:** https://github.com/Wan-Video/Wan2.1
**License:** Apache 2.0
**Usage:** The streaming VAE design and several module conventions follow Wan (the VAE is based on the Wan2.1 VAE). No GPL/strong-copyleft code is introduced.

---

## 3. TAEHV — Tiny AutoEncoder (optional bank decoder)

**File:** `alaya/inference/taehv.py` (vendored)
**Source:** madebyollin
**Repository:** https://github.com/madebyollin/taehv
**License:** MIT
**Usage:** Optional tiny latent decoder (`--bank-taehv`), `taeltx2_3_wide` variant matching LTX-2.3 latents. Attribution header is preserved in the source file.

---

## 4. Depth-Anything-3 (spatial-memory depth)

**Directory:** `checkpoints/Depth-Anything-3/` (external code repo, installed at runtime; **not** bundled in this repository)
**Source:** ByteDance-Seed
**Repository:** https://github.com/ByteDance-Seed/Depth-Anything-3
**Source Code License:** Apache 2.0
**Model License:** see the DA3 model card (`depth-anything/DA3NESTED-GIANT-LARGE-1.1`)
**Usage:** Provides the depth estimation used by the spatial-memory (3D cache) branch. Both the DA3 code and its weights are obtained by the user from the official sources; neither is redistributed here.

---

## 5. Gemma-3 (text encoder)

**Source:** Google
**Repository / Weights:** https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
**License:** Gemma Terms of Use (https://ai.google.dev/gemma/terms) — restricted, not an OSI-approved open-source license
**Usage:** Used at runtime as the text encoder, loaded from a user-downloaded checkpoint. Gemma weights are **not redistributed** by this project; users must download them from Google/Hugging Face and accept the Gemma Terms of Use.

---

## 6. Frameworks (pip dependencies, not code-copied)

| Component | Source | License |
|-----------|--------|---------|
| diffusers, transformers, huggingface-hub | Hugging Face | Apache 2.0 |
| PyTorch (torch/torchvision/torchaudio) | PyTorch | BSD-3-Clause |
| xformers | Meta | BSD-3-Clause |
| NumPy, SciPy | community | BSD-3-Clause |
| OpenCV (opencv-python) | OpenCV | Apache 2.0 |
| einops | community | MIT |
| safetensors, peft | Hugging Face | Apache 2.0 |
| bitsandbytes | community | MIT |

---

## License Compatibility

This project is released as a whole under the **LTX-2 Community License Agreement**.
The third-party components below are compatible with being used/included under it.

| Component | License | Note |
|-----------|---------|------|
| LTX-2 (source code — basis of `ltx2/`) | LTX-2 Community License Agreement | this project's governing license |
| LTX-2.3 (model weights — base of our weights) | LTX-2 Community License Agreement | our weights are a derivative, shipped under the same license |
| Wan / Wan2.1 (source code) | Apache 2.0 | ✓ permissive |
| TAEHV | MIT | ✓ permissive |
| Depth-Anything-3 (source code) | Apache 2.0 | ✓ permissive |
| Depth-Anything-3 (model weights) | DA3 model license | not redistributed |
| Gemma-3 (model weights) | Gemma Terms of Use | not redistributed |
| diffusers / transformers / HF libs | Apache 2.0 | ✓ |
| NumPy / SciPy / PyTorch / xformers | BSD-3-Clause | ✓ |
| OpenCV | Apache 2.0 | ✓ |
| einops / bitsandbytes | MIT | ✓ |

Model weights are subject to their providers' separate license terms. Both the
`ltx2/` code and the released `merged_infer.safetensors` weights are
derivatives of LTX-2.3 and are governed by the **LTX-2 Community License
Agreement**. Gemma-3 and Depth-Anything-3 weights are not redistributed here.

---

## Legal Notice

- This project is released, as a whole, under the **LTX-2 Community License Agreement** (see LICENSE); it is for academic research and non-commercial use only.
- All original LTX-2 copyright, license, patent, trademark, and attribution notices are retained; modifications by Alaya Lab are documented in file headers.
- Other third-party code (Wan, TAEHV, Depth-Anything-3, framework libraries) remains under its original permissive license.
- **Model weights may be subject to additional license terms from their respective providers.** Gemma-3 and Depth-Anything-3 weights are not redistributed by this project.

For questions, open an issue in the project repository.
