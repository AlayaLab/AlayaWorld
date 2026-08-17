# Third-party code

This repository vendors two directories that originate from third-party projects and were
substantially refactored/extended in-house. **Before publishing, confirm the license terms and
add the required notices/attribution for both.**

| dir | origin | notes |
|---|---|---|
| `ltx2/` | LTX-2.3 video DiT (Lightricks LTX-Video line), refactored WAN-style in-house | gated attention, cross-attn AdaLN, camera-control AdaLN, patchifier/rope/scheduler |
| `fastvideo/` | FastVideo (hao-ai-lab) subset | dataset pipeline (`dataset/t2v_datasets.py`), streaming VAE, rollout error-bank / drift-simulator |

| `fastvideo/rollout/drift_simulator.py`, `alaya/dmd/*` | ports of the anti-drift / GAN-hook recipes from [PKU-YuanGroup/Helios](https://github.com/PKU-YuanGroup/Helios) | the code is a reimplementation against this stack, but the algorithm and defaults follow Helios; keep the attribution |

`alaya/` is first-party training code (stage1 / stage2a / stage2b-vigeo / stage3-vigeo).

Also note: the ViGeo geometry estimator and Depth-Anything-3 are used as **external checkpoints**
loaded at runtime (paths are config keys, see README); their weights/licenses are not part of this repo.

## Datasets referenced by the shipped configs

The configs point at **Sekai-Real-Walking-HQ** (`data.sources: {sekai_real_hq: 1.0}`), released as a gated Hugging Face dataset alongside this repository (see README). Its footage derives from [sekai-codebase](https://github.com/Lixsp11/sekai-codebase): **by using the data you are deemed to have agreed to that project's data usage agreement**, in addition to the terms accepted on the dataset page. The camera poses shipped with it are our own estimates, not source annotations.

`alaya/data/wbench.py` is a loader for an external navigation benchmark; no benchmark data, prompts or annotations are bundled in this repository.
