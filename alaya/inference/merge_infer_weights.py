"""聚合最终推理权重为单个 safetensors (base_transformer + vae + text-encoder + history_encoder 一体).

把 SFT base (transformer.pt) + DMD/GAN generator LoRA (lora.safetensors) + history_encoder.pt
合并成一份 self-contained 的 merged_infer.safetensors, 使 val 配置里 resume_checkpoint / dmd_resume /
history_encoder 全部留空即可直接用作 base_transformer 和 vae:

    paths:
      base_transformer: checkpoints/merged_infer.safetensors
      vae:              checkpoints/merged_infer.safetensors
      resume_checkpoint: ''
      dmd_resume: ''
      # history_encoder 无需配置, 已折叠进 merged 文件 (history_encoder.* 前缀)

文件内容 (见 alaya/model/loader.py 的三处消费 + engine.setup 的 history encoder):
  - transformer 权重: 取自 SFT transformer.pt (model 内部命名 blocks.*, 完整 state_dict),
        其中 480 个 LoRA target 线性层已 merge 进 generator LoRA delta;
  - vae.* :                     取自原始 LTX 文件 (load_vae 用);
  - text_embedding_projection.*, model.diffusion_model.video_embeddings_connector.* :
        取自原始 LTX 文件 (load_text_encoder 用);
  - history_encoder.* :         取自 SFT history_encoder.pt (engine.setup 抽取该子集加载);
  - safetensors metadata (config/model_version/...) : 取自原始 LTX 文件, 必须保留
        (loader 从 metadata 读 transformer/vae/text-encoder 架构 config)。

不包含 audio_/av_ca_/vocoder (推理不加载), 以及 action_adaln.pt (本 ckpt 里为空张量, 无效)。

Merge 公式 (见 alaya/model/lora.py: forward 为 out + x @ A.T @ B.T, scaling 已在 init 时
折进 A):  delta_W = (lora_B @ lora_A) * (alpha/rank);  merged_W = base_W + delta_W
"""

import argparse
import json
from pathlib import Path

import safetensors
import safetensors.torch as st
import torch

LORA_PREFIX = "diffusion_model.transformer_blocks."
HISTORY_ENCODER_PREFIX = "history_encoder."
KEEP_FROM_LTX = (
    "vae.",
    "text_embedding_projection.",
    "model.diffusion_model.video_embeddings_connector.",
)


# Placeholder paths — pass your own training outputs via the CLI flags
# (--transformer_pt / --lora / --history_encoder / --ltx / --output).
DEFAULT_TRANSFORMER_PT = "path/to/sft/transformer.pt"
DEFAULT_LORA = "path/to/dmd/lora.safetensors"
DEFAULT_LTX = "path/to/ltx-2.3-22b-dev.safetensors"
DEFAULT_OUTPUT = "checkpoints/merged_infer.safetensors"
# history_encoder.pt 折叠进 merged 文件 (history_encoder.* 前缀), 推理时无需单独加载。
DEFAULT_HISTORY_ENCODER = "path/to/history_encoder.pt"


def main():
    p = argparse.ArgumentParser(
        description="Merge SFT base + DMD LoRA + history_encoder into one inference safetensors. "
        "默认直接读取原始训练产出位置, 输出聚合权重到 checkpoints/merged_infer.safetensors。"
    )
    p.add_argument(
        "--transformer_pt",
        default=DEFAULT_TRANSFORMER_PT,
        help="SFT checkpoint transformer.pt (含完整 transformer + action adaln)",
    )
    p.add_argument("--lora", default=DEFAULT_LORA, help="DMD/GAN generator lora.safetensors")
    p.add_argument("--ltx", default=DEFAULT_LTX, help="原始 LTX safetensors (供 vae/text-encoder 键 + metadata)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 merged_infer.safetensors")
    p.add_argument(
        "--history_encoder",
        default=DEFAULT_HISTORY_ENCODER,
        help="SFT history_encoder.pt; 折叠进 merged 文件 (history_encoder.* 前缀, 设为空字符串可跳过)",
    )
    p.add_argument("--lora_alpha", type=float, default=128.0)
    p.add_argument("--lora_rank", type=float, default=128.0)
    args = p.parse_args()

    scaling = args.lora_alpha / args.lora_rank
    out_path = Path(args.output)

    # 1) 完整 transformer state_dict (model 内部命名)
    print(f"[Merge] loading transformer base = {args.transformer_pt}")
    out = torch.load(args.transformer_pt, map_location="cpu", weights_only=True)
    print(f"[Merge]   transformer keys = {len(out)}")

    # 2) merge generator LoRA delta
    print(f"[Merge] loading lora = {args.lora}  (alpha={args.lora_alpha} rank={args.lora_rank} scaling={scaling})")
    lora = st.load_file(args.lora, device="cpu")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora.items():
        assert k.startswith(LORA_PREFIX), f"unexpected lora key: {k}"
        body = k[len(LORA_PREFIX) :]
        if body.endswith(".lora_A.weight"):
            pairs.setdefault("blocks." + body[: -len(".lora_A.weight")], {})["A"] = v
        elif body.endswith(".lora_B.weight"):
            pairs.setdefault("blocks." + body[: -len(".lora_B.weight")], {})["B"] = v
        else:
            raise ValueError(f"unexpected lora key suffix: {k}")

    merged = 0
    for mk, AB in pairs.items():
        assert "A" in AB and "B" in AB, f"incomplete LoRA pair: {mk}"
        wk = mk + ".weight"
        assert wk in out, f"lora target weight missing in transformer: {wk}"
        delta = (AB["B"].float() @ AB["A"].float()) * scaling
        base_w = out[wk]
        assert base_w.shape == delta.shape, f"shape mismatch {wk}: {base_w.shape} vs {delta.shape}"
        out[wk] = (base_w.float() + delta).to(base_w.dtype)
        merged += 1
    print(f"[Merge]   merged LoRA layers = {merged}/{len(pairs)}")
    assert merged == len(pairs), "some LoRA pairs failed to merge"

    # 3) 追加 vae + text-encoder 键 + metadata (来自原始 LTX)
    print(f"[Merge] pulling vae/text-encoder keys + metadata from {args.ltx}")
    added = 0
    with safetensors.safe_open(args.ltx, framework="pt") as h:
        metadata = dict(h.metadata() or {})
        for k in h.keys():
            if k.startswith(KEEP_FROM_LTX):
                assert k not in out, f"unexpected collision: {k}"
                out[k] = h.get_tensor(k).contiguous()
                added += 1
    print(f"[Merge]   added keys from LTX = {added}  (metadata fields: {list(metadata.keys())})")
    assert "config" in metadata, "LTX metadata has no 'config' — loader would build wrong architecture"

    # sanity: vae 三类 + text proj + connector 都在
    def count(pred):
        return sum(1 for k in out if pred(k))

    print(f"[Merge]   vae.encoder/decoder/stats = {count(lambda k: k.startswith('vae.'))}")
    print(f"[Merge]   text_embedding_projection = {count(lambda k: k.startswith('text_embedding_projection.'))}")
    print(f"[Merge]   video_embeddings_connector = {count(lambda k: 'video_embeddings_connector' in k)}")

    # 4) history_encoder 折叠进 merged 文件 (前缀 history_encoder.), 推理时无需单独的
    #    history_encoder.pt: engine 从 merged_state 抽取该子集加载 (见 engine.setup)。
    if args.history_encoder:
        print(f"[Merge] folding history_encoder = {args.history_encoder}")
        he = torch.load(args.history_encoder, map_location="cpu", weights_only=True)
        if isinstance(he, dict) and "state_dict" in he and not any(
            isinstance(v, torch.Tensor) for v in list(he.values())[:3]
        ):
            he = he["state_dict"]
        folded = 0
        for k, v in he.items():
            assert isinstance(v, torch.Tensor), f"history_encoder non-tensor value: {k}"
            mk = HISTORY_ENCODER_PREFIX + k
            assert mk not in out, f"history_encoder key collision: {mk}"
            out[mk] = v.contiguous()
            folded += 1
        print(f"[Merge]   folded history_encoder tensors = {folded} (prefix {HISTORY_ENCODER_PREFIX!r})")

    # 5) save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Merge] saving {len(out)} tensors -> {out_path}")
    st.save_file(out, str(out_path), metadata=metadata)
    sz = out_path.stat().st_size / (1024**3)
    print(f"[Merge] DONE: {out_path}  ({sz:.2f} GB, {len(out)} tensors)")


if __name__ == "__main__":
    main()
