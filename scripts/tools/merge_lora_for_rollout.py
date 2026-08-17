"""
Merge the LoRA delta of a history-pretrain checkpoint into the base transformer,
producing a merged base that stage2b (rollout) training can consume directly.

Usage:
    python scripts/tools/merge_lora_for_rollout.py \
        --ckpt_dir <history_pretrain checkpoint-XXXX> \
        --base_transformer <clean base safetensors> \
        --output <merged_dir>

Inputs:
    ckpt_dir/lora.safetensors           (480 LoRA pairs, rank=128)
    ckpt_dir/history_encoder.pt         (HistoryEncoder state; copied as-is, not merged)
    base_transformer (safetensors)      (clean base transformer)

Outputs:
    output/diffusion_pytorch_model.safetensors   (base + LoRA delta merged)
    output/history_encoder.pt                     (copied from ckpt_dir)

Merge formula:
    scaling = lora_alpha / lora_rank
    delta_W = (lora_B @ lora_A) * scaling     [out, rank] @ [rank, in] = [out, in]
    merged_W = base_W + delta_W

Key mapping:
    LoRA: diffusion_model.transformer_blocks.X.YY.lora_{A,B}.weight
    Base: blocks.X.YY.weight
    -> strip "diffusion_model.transformer_blocks." and replace with "blocks."
"""
import argparse
import shutil
from pathlib import Path

import safetensors.torch as st
import torch


def merge_lora(base_path: Path, lora_path: Path, alpha: float, rank: int,
               out_path: Path, dtype: str = 'bf16') -> int:
    """Merge LoRA into the base weights; returns the number of merged layers."""
    scaling = alpha / rank
    print(f"[Merge] base = {base_path}")
    print(f"[Merge] lora = {lora_path}")
    print(f"[Merge] alpha={alpha}, rank={rank}, scaling={scaling}")

    base_sd = st.load_file(str(base_path), device='cpu')
    lora_sd = st.load_file(str(lora_path), device='cpu')
    print(f"[Merge] base keys: {len(base_sd)}, lora keys: {len(lora_sd)}")

    # Parse LoRA A/B pairs
    LORA_PREFIX = 'diffusion_model.transformer_blocks.'
    BASE_PREFIX = 'blocks.'
    lora_pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora_sd.items():
        assert k.startswith(LORA_PREFIX), f"unexpected lora key: {k}"
        body = k[len(LORA_PREFIX):]   # e.g. "0.attn1.to_q.lora_A.weight"
        if body.endswith('.lora_A.weight'):
            module_key = BASE_PREFIX + body[:-len('.lora_A.weight')]
            lora_pairs.setdefault(module_key, {})['A'] = v
        elif body.endswith('.lora_B.weight'):
            module_key = BASE_PREFIX + body[:-len('.lora_B.weight')]
            lora_pairs.setdefault(module_key, {})['B'] = v
        else:
            raise ValueError(f"unexpected lora key suffix: {k}")
    print(f"[Merge] LoRA pairs: {len(lora_pairs)}")

    # Merge: clone base_sd, then add the delta layer by layer
    out_sd = {k: v.clone() for k, v in base_sd.items()}
    merged = 0
    skipped = []
    for module_key, AB in lora_pairs.items():
        if 'A' not in AB or 'B' not in AB:
            print(f"[Merge] WARN: incomplete LoRA pair for {module_key}, skip")
            continue
        A = AB['A']
        B = AB['B']
        # delta_W = (B @ A) * scaling, shape [out, in]
        delta = (B.float() @ A.float()) * scaling

        base_key = module_key + '.weight'
        if base_key not in out_sd:
            skipped.append(base_key)
            continue
        base_w = out_sd[base_key]
        assert base_w.shape == delta.shape, (
            f"shape mismatch {base_key}: base={base_w.shape} vs delta={delta.shape}"
        )
        out_sd[base_key] = (base_w.float() + delta).to(base_w.dtype)
        merged += 1

    print(f"[Merge] merged {merged}/{len(lora_pairs)} layers")
    if skipped:
        print(f"[Merge] WARN: {len(skipped)} layers skipped (base key not found), first 5:")
        for k in skipped[:5]:
            print(f"    {k}")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Merge] saving to {out_path} ...")
    st.save_file(out_sd, str(out_path))
    sz_gb = out_path.stat().st_size / (1024**3)
    print(f"[Merge] saved: {out_path}  ({sz_gb:.2f} GB)")
    return merged


def is_merge_complete(out_dir: Path, expect_he: bool = True) -> bool:
    """Return True if the output directory already holds a finished merge."""
    transformer_path = out_dir / 'diffusion_pytorch_model.safetensors'
    he_path = out_dir / 'history_encoder.pt'
    if not transformer_path.exists():
        return False
    # Sanity check: the merged file must be multiple GB, not empty
    if transformer_path.stat().st_size < 1024 * 1024 * 1024:  # < 1GB means the merge did not finish
        return False
    if expect_he and not he_path.exists():
        return False
    return True


def main():
    p = argparse.ArgumentParser(description='Merge LoRA delta into base transformer for stage-2 rollout training')
    p.add_argument('--ckpt_dir', required=True, type=str,
                   help='history-pretrain checkpoint directory with lora.safetensors + history_encoder.pt')
    p.add_argument('--base_transformer', required=True, type=str,
                   help='path to the clean base transformer safetensors')
    p.add_argument('--lora_alpha', type=float, default=128.0)
    p.add_argument('--lora_rank', type=int, default=128)
    p.add_argument('--output', required=True, type=str,
                   help='output directory for the merged base')
    p.add_argument('--copy_history_encoder', type=lambda x: x.lower() == 'true', default=True,
                   help='also copy history_encoder.pt into the output directory')
    p.add_argument('--force', action='store_true',
                   help='force a re-merge even if the output already exists')
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    base_path = Path(args.base_transformer)
    lora_path = ckpt_dir / 'lora.safetensors'
    he_src = ckpt_dir / 'history_encoder.pt'
    out_dir = Path(args.output)
    out_path = out_dir / 'diffusion_pytorch_model.safetensors'

    # Idempotency: skip if the merge is already done
    if not args.force and is_merge_complete(out_dir, expect_he=args.copy_history_encoder):
        print(f"[Merge] ✓ already merged at {out_dir}, skip (use --force to re-merge)")
        sz = out_path.stat().st_size / (1024**3)
        print(f"[Merge]   transformer: {out_path}  ({sz:.2f} GB)")
        if (out_dir / 'history_encoder.pt').exists():
            print(f"[Merge]   he:          {out_dir / 'history_encoder.pt'}")
        return

    assert lora_path.exists(), f"lora.safetensors not found: {lora_path}"
    assert base_path.exists(), f"base transformer not found: {base_path}"

    # Merge
    n_merged = merge_lora(
        base_path=base_path,
        lora_path=lora_path,
        alpha=args.lora_alpha,
        rank=args.lora_rank,
        out_path=out_path,
    )

    # Copy the HistoryEncoder state
    if args.copy_history_encoder and he_src.exists():
        he_dst = out_dir / 'history_encoder.pt'
        shutil.copy2(str(he_src), str(he_dst))
        print(f"[Merge] history_encoder.pt copied to {he_dst}")
    elif args.copy_history_encoder:
        print(f"[Merge] WARN: history_encoder.pt not found in {ckpt_dir}, skipping the copy")

    print(f"\n[Merge] DONE. merged {n_merged} layers.")
    print(f"[Merge] output dir: {out_dir}")
    print(f"  - diffusion_pytorch_model.safetensors  (base + LoRA merged)")
    print(f"  - history_encoder.pt                    (stage-1 HE)")


if __name__ == '__main__':
    main()
