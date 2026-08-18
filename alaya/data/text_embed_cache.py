"""On-disk text-embedding cache plus an optional full prebuild before training.

Design:
- key = sha1 of the final prompt string (after the source prefix and camera-variant handling);
  file {dir}/{h[:2]}/{h}.pt holding {"prompt": str, "ctx": bf16 CPU tensor [L,4096]}.
- during training: RAM LRU -> disk -> text encoder (a miss is written back, so it resumes).
- precache_text_embeds(): enumerate every reachable prompt at startup, mirroring the dataset's
  caption sampling space (segment and overall fields, the two <camera> variants, caption prefixes,
  the negative prompt), shard across ranks and encode in batches with a progress bar.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

import torch
import torch.distributed as dist


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def cache_path(cache_dir: str, prompt: str) -> str:
    h = _sha1(prompt)
    return os.path.join(cache_dir, h[:2], f"{h}.pt")


def disk_get(cache_dir: str, prompt: str) -> torch.Tensor | None:
    p = cache_path(cache_dir, prompt)
    if not os.path.exists(p):
        return None
    try:
        d = torch.load(p, map_location="cpu", weights_only=True)
        return d["ctx"]
    except Exception:
        return None  # treat a corrupt file as a miss (it will be rewritten)


def disk_put(cache_dir: str, prompt: str, ctx: torch.Tensor) -> None:
    p = cache_path(cache_dir, prompt)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f".tmp{os.getpid()}"
    torch.save({"prompt": prompt, "ctx": ctx.detach().to("cpu", torch.bfloat16)}, tmp)
    os.replace(tmp, p)  # atomic replace, safe across processes


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_CAM_KEEP = re.compile(r"<camera\b[^>]*>(.*?)</camera>", re.IGNORECASE | re.DOTALL)
_CAM_DROP = re.compile(r"<camera\b[^>]*>.*?</camera>", re.IGNORECASE | re.DOTALL)
_WS = re.compile(r"\s+")


def _camera_variants(text: str) -> list[str]:
    """The two reachable <camera> variants: keep the description without the tag, or drop it entirely."""
    if not isinstance(text, str) or "<camera" not in text.lower():
        return [text]
    keep = _WS.sub(" ", _CAM_KEEP.sub(r"\1", text)).strip()
    drop = _WS.sub(" ", _CAM_DROP.sub(" ", text)).strip()
    return [keep, drop] if keep != drop else [keep]


def _clip_prompts(caption_json: dict, seg_field: str, ov_field: str) -> set[str]:
    """Every reachable caption string of one caption json (without the prefix)."""
    out: set[str] = set()
    ov = caption_json.get("overall")
    if isinstance(ov, dict):
        for f in (ov_field, "description"):
            v = ov.get(f)
            if v:
                out.update(_camera_variants(str(v)))
    for key in ("merged_segments", "segments"):
        for seg in caption_json.get(key) or []:
            if not isinstance(seg, dict):
                continue
            for f in (seg_field, "short_prompt"):
                v = seg.get(f)
                if v:
                    out.update(_camera_variants(str(v)))
    for f in ("overall_caption", "caption", "text"):
        v = caption_json.get(f)
        if v:
            out.update(_camera_variants(str(v)))
    return out


def enumerate_all_prompts(cfg, dataset) -> list[str]:
    """Walk every dataset sample and enumerate the full reachable prompt set (including prefixes)."""
    from fastvideo.dataset.t2v_datasets import MultiSourceVideoDataset

    prompts: set[str] = set()
    neg = getattr(cfg.validation, "negative_prompt", None)
    if neg:
        prompts.add(str(neg))
    seen_caption_paths: set[str] = set()
    for sample in dataset.samples:
        _video, caption_path, _pose, source_name, _vid = sample
        if caption_path is None:
            continue
        src_cfg = MultiSourceVideoDataset.SOURCE_CONFIGS.get(source_name, {})
        prefix = src_cfg.get("caption_prefix", "") or ""
        if caption_path.startswith("__INLINE__:"):
            for v in _camera_variants(caption_path[len("__INLINE__:"):]):
                prompts.add(prefix + v)
            continue
        if caption_path in seen_caption_paths or not os.path.exists(caption_path):
            continue
        seen_caption_paths.add(caption_path)
        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        seg_field = src_cfg.get("segment_caption_field", "full_prompt")
        ov_field = src_cfg.get("overall_caption_field", "short_prompt")
        for v in _clip_prompts(data, seg_field, ov_field):
            prompts.add(prefix + v)
    return sorted(prompts)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_batch(encoder, texts: list[str]) -> list[torch.Tensor]:
    """Batch through the encoder with right padding; returns each video_encoding [L,4096]."""
    pairs = [encoder.tokenizer.tokenize_with_weights(t)["gemma"] for t in texts]
    max_len = max(len(p) for p in pairs)
    dev = encoder.model.device
    ids = torch.zeros(len(pairs), max_len, dtype=torch.long, device=dev)
    mask = torch.zeros(len(pairs), max_len, dtype=torch.long, device=dev)
    for i, p in enumerate(pairs):
        ids[i, : len(p)] = torch.tensor([t[0] for t in p], device=dev)
        mask[i, : len(p)] = torch.tensor([t[1] for t in p], device=dev)
    outputs = encoder.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    projected = encoder._run_feature_extractor(outputs.hidden_states, mask, "left")
    video_enc, _audio, _m = encoder._run_connectors(projected, mask)
    return [video_enc[i] for i in range(len(texts))]


@torch.no_grad()
def _encode_single(components, prompt: str) -> torch.Tensor:
    output = components.encode_text(components.text_encoder, [prompt])
    ctx = output[0][0] if isinstance(output, list) and output[0].dim() == 3 else output[0]
    return ctx


def _batch_matches_single(components, sample_prompts: list[str], atol: float = 5e-3) -> bool:
    """Self-check the batch output against single-sample output (bf16 tolerance); fall back on mismatch."""
    try:
        batch = _encode_batch(components.text_encoder, sample_prompts)
        for p, b in zip(sample_prompts, batch):
            s = _encode_single(components, p)
            if b.shape != s.shape or not torch.allclose(
                b.float().cpu(), s.float().cpu(), atol=atol, rtol=1e-2
            ):
                return False
        return True
    except Exception:
        return False


def precache_text_embeds(cfg, components, dist_state, dataset, *, batch_size: int = 8) -> None:
    """Encode every reachable prompt into the disk cache before training (sharded, resumable)."""
    cache_dir = cfg.runtime.text_embed_cache_dir
    assert cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    prompts = enumerate_all_prompts(cfg, dataset)
    missing = [p for p in prompts if not os.path.exists(cache_path(cache_dir, p))]
    if dist_state.is_main:
        print(
            f"[TextCache] {len(prompts)} prompts in total, {len(prompts) - len(missing)} already cached, "
            f"{len(missing)} to encode (dir={cache_dir})",
            flush=True,
        )
    if not missing:
        if dist.is_initialized():
            dist.barrier()
        return

    world = max(1, dist_state.world_size)
    shard = missing[dist_state.rank::world]

    def _tok_len(p: str) -> int:
        return len(components.text_encoder.tokenizer.tokenize_with_weights(p)["gemma"])

    groups: dict[int, list[str]] = {}
    for p in shard:
        groups.setdefault(_tok_len(p), []).append(p)
    same_len_sample = next((g[:3] for g in groups.values() if len(g) >= 2), None)
    use_batch = batch_size > 1 and same_len_sample is not None and _batch_matches_single(components, same_len_sample)
    if dist_state.is_main:
        print(
            f"[TextCache] {len(groups)} equal-length groups; batch self-check "
            + ("passed, batch=" + str(batch_size) if use_batch else "falling back to single-prompt encoding"),
            flush=True,
        )

    chunks: list[list[str]] = []
    if use_batch:
        for g in groups.values():
            for i in range(0, len(g), batch_size):
                chunks.append(g[i : i + batch_size])
    else:
        chunks = [[p] for p in shard]

    t0 = time.time()
    done = 0
    total = len(shard)
    report_every = max(1, total // 50)  # report about every 2%
    for chunk in chunks:
        try:
            ctxs = _encode_batch(components.text_encoder, chunk) if len(chunk) > 1 else [
                _encode_single(components, chunk[0])
            ]
        except Exception:
            ctxs = [_encode_single(components, p) for p in chunk]  # fall back per prompt if the batch fails
        for p, c in zip(chunk, ctxs):
            disk_put(cache_dir, p, c)
        done += len(chunk)
        if done % report_every < len(chunk) or done == total:
            el = time.time() - t0
            ips = done / max(el, 1e-6)
            eta = (total - done) / max(ips, 1e-6)
            bar_n = int(30 * done / total)
            print(
                f"[TextCache] rank{dist_state.rank} |{'█' * bar_n}{'.' * (30 - bar_n)}| "
                f"{done}/{total} ({100 * done / total:.1f}%) {ips:.1f} it/s ETA {eta / 60:.1f}min",
                flush=True,
            )
    if dist.is_initialized():
        dist.barrier()
    if dist_state.is_main:
        print(f"[TextCache] prebuild done in {(time.time() - t0) / 60:.1f}min", flush=True)
