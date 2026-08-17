from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Callable

import safetensors
import safetensors.torch
import torch
import torch.nn as nn

from fastvideo.ltx2_streaming_vae import StreamingVAEEncoder
from ltx2.modules.attention import AttentionFunction
from ltx2.modules.model_ltx_2_3 import LTX23Model
from ltx2.modules.rope import LTXRopeType
from ltx2.modules.vae import create_video_decoder, create_video_encoder

from alaya.config.schema import TrainConfig
from alaya.model.components import ModelComponents
from alaya.model.fsdp import maybe_enable_gradient_checkpointing
from alaya.model.lora import LoRAForwardManager


def _rank0() -> bool:
    """torchrun sets RANK per process (0 when unset). Print from rank 0 only to avoid duplicate lines."""
    import os
    return os.environ.get("RANK", "0") == "0"


def build_model_components(cfg: TrainConfig, device: torch.device, dtype: torch.dtype) -> ModelComponents:
    checkpoint_path = cfg.paths.effective_transformer
    if _rank0():
        print(f"[Paths] transformer checkpoint = {checkpoint_path}")
    transformer = load_transformer(checkpoint_path, cfg, device=device, dtype=dtype)
    if cfg.training.mode == "lora":
        for param in transformer.parameters():
            param.requires_grad_(False)
    elif cfg.training.mode == "sft":
        for param in transformer.parameters():
            param.requires_grad_(True)
    else:
        raise ValueError(f"unknown training.mode={cfg.training.mode!r}")

    if cfg.control.uses("action"):
        action_count = _set_action_adaln_trainable(transformer, enabled=True)
        if action_count == 0:
            raise RuntimeError("control uses 'action' but no action AdaLN parameters were created")
        print(f"[Control:action] trainable action_adaln tensors={action_count}")

    lora_manager = None
    if cfg.training.mode == "lora" and cfg.lora.enabled:
        lora_manager = LoRAForwardManager(trainable=cfg.lora.train)
        count = lora_manager.init_for_training(
            transformer,
            target_keywords=cfg.lora.targets,
            rank=cfg.lora.rank,
            alpha=cfg.lora.alpha,
            dtype=dtype,
            device=device,
        )
        hooks = lora_manager.register_hooks(transformer)
        lora_manager.enable()
        print(f"[LoRA] initialized={count} hooks={hooks} trainable={cfg.lora.train}")
    elif cfg.training.mode == "sft" and cfg.lora.enabled:
        print("[LoRA] ignored because training.mode=sft")

    maybe_enable_gradient_checkpointing(transformer, cfg.runtime.gradient_checkpointing)

    score_model = None
    critic_lora = None
    gan_discriminator = None
    if cfg.dmd.enabled:
        score_model, critic_lora = build_dmd_score_model(cfg, device=device, dtype=dtype)
        if cfg.dmd.is_use_gan:
            from alaya.dmd.discriminator import GanDiscriminator

            gan_discriminator = GanDiscriminator(
                score_model,
                hooks=list(cfg.dmd.gan_hooks),
                inner_dim=score_model.inner_dim,
                cond_map_dim=cfg.dmd.gan_cond_map_dim,
                dtype=dtype,
                device=device,
            )
            n_params = sum(p.numel() for p in gan_discriminator.parameters()) / 1e6
            print(
                f"[DMD-GAN] discriminator heads on blocks={cfg.dmd.gan_hooks} "
                f"inner_dim={score_model.inner_dim} cond_map_dim={cfg.dmd.gan_cond_map_dim} "
                f"params={n_params:.1f}M"
            )

    next_forcing_head = None
    if cfg.next_forcing.enabled:
        from alaya.dmd.next_forcing import NextForcingHead

        next_forcing_head = NextForcingHead(
            transformer,
            hook_layers=list(cfg.next_forcing.hook_layers),
            num_blocks=cfg.next_forcing.num_blocks,
            fuse_hidden_mult=cfg.next_forcing.fuse_hidden_mult,
            dtype=dtype,
            device=device,
        )
        nf_params = sum(p.numel() for p in next_forcing_head.parameters()) / 1e6
        print(
            f"[NextForcing] depth=1 MCP head: hooks={cfg.next_forcing.hook_layers} "
            f"num_blocks={cfg.next_forcing.num_blocks} loss_weight={cfg.next_forcing.loss_weight} "
            f"sigma_shift={cfg.next_forcing.sigma_shift} params={nf_params:.1f}M"
        )

    vae_encoder_raw, vae_decoder = load_vae(cfg.paths.vae, device=device, dtype=dtype)
    vae_encoder = StreamingVAEEncoder(vae_encoder_raw, device=device, dtype=dtype)
    text_encoder, encode_text = load_text_encoder(checkpoint_path, cfg.paths.gemma, device=device, dtype=dtype)

    return ModelComponents(
        transformer=transformer,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        text_encoder=text_encoder,
        encode_text=encode_text,
        lora_manager=lora_manager,
        score_model=score_model,
        critic_lora=critic_lora,
        gan_discriminator=gan_discriminator,
        next_forcing_head=next_forcing_head,
    )


def build_dmd_score_model(cfg: TrainConfig, device: torch.device, dtype: torch.dtype):
    """Build the frozen real score model plus trainable critic LoRA for DMD."""
    base = cfg.paths.real_score_model or cfg.paths.effective_transformer
    if not base:
        raise ValueError("dmd.enabled requires paths.real_score_model or an effective transformer path")
    print(f"[DMD] real/fake score base = {base}")
    score_model = load_transformer(base, cfg, device=device, dtype=dtype)
    score_model.requires_grad_(False)

    critic_lora = LoRAForwardManager(trainable=True)
    count = critic_lora.init_for_training(
        score_model,
        target_keywords=cfg.dmd.critic_lora_targets or cfg.lora.targets,
        rank=cfg.dmd.critic_lora_rank,
        alpha=cfg.dmd.critic_lora_alpha,
        dtype=dtype,
        device=device,
    )
    hooks = critic_lora.register_hooks(score_model)
    critic_lora.enable()
    maybe_enable_gradient_checkpointing(score_model, cfg.runtime.gradient_checkpointing)
    print(f"[DMD] critic LoRA initialized={count} hooks={hooks} rank={cfg.dmd.critic_lora_rank}")
    return score_model, critic_lora


def load_transformer(checkpoint_path: str, cfg: TrainConfig, device: torch.device, dtype: torch.dtype) -> LTX23Model:
    _configure_control_env(cfg)
    config = _read_transformer_config(checkpoint_path)
    config.update(_runtime_transformer_overrides(cfg))

    valid_params = set(inspect.signature(LTX23Model.__init__).parameters.keys())
    filtered = {key: value for key, value in config.items() if key in valid_params}
    model = LTX23Model(**filtered)

    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
        model_keys = {name for name, _ in model.named_parameters()}
        model_keys.update(name for name, _ in model.named_buffers())
        converted = convert_transformer_state_dict(state_dict, model_keys)
        missing, unexpected = model.load_state_dict(converted, strict=False)
        if _rank0():
            print(f"[Transformer] loaded={len(converted)} missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device=device, dtype=dtype)
    if _rank0():
        print(f"[Transformer] params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model


def _read_transformer_config(checkpoint_path: str) -> dict:
    if not checkpoint_path or not Path(checkpoint_path).exists() or not checkpoint_path.endswith(".safetensors"):
        return {}
    with safetensors.safe_open(checkpoint_path, framework="pt") as handle:
        metadata = handle.metadata() or {}
        return json.loads(metadata.get("config", "{}")).get("transformer", {})


def _runtime_transformer_overrides(cfg: TrainConfig) -> dict:
    attention_map = {
        "flash_attention_3": AttentionFunction.FLASH_ATTENTION_3,
        "xformers": AttentionFunction.XFORMERS,
        "pytorch": AttentionFunction.PYTORCH,
    }
    use_action = cfg.control.uses("action")
    return {
        "attention_type": attention_map.get(cfg.runtime.attention_type, AttentionFunction.FLASH_ATTENTION_3),
        "rope_type": LTXRopeType.SPLIT,
        "normalize_time_by_fps": cfg.runtime.norm_by_fps,
        "normalize_rope_positions": cfg.runtime.norm_by_max_frames,
        "positional_embedding_max_pos": [
            int(x.strip()) for x in cfg.runtime.positional_embedding_max_pos.split(",")
        ],
        "apply_gated_attention": True,
        "cross_attention_adaln": True,
        "caption_proj_before_connector": True,
        "enable_action_control": use_action,
        "compact_spatial_tokens": cfg.runtime.compact_spatial_tokens,
    }


def _configure_control_env(cfg: TrainConfig) -> None:
    use_action = cfg.control.uses("action")
    os.environ["LTX_USE_ACTION_CONTROL"] = "1" if use_action else "0"
    os.environ["LTX_ACTION_SCALE"] = cfg.control.action_scale
    os.environ["LTX_ACTION_FREQ_SCALE"] = str(cfg.control.action_freq_scale)
    os.environ["LTX_ACTION_FREQ_DIM_PER_AXIS"] = str(cfg.control.action_freq_dim_per_axis)


def _set_action_adaln_trainable(model: nn.Module, enabled: bool) -> int:
    count = 0
    for name, param in model.named_parameters():
        if "action_adaln_embedder" in name or "action_adaln_projection" in name:
            param.requires_grad_(enabled)
            count += 1
    return count


def convert_transformer_state_dict(state_dict: dict, model_keys: set[str]) -> dict:
    converted = {}
    skip_prefixes = (
        "audio_",
        "av_ca_",
        "_a2v_",
        "_v2a_",
        "vae.",
        "vocoder.",
        "text_embedding_projection.",
        "model.diffusion_model.video_embeddings_connector.",
        "model.diffusion_model.audio_",
        "model.diffusion_model.av_ca_",
    )
    for raw_key, value in state_dict.items():
        if any(raw_key.startswith(prefix) for prefix in skip_prefixes):
            continue
        candidates = []
        if raw_key.startswith("model.diffusion_model."):
            candidates.append(raw_key.removeprefix("model.diffusion_model.").replace("transformer_blocks.", "blocks."))
        cleaned = raw_key.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
        cleaned = cleaned.replace("transformer_blocks.", "blocks.")
        candidates.extend([cleaned, raw_key])
        for key in candidates:
            if key in model_keys:
                converted[key] = value
                break
    return converted


def load_vae(
    checkpoint_path: str, device: torch.device, dtype: torch.dtype, state_dict: dict | None = None
) -> tuple[nn.Module, nn.Module]:
    config = {}
    if checkpoint_path and Path(checkpoint_path).exists():
        with safetensors.safe_open(checkpoint_path, framework="pt") as handle:
            config = json.loads((handle.metadata() or {}).get("config", "{}"))

    encoder = create_video_encoder(config)
    decoder = create_video_decoder(config)

    # da3 inference: a preloaded merged state may be shared so the one-file
    # checkpoint is read once; None (vigeo paths) keeps the original behavior.
    if state_dict is None and checkpoint_path and Path(checkpoint_path).exists():
        state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
    if state_dict is not None:
        enc = {}
        dec = {}
        for key, value in state_dict.items():
            if key.startswith("vae.encoder."):
                enc[key.removeprefix("vae.encoder.")] = value
            elif key.startswith("vae.decoder."):
                dec[key.removeprefix("vae.decoder.")] = value
            elif key.startswith("vae.per_channel_statistics."):
                short = key.removeprefix("vae.")
                enc[short] = value
                dec[short] = value
        encoder.load_state_dict(enc, strict=True)
        decoder.load_state_dict(dec, strict=True)

    encoder.to(device=device, dtype=dtype).eval()
    decoder.to(device=device, dtype=dtype).eval()
    for module in (encoder, decoder):
        for param in module.parameters():
            param.requires_grad_(False)
    return encoder, decoder


def load_text_encoder(
    checkpoint_path: str,
    gemma_root: str,
    device: torch.device,
    dtype: torch.dtype,
    state_dict: dict | None = None,
) -> tuple[nn.Module, Callable]:
    from transformers import Gemma3ForConditionalGeneration
    from ltx2.modules.text_encoder import (
        AVGemmaTextEncoderModel,
        Embeddings1DConnector,
        GemmaFeaturesExtractorProjLinear,
        LTXVGemmaTokenizer,
    )

    with safetensors.safe_open(checkpoint_path, framework="pt") as handle:
        config = json.loads((handle.metadata() or {}).get("config", "{}"))
    tf_config = config.get("transformer", {})

    caption_proj_before_connector = tf_config.get("caption_proj_before_connector", True)
    if caption_proj_before_connector:
        video_inner_dim = tf_config.get("num_attention_heads", 32) * tf_config.get("attention_head_dim", 128)
        feature_extractor = GemmaFeaturesExtractorProjLinear(out_dim=video_inner_dim, bias=True, use_video_key=True)
    else:
        feature_extractor = GemmaFeaturesExtractorProjLinear()

    connector_head_dim = tf_config.get("connector_attention_head_dim", 128)
    connector_heads = tf_config.get("connector_num_attention_heads", 32)
    connector = Embeddings1DConnector(
        attention_head_dim=connector_head_dim,
        num_attention_heads=connector_heads,
        num_layers=tf_config.get("connector_num_layers", 8),
        positional_embedding_max_pos=tf_config.get("connector_positional_embedding_max_pos", [1]),
        rope_type=LTXRopeType(tf_config.get("rope_type", "interleaved")),
        apply_gated_attention=tf_config.get("connector_apply_gated_attention", True),
    )

    tokenizer = LTXVGemmaTokenizer(gemma_root)
    gemma = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_root,
        local_files_only=True,
        dtype=dtype,
        torch_dtype=dtype,
    ).to(device).eval()
    text_encoder = AVGemmaTextEncoderModel(
        feature_extractor,
        connector,
        None,
        tokenizer=tokenizer,
        model=gemma,
        dtype=dtype,
        use_v2_norm=caption_proj_before_connector,
        gemma_embedding_dim=3840,
    )

    if state_dict is None:  # da3: shared merged state when provided; else read here (vigeo path)
        state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
    fe = {
        key.removeprefix("text_embedding_projection."): value
        for key, value in state_dict.items()
        if key.startswith("text_embedding_projection.")
    }
    if fe:
        text_encoder.feature_extractor_linear.load_state_dict(fe, strict=False)
    ec = {
        key.replace("model.diffusion_model.video_embeddings_connector.", ""): value
        for key, value in state_dict.items()
        if "video_embeddings_connector" in key
    }
    if ec:
        text_encoder.embeddings_connector.load_state_dict(ec, strict=False)

    text_encoder.to(device).eval()
    for param in text_encoder.parameters():
        param.requires_grad_(False)

    def encode_text(encoder: nn.Module, prompts: list[str]):
        outputs = []
        with torch.no_grad():
            for prompt in prompts:
                outputs.append(encoder(prompt).video_encoding)
        return outputs

    return text_encoder, encode_text
