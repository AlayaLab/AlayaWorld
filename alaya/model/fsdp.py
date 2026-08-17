from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist

from alaya.config.schema import RuntimeConfig


def maybe_enable_gradient_checkpointing(model: nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


def maybe_wrap_fsdp(model: nn.Module, cfg: RuntimeConfig, device: torch.device) -> nn.Module:
    if not cfg.fsdp or not dist.is_initialized() or dist.get_world_size() == 1:
        return model

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from functools import partial
    from ltx2.modules.model_ltx_2_3 import LTX23AttentionBlock

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LTX23AttentionBlock},
    )
    return FSDP(
        model,
        device_id=device if device.type == "cuda" else None,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        use_orig_params=True,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        limit_all_gathers=True,
        sync_module_states=True,
    )
