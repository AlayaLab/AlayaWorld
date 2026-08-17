# -*- coding: utf-8 -*-
"""
Context Parallel (Ulysses-style) Implementation for Long Video Training

Ulysses-style context parallelism:
- partition the input tensor along the temporal (sequence) dimension across GPUs
- redistribute activations with all-to-all communication inside attention
- each device computes attention on its own sequence shard

Reference: https://arxiv.org/abs/2601.20540 Section 3.3.3
"""

import datetime
import torch
import torch.distributed as dist
from typing import Optional, Tuple
from dataclasses import dataclass

from fastvideo.utils.parallel_states import nccl_info

_CP_TIMEOUT = datetime.timedelta(hours=2)


@dataclass
class ContextParallelConfig:
    """Context-parallel configuration."""
    enabled: bool = False
    cp_size: int = 1  # number of GPUs in the context-parallel group
    cp_rank: int = 0  # rank of this GPU inside the group
    cp_group: Optional[dist.ProcessGroup] = None


_cp_config = ContextParallelConfig()


def initialize_context_parallel(cp_size: int = 1):
    """
    Initialize the context-parallel group.
    
    Args:
        cp_size: number of GPUs; 1 disables context parallelism.
    """
    global _cp_config
    
    if cp_size <= 1:
        _cp_config = ContextParallelConfig(enabled=False, cp_size=1, cp_rank=0, cp_group=None)
        return
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    assert world_size % cp_size == 0, f"world_size ({world_size}) must be divisible by cp_size ({cp_size})"
    
    num_cp_groups = world_size // cp_size
    for i in range(num_cp_groups):
        ranks = list(range(i * cp_size, (i + 1) * cp_size))
        group = dist.new_group(ranks, timeout=_CP_TIMEOUT)
        if rank in ranks:
            _cp_config = ContextParallelConfig(
                enabled=True,
                cp_size=cp_size,
                cp_rank=rank - i * cp_size,
                cp_group=group,
            )
    
    if rank == 0:
        print(f"[Context Parallel] Initialized with cp_size={cp_size}, num_groups={num_cp_groups}")


def warmup_context_parallel(hidden_dim: int = 4096, num_heads: int = 32,
                            seq_len: int = 1024, device: torch.device = None):
    """
    Warm up NCCL communication.

    Runs a dummy all-to-all before the first forward so NCCL finishes communicator
    initialization and topology setup; otherwise the lazy init during the first forward
    can time out on some ranks.

    Args:
        hidden_dim: model hidden dimension
        num_heads: number of attention heads
        seq_len: dummy sequence length
        device: GPU device
    """
    if not _cp_config.enabled:
        return

    if device is None:
        device = torch.cuda.current_device()

    rank = dist.get_rank()
    head_dim = hidden_dim // num_heads
    cp_size = _cp_config.cp_size
    group = _cp_config.cp_group

    if rank == 0:
        print(f"[CP Warmup] starting NCCL warmup: cp_size={cp_size}, "
              f"hidden={hidden_dim}, heads={num_heads}, seq={seq_len}")

    dummy = torch.zeros(1, seq_len // cp_size, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    _ = _all_to_all_single(dummy, scatter_dim=2, gather_dim=1, group=group)

    dummy2 = torch.zeros(1, seq_len, num_heads // cp_size, head_dim, device=device, dtype=torch.bfloat16)
    _ = _all_to_all_single(dummy2, scatter_dim=1, gather_dim=2, group=group)

    dummy3 = torch.zeros(seq_len // cp_size, device=device, dtype=torch.bfloat16)
    gathered = [torch.zeros_like(dummy3) for _ in range(cp_size)]
    dist.all_gather(gathered, dummy3, group=group)

    dist.barrier()

    del dummy, dummy2, dummy3, gathered
    torch.cuda.empty_cache()

    if rank == 0:
        print(f"[CP Warmup] NCCL warmup done")


def get_cp_config() -> ContextParallelConfig:
    """Return the context-parallel configuration."""
    return _cp_config


def destroy_context_parallel():
    """Destroy the context-parallel group."""
    global _cp_config
    _cp_config = ContextParallelConfig()


def _all_to_all_single(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    All-to-all communication used by Ulysses-style context parallelism.
    
    Scatters along scatter_dim and gathers along gather_dim.
    
    Args:
        input_: input tensor
        scatter_dim: dimension to scatter
        gather_dim: dimension to gather
        group: process group
    
    Returns:
        the redistributed tensor
    """
    world_size = dist.get_world_size(group)
    
    if world_size == 1:
        return input_
    
    input_list = [chunk.contiguous() for chunk in torch.chunk(input_, world_size, dim=scatter_dim)]

    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    
    dist.all_to_all(output_list, input_list, group=group)
    
    return torch.cat(output_list, dim=gather_dim)


class _SeqAllToAll(torch.autograd.Function):
    """
    Sequence-dimension all-to-all with autograd support.
    
    Used by Ulysses-style context parallelism.
    """
    
    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        return _all_to_all_single(input_, scatter_dim, gather_dim, group)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        grad_input = _all_to_all_single(
            grad_output, 
            ctx.gather_dim, 
            ctx.scatter_dim, 
            ctx.group
        )
        return grad_input, None, None, None


def seq_all_to_all(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """
    Sequence-dimension all-to-all.
    
    Args:
        input_: input tensor
        scatter_dim: dimension to scatter
        gather_dim: dimension to gather
    
    Returns:
        the redistributed tensor
    """
    if not _cp_config.enabled:
        return input_
    return _SeqAllToAll.apply(input_, scatter_dim, gather_dim, _cp_config.cp_group)


def scatter_sequence(input_: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Scatter the sequence across all context-parallel GPUs.
    
    Args:
        input_: [B, S, ...] or [B, ..., S, ...]
        dim: sequence dimension
    
    Returns:
        the scattered tensor [B, S//cp_size, ...]
    """
    if not _cp_config.enabled:
        return input_
    
    seq_len = input_.shape[dim]
    assert seq_len % _cp_config.cp_size == 0, \
        f"Sequence length ({seq_len}) must be divisible by cp_size ({_cp_config.cp_size})"
    
    chunks = torch.chunk(input_, _cp_config.cp_size, dim=dim)
    return chunks[_cp_config.cp_rank].contiguous()


def gather_sequence(input_: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Gather the sequence from all context-parallel GPUs.
    
    Args:
        input_: this GPU's sequence shard
        dim: sequence dimension
    
    Returns:
        the gathered full sequence
    """
    if not _cp_config.enabled:
        return input_
    
    # All-gather
    world_size = _cp_config.cp_size
    gathered = [torch.empty_like(input_) for _ in range(world_size)]
    dist.all_gather(gathered, input_.contiguous(), group=_cp_config.cp_group)
    
    return torch.cat(gathered, dim=dim)


class _GatherForward(torch.autograd.Function):
    """Gather on forward, scatter on backward (for loss computation)."""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int) -> torch.Tensor:
        ctx.dim = dim
        return gather_sequence(input_, dim)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return scatter_sequence(grad_output, ctx.dim), None


def gather_for_loss(input_: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Gather for loss computation; gradients are scattered automatically on backward."""
    if not _cp_config.enabled:
        return input_
    return _GatherForward.apply(input_, dim)


def apply_ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    heads: int,
    attention_fn,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Attention with Ulysses-style context parallelism.
    
    1. All-to-all: scatter sequence dim, gather head dim
    2. run standard attention (each GPU handles a subset of heads over the full sequence)
    3. All-to-all: scatter head dim, gather sequence dim
    
    Args:
        q: Query [B, S, H*D]
        k: Key [B, S, H*D]  
        v: Value [B, S, H*D]
        heads: number of attention heads
        attention_fn: attention function
        mask: optional attention mask
    
    Returns:
        attention output [B, S, H*D]
    """
    if not _cp_config.enabled:
        return attention_fn(q, k, v, heads, mask)
    
    B, S, HD = q.shape
    H = heads
    D = HD // H
    cp_size = _cp_config.cp_size
    
    assert H % cp_size == 0, f"Number of heads ({H}) must be divisible by cp_size ({cp_size})"
    
    # 1. Reshape to expose head dimension: [B, S, H, D]
    q = q.view(B, S, H, D)
    k = k.view(B, S, H, D)
    v = v.view(B, S, H, D)
    
    # 2. All-to-all: scatter heads (dim=2), gather sequence (dim=1)
    # [B, S_local, H, D] -> [B, S_full, H//cp_size, D]
    q = seq_all_to_all(q, scatter_dim=2, gather_dim=1)
    k = seq_all_to_all(k, scatter_dim=2, gather_dim=1)
    v = seq_all_to_all(v, scatter_dim=2, gather_dim=1)
    
    # 3. Reshape back for attention: [B, S*cp_size, (H//cp_size)*D]
    _, S_full, H_local, _ = q.shape
    q = q.view(B, S_full, H_local * D)
    k = k.view(B, S_full, H_local * D)
    v = v.view(B, S_full, H_local * D)
    
    # 4. Execute attention with local heads
    if mask is not None:
        mask = gather_sequence(mask, dim=-1) if mask.shape[-1] == S else mask
    
    out = attention_fn(q, k, v, H_local, mask)
    
    # 5. Reshape for all-to-all back: [B, S_full, H_local, D]
    out = out.view(B, S_full, H_local, D)
    
    # 6. All-to-all: scatter sequence (dim=1), gather heads (dim=2)
    # [B, S_full, H//cp_size, D] -> [B, S_local, H, D]
    out = seq_all_to_all(out, scatter_dim=1, gather_dim=2)
    
    # 7. Reshape to original format: [B, S, H*D]
    out = out.view(B, S, HD)
    
    return out


def pad_to_cp_divisible(tensor: torch.Tensor, dim: int = 1) -> Tuple[torch.Tensor, int]:
    """
    Pad a tensor so the dimension is divisible by cp_size.
    
    Args:
        tensor: input tensor
        dim: dimension to pad
    
    Returns:
        (padded_tensor, original_length)
    """
    if not _cp_config.enabled:
        return tensor, tensor.shape[dim]
    
    original_length = tensor.shape[dim]
    cp_size = _cp_config.cp_size
    
    if original_length % cp_size == 0:
        return tensor, original_length
    
    pad_length = cp_size - (original_length % cp_size)
    
    pad_spec = [0] * (2 * tensor.ndim)
    pad_idx = 2 * (tensor.ndim - 1 - dim)
    pad_spec[pad_idx + 1] = pad_length  # pad at the end of this dimension
    
    padded = torch.nn.functional.pad(tensor, pad_spec, mode='constant', value=0)
    return padded, original_length


def unpad_from_cp(tensor: torch.Tensor, original_length: int, dim: int = 1) -> torch.Tensor:
    """
    Remove context-parallel padding.
    
    Args:
        tensor: padded tensor
        original_length: original length
        dim: padded dimension
    
    Returns:
        the unpadded tensor
    """
    if not _cp_config.enabled or tensor.shape[dim] == original_length:
        return tensor
    
    return tensor.narrow(dim, 0, original_length)


# =============================================================================
# =============================================================================

def prepare_cp_inputs(
    video_latent: torch.Tensor,
    audio_latent: Optional[torch.Tensor] = None,
    video_positions: Optional[torch.Tensor] = None,
    audio_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], dict]:
    """
    Prepare inputs for context parallelism.
    
    Shards the latents and positional encodings across GPUs.
    
    Args:
        video_latent: [B, T, ...] patchified video latent
        audio_latent: [B, T, ...] patchified audio latent
        video_positions: [B, 1, T, D] video positional encoding
        audio_positions: [B, 1, T, D] audio positional encoding
    
    Returns:
        the sharded (video_latent, audio_latent, video_positions, audio_positions, metadata)
    """
    metadata = {
        'video_original_length': video_latent.shape[1] if video_latent is not None else 0,
        'audio_original_length': audio_latent.shape[1] if audio_latent is not None else 0,
    }
    
    if not _cp_config.enabled:
        return video_latent, audio_latent, video_positions, audio_positions, metadata
    
    if video_latent is not None:
        video_latent = scatter_sequence(video_latent, dim=1)
    
    if audio_latent is not None:
        audio_latent = scatter_sequence(audio_latent, dim=1)
    
    if video_positions is not None:
        video_positions = scatter_sequence(video_positions, dim=2)
    
    if audio_positions is not None:
        audio_positions = scatter_sequence(audio_positions, dim=2)
    
    return video_latent, audio_latent, video_positions, audio_positions, metadata


def gather_cp_outputs(
    video_output: torch.Tensor,
    audio_output: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Gather context-parallel outputs.
    
    Args:
        video_output: video output shard
        audio_output: audio output shard
    
    Returns:
        the gathered (video_output, audio_output)
    """
    if not _cp_config.enabled:
        return video_output, audio_output
    
    video_output = gather_sequence(video_output, dim=1)
    if audio_output is not None:
        audio_output = gather_sequence(audio_output, dim=1)
    
    return video_output, audio_output


def get_cp_world_size() -> int:
    """Return the context-parallel world size."""
    return _cp_config.cp_size if _cp_config.enabled else 1


def get_cp_rank() -> int:
    """Return the context-parallel rank."""
    return _cp_config.cp_rank if _cp_config.enabled else 0


def is_cp_enabled() -> bool:
    """Return True when context parallelism is enabled."""
    return _cp_config.enabled


def compute_cp_divisible_frames(frames: int, cp_size: int, temporal_stride: int = 8) -> int:
    """
    Compute a frame count divisible by both cp_size and the temporal stride.
    
    The VAE requires (frames - 1) % 8 == 0 and context parallelism requires
    latent_frames % cp_size == 0.
    
    Args:
        frames: target frame count
        cp_size: CP size
        temporal_stride: VAE temporal stride
    
    Returns:
        the adjusted frame count
    """
    # LTX2 VAE: latent_frames = (frames - 1) // 8 + 1
    frames = ((frames - 1) // temporal_stride) * temporal_stride + 1
    
    latent_frames = (frames - 1) // temporal_stride + 1
    if latent_frames % cp_size != 0:
        latent_frames = ((latent_frames // cp_size) + 1) * cp_size
        frames = (latent_frames - 1) * temporal_stride + 1
    
    return frames
