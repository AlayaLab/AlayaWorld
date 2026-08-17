# Copyright 2024-2025 LTX-2 Refactored (WAN-style)
"""
LTX-2 Diffusion Model following WAN style with Audio-Video support.
All core components in one file as per WAN convention.
"""
import math
import os
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from typing import Optional, Tuple
from ltx2.modules.attention import AttentionFunction, AttentionCallable
from ltx2.modules.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx2.modules.timestep_embedding import get_timestep_embedding, TimestepEmbedding
from ltx2.modules.rope import apply_rotary_emb, LTXRopeType, precompute_freqs_cis,generate_freq_grid_np,generate_freq_grid_pytorch
from fastvideo.utils.context_parallel import (
    is_cp_enabled, get_cp_world_size,
    scatter_sequence, gather_sequence, gather_for_loss,
    apply_ulysses_attention,
    pad_to_cp_divisible, unpad_from_cp,
)



__all__ = [
    'LTX2Model', 'LTX2ModelType', 'VideoLatentShape', 'X0Model', 'to_denoised', 'Modality',
    # Re-exports for backward compatibility (used by text_encoder.py)
    'Attention', 'FeedForward', 'LTXRopeType', 'precompute_freqs_cis', 'rms_norm',
]



def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square (RMS) normalize `x` over its last dimension.
    Thin wrapper around `torch.nn.functional.rms_norm` that infers the normalized
    shape and forwards `weight` and `eps`.
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)



@dataclass(frozen=True)
class Modality:

    latent: (
        torch.Tensor
    )  # Shape: (B, T, D) where B is the batch size, T is the number of tokens, and D is input dimension
    timesteps: torch.Tensor  # Shape: (B, T) where T is the number of timesteps
    positions: (
        torch.Tensor
    )  # Shape: (B, 3, T) for video, where 3 is the number of dimensions and T is the number of tokens
    context: torch.Tensor
    enabled: bool = True
    context_mask: torch.Tensor | None = None



@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int

class PixArtAlphaTextProjection(torch.nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.
    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features: int, hidden_size: int, out_features: int | None = None, act_fn: str = "gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = torch.nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = torch.nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = torch.nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = torch.nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Timesteps(torch.nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb

class PixArtAlphaCombinedTimestepSizeEmbeddings(torch.nn.Module):
    """
    For PixArt-Alpha.
    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(
        self,
        embedding_dim: int,
        size_emb_dim: int,
    ):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)
        return timesteps_emb


class AdaLayerNormSingle(torch.nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).
    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).
    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
        )

        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep




class VideoLatentShape:
    """Helper class for video latent shape calculations."""
    def __init__(self, batch_size, num_frames, num_channels, height, width):
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.height = height
        self.width = width
    
    def to_tuple(self):
        return (self.batch_size, self.num_frames, self.num_channels, self.height, self.width)


class LTX2ModelType(Enum):
    """Model type for LTX-2."""
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTX2ModelType.AudioVideo, LTX2ModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTX2ModelType.AudioVideo, LTX2ModelType.AudioOnly)


def sinusoidal_embedding_1d(dim, position):
    """
    Sinusoidal positional embedding for timesteps (WAN-style).
    
    Args:
        dim: Embedding dimension (must be even)
        position: Position tensor [B]
        
    Returns:
        Embeddings [B, dim]
    """
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x




class LTX2RMSNorm(nn.Module):
    """
    RMS Normalization (WAN-style).
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class LTX2LayerNorm(nn.LayerNorm):
    """
    Layer Normalization (WAN-style).
    """

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        """
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)



class GELUApprox(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)

        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        # attention_function can be an enum *or* a custom callable
        out = self.attention_function(q, k, v, self.heads, mask)
        return self.to_out(out)


class LTX2SelfAttention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        # Sparse attention parameters
        enable_sparse_attention: bool = False,
        sparse_block_size: tuple = (4, 4, 4),
        sparse_ratio: float = 0.125,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

        # Sparse attention support
        self.enable_sparse_attention = enable_sparse_attention
        if enable_sparse_attention:
            from ltx2.modules.sparse_attention import BlockSparseAttention
            self.sparse_attn = BlockSparseAttention(
                block_size=sparse_block_size,
                sparsity_ratio=sparse_ratio,
                num_heads=heads,
                head_dim=dim_head,
            )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        video_shape: tuple = None,  # (T, H, W) for sparse attention
    ) -> torch.Tensor:
        is_self_attn = context is None
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        # Choose attention implementation
        if self.enable_sparse_attention and video_shape is not None:
            # Use sparse attention
            # Reshape to [B, seq_len, num_heads, head_dim]
            q = q.view(q.shape[0], q.shape[1], self.heads, self.dim_head)
            k = k.view(k.shape[0], k.shape[1], self.heads, self.dim_head)
            v = v.view(v.shape[0], v.shape[1], self.heads, self.dim_head)

            out = self.sparse_attn(q, k, v, video_shape=video_shape, mask=mask)

            # Flatten back: [B, seq_len, num_heads * head_dim]
            out = out.reshape(q.shape[0], q.shape[1], -1)
        elif is_self_attn and is_cp_enabled():
            # Ulysses-style context parallel for self-attention
            out = apply_ulysses_attention(q, k, v, self.heads, self.attention_function, mask)
        else:
            # Use standard attention (cross-attention or CP disabled)
            out = self.attention_function(q, k, v, self.heads, mask)

        return self.to_out(out)

class LTX2CrossAttention(LTX2SelfAttention):
    """
    Cross attention layer - inherits from LTX2SelfAttention.
    Uses the parent class's forward method by simply passing context.
    """
    pass  # All functionality is inherited from parent class

# Alias for backward compatibility (used by text_encoder.py)
# Attention = LTX2SelfAttention

class LTX2AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        enable_camera_injection: bool = False,
        # Sparse attention parameters
        enable_sparse_attention: bool = False,
        sparse_block_size: tuple = (4, 4, 4),
        sparse_ratio: float = 0.125,
    ):
        super().__init__()

        self.idx = idx
        self.enable_camera_injection = enable_camera_injection
        self.attn1 = LTX2SelfAttention(
            query_dim=video.dim,
            heads=video.heads,
            dim_head=video.d_head,
            context_dim=None,
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
            # Pass sparse parameters to self-attention
            enable_sparse_attention=enable_sparse_attention,
            sparse_block_size=sparse_block_size,
            sparse_ratio=sparse_ratio,
        )
        # Cross-attention does not use sparse attention
        self.attn2 = LTX2CrossAttention(
            query_dim=video.dim,
            context_dim=video.context_dim,
            heads=video.heads,
            dim_head=video.d_head,
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
        )
        self.ff = FeedForward(video.dim, dim_out=video.dim)
        self.scale_shift_table = torch.nn.Parameter(torch.empty(6, video.dim))

        self.norm_eps = norm_eps

        # Camera injection layers (scale/shift modulation, zero-initialized)
        if enable_camera_injection:
            self.cam_injector_layer1 = nn.Linear(video.dim, video.dim)
            self.cam_injector_layer2 = nn.Linear(video.dim, video.dim)
            self.cam_scale_layer = nn.Linear(video.dim, video.dim)
            self.cam_shift_layer = nn.Linear(video.dim, video.dim)
            self._init_camera_weights()

    def _init_camera_weights(self):
        """Initialize camera injection layers with small values for stable training."""
        nn.init.xavier_uniform_(self.cam_injector_layer1.weight)
        nn.init.zeros_(self.cam_injector_layer1.bias)
        nn.init.xavier_uniform_(self.cam_injector_layer2.weight)
        nn.init.zeros_(self.cam_injector_layer2.bias)
        nn.init.constant_(self.cam_scale_layer.weight, 1e-6)
        nn.init.constant_(self.cam_scale_layer.bias, 0.0)
        nn.init.constant_(self.cam_shift_layer.weight, 1e-6)
        nn.init.constant_(self.cam_shift_layer.bias, 0.0)

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

   

    def forward(  # noqa: PLR0915
        self,
        x: torch.Tensor | None,
        timesteps: torch.Tensor | None = None,
        freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
        plucker_emb: Optional[torch.Tensor] = None,
        video_shape: tuple = None,  # (T, H, W) for sparse attention
    ) -> torch.Tensor:

        batch_size = x.shape[0] if x is not None else  1
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)
        
        vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
            self.scale_shift_table, x.shape[0], timesteps, slice(0, 3)
        )
        if not perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx):
            norm_vx = rms_norm(x, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
            v_mask = perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, x)
            x = x + self.attn1(norm_vx, pe=freqs, video_shape=video_shape) * vgate_msa * v_mask

        # Camera signal injection: scale/shift after self-attention, before cross-attention
        if self.enable_camera_injection:
            if plucker_emb is not None:
                cam_hidden = self.cam_injector_layer1(plucker_emb)
                cam_hidden = torch.nn.functional.silu(cam_hidden)
                cam_hidden = self.cam_injector_layer2(cam_hidden)
                cam_hidden = cam_hidden + plucker_emb  # residual
                cam_scale = self.cam_scale_layer(cam_hidden)
                cam_shift = self.cam_shift_layer(cam_hidden)
                del cam_hidden  # free immediately
                x = (1.0 + cam_scale) * x + cam_shift
                del cam_scale, cam_shift  # free immediately
            else:
                _dummy = torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
                _ = self.cam_injector_layer1(_dummy)
                _ = self.cam_injector_layer2(_dummy)
                _ = self.cam_scale_layer(_dummy)
                _ = self.cam_shift_layer(_dummy)

        x = x + self.attn2(rms_norm(x, eps=self.norm_eps), context=context, mask=context_mask)

        del vshift_msa, vscale_msa, vgate_msa

        vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
            self.scale_shift_table, x.shape[0], timesteps, slice(3, None)
        )
        x_scaled = rms_norm(x, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
        x = x + self.ff(x_scaled) * vgate_mlp

        del vshift_mlp, vscale_mlp, vgate_mlp

        return x




class LTX2Model(ModelMixin, ConfigMixin):


    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['LTX2AttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 # Model type
                 model_type='video_only',
                 # Video params from LTX2
                num_attention_heads: int = 32,
                attention_head_dim: int = 128,
                in_channels: int = 128,
                out_channels: int = 128,
                num_layers: int = 48,
                cross_attention_dim: int = 4096,
                norm_eps: float = 1e-06,
                attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
                caption_channels: int = 3840,
                positional_embedding_theta: float = 10000.0,
                positional_embedding_max_pos: list[int] | None = None,
                timestep_scale_multiplier: int = 1000,
                use_middle_indices_grid: bool = True,
                rope_type: LTXRopeType = LTXRopeType.SPLIT,
                double_precision_rope: bool = False,
                normalize_rope_positions: bool = True,  # Whether to normalize positions to [0, 1] in RoPE
                normalize_time_by_fps: bool = True,  # Whether to normalize time positions by FPS (divide by 25)
                 # Legacy WAN params (kept for compatibility)
                 patch_size=(1, 1, 1),
                 text_len=1024,
                 in_dim=128,
                 dim=4096,
                 ffn_dim=16384,
                 freq_dim=256,
                 text_dim=3840,
                 out_dim=128,
                 num_heads=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 # Camera control params (optional)
                 enable_camera_control: bool = False,
                 camera_injection_mode: str = "scale_shift",  # "scale_shift", "additive", or "wan_inject"
                 camera_split_plucker: bool = False,  # split moment/direction branches
                 num_actions: int = 81,
                 plucker_hidden_dim: int = 64,
                 continuous_camera_dropout_prob: float = 0.1,
                 discrete_camera_dropout_prob: float = 0.1,
                 enable_continuous_camera: bool = True,
                 enable_discrete_camera: bool = True,
                 # Sparse Attention params (optional, default off)
                 enable_sparse_attention: bool = False,
                 sparse_block_size: tuple = (4, 4, 4),
                 sparse_ratio_train: float = 0.125,  # 1/8 for training
                 sparse_ratio_inference: float = 0.0625,  # 1/16 for inference
                 ):
      
        super().__init__()

        # Parse model type
        attention_type = AttentionFunction.FLASH_ATTENTION_3
        print("forcing flash-attn3")
        self.model_type = LTX2ModelType.VideoOnly
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # LTX2 specific params
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.use_middle_indices_grid = use_middle_indices_grid
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.normalize_rope_positions = normalize_rope_positions  # Whether to normalize positions to [0, 1] in RoPE
        self.normalize_time_by_fps = normalize_time_by_fps  # Whether to normalize time positions by FPS
        self.yarn_rope = False  # YaRN RoPE extrapolation for long videos (off by default)
        self.yarn_max_train_seconds = 20.0  # maximum duration the pretrained model supports

        if positional_embedding_max_pos is None:
            positional_embedding_max_pos = [2048, 2048, 2048]
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        ## VIDEO INPUT COMPONENTS
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        self.adaln_single = AdaLayerNormSingle(self.inner_dim)
        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels,hidden_size=self.inner_dim)
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

        # ===== Sparse Attention configuration =====
        self.enable_sparse_attention = enable_sparse_attention
        self.sparse_block_size = sparse_block_size
        self.sparse_ratio = sparse_ratio_train  # Default to training ratio
        self.sparse_ratio_train = sparse_ratio_train
        self.sparse_ratio_inference = sparse_ratio_inference

        # ===== Camera control modules (optional) =====
        self.enable_camera_control = enable_camera_control
        self.camera_injection_mode = camera_injection_mode
        self.camera_split_plucker = camera_split_plucker
        self.enable_continuous_camera = enable_continuous_camera
        self.enable_discrete_camera = enable_discrete_camera
        self._camera_use_cross_norm = os.environ.get('LTX_CAMERA_USE_CROSS_NORM', '0') == '1'
        self._cross_norm_scale = float(os.environ.get('LTX_CROSS_NORM_SCALE', '0.2'))

        if enable_camera_control:
            from ltx2.modules.camera_control import (
                WanCameraAdapter,
                DiscreteActionEmbedder, CameraControlDropout, DiscreteActionDropout,
            )
            print(f"[LTX2-CameraControl] Enabled: mode={camera_injection_mode}, "
                  f"continuous={enable_continuous_camera}, discrete={enable_discrete_camera}")

            if enable_discrete_camera:
                self.action_embedder = DiscreteActionEmbedder(
                    dim=self.inner_dim, freq_dim=freq_dim, num_actions=num_actions)
                self.discrete_action_dropout = DiscreteActionDropout(discrete_camera_dropout_prob)

            if enable_continuous_camera:
                self.camera_dropout = CameraControlDropout(continuous_camera_dropout_prob)
                # All modes now use WanCameraAdapter with pixel-resolution Plucker
                # (PixelUnshuffle(8) + Conv2d(k=4,s=4) = 32x lossless spatial downsample)
                _adapter_kwargs = dict(in_channels=6, out_dim=self.inner_dim,
                                       vae_temporal_stride=8, split_plucker=camera_split_plucker)
                if camera_injection_mode == "scale_shift":
                    self.plucker_processor = WanCameraAdapter(**_adapter_kwargs)
                elif camera_injection_mode == "additive":
                    self.additive_camera_adapter = WanCameraAdapter(**_adapter_kwargs)
                elif camera_injection_mode == "wan_inject":
                    self.wan_camera_adapter = WanCameraAdapter(**_adapter_kwargs)

        # ===== Transformer blocks =====
        # Note: No need for preprocessors in WAN-style - all processing happens directly in forward()
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
            )
            if self.model_type.is_video_enabled()
            else None
        )
     
        _enable_block_camera = (enable_camera_control and enable_continuous_camera
                                and camera_injection_mode == "scale_shift")
        self.blocks = torch.nn.ModuleList(
            [
                LTX2AttentionBlock(
                    idx=idx,
                    video=video_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                    enable_camera_injection=_enable_block_camera,
                    # Sparse attention configuration
                    enable_sparse_attention=enable_sparse_attention,
                    sparse_block_size=sparse_block_size,
                    sparse_ratio=self.sparse_ratio,
                )
                for idx in range(num_layers)
            ]
        )


        # ===== RoPE frequencies =====
        # Note: RoPE frequencies are computed dynamically in forward() using LTX2's precompute_freqs_cis
        # This allows for flexible video sizes and maintains LTX2's original behavior

        # initialize weights (skip if loading from checkpoint to save time)
        # self.init_weights()  # Will be called manually if needed

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for transformer blocks (WAN-style)."""
        self.gradient_checkpointing = True

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks."""
        self.gradient_checkpointing = enable

    def set_sparse_ratio(self, ratio: float):
        """
        Dynamically adjust sparsity ratio (for switching between training/inference).
        
        Args:
            ratio: float - new sparsity ratio (e.g., 0.125 for training, 0.0625 for inference)
        """
        self.sparse_ratio = ratio
        for block in self.blocks:
            if hasattr(block.attn1, 'sparse_attn'):
                block.attn1.sparse_attn.sparsity_ratio = ratio


    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTX2."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        action_labels: Optional[torch.Tensor] = None,
        plucker_coords: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass through the diffusion model (WAN-style interface).

        Args:
            x (List[Tensor]): List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor): Diffusion timesteps tensor of shape [B]
            context (List[Tensor]): List of text embeddings each with shape [L, C]
            seq_len (int): Maximum sequence length for positional encoding
            action_labels (Tensor, optional): Discrete action labels [B] or [B, T] (0~80)
            plucker_coords (Tensor, optional): Plucker coordinates [B, 6, F, H, W]

        Returns:
            Tensor: Denoised video tensors with shape [B, C_out, F, H, W]
        """
        return self._forward(x, t, context, seq_len,
                             action_labels=action_labels,
                             plucker_coords=plucker_coords,
                             cond_latent_frames=kwargs.get('cond_latent_frames', 0),
                             **{k: v for k, v in kwargs.items() if k != 'cond_latent_frames'})

    def _forward(
        self,
        x,
        t,
        context,
        seq_len,
        fps: float = 25.0,
        action_labels: Optional[torch.Tensor] = None,
        plucker_coords: Optional[torch.Tensor] = None,
        cond_latent_frames: int = 0,
        **kwargs,
    ):
        """
        Internal forward pass (WAN-style).
        
        Args:
            x: List[[C, F, H, W]] or [B, C, F, H, W] input latents
            t: [B] timesteps (0-1000 range)
            context: List[[L, C]] or [B, L, C] text embeddings
            seq_len: max sequence length
            fps: frames per second for position normalization (default 25.0)
            action_labels: Optional discrete action labels [B] or [B, T]
            plucker_coords: Optional Plucker coordinates [B, 6, F, H, W]
            cond_latent_frames: number of i2v/v2v condition latents; their timestep is forced to 0
        """
        from ltx2.modules.patchifier import VideoLatentPatchifier, VideoLatentShape, SpatioTemporalScaleFactors, get_pixel_coords
        
        device = self.patchify_proj.weight.device
        dtype = self.patchify_proj.weight.dtype
        
        # x is List[[C, F, H, W]], convert to [B, C, F, H, W]
        if isinstance(x, list):
            x_stacked = torch.stack([u for u in x])  # [B, C, F, H, W]
        else:
            x_stacked = x
        
        batch_size, channels, num_frames, height, width = x_stacked.shape
        
        # Video shape for sparse attention (T, H, W)
        video_shape = (num_frames, height, width) if self.enable_sparse_attention else None
        
        # Patchify: [B, C, F, H, W] -> [B, num_tokens, C]
        patchifier = VideoLatentPatchifier(patch_size=1)
        target_shape = VideoLatentShape(
            batch=batch_size, channels=channels, frames=num_frames,
            height=height, width=width
        )
        latent = patchifier.patchify(x_stacked).to(dtype)
        num_tokens = latent.shape[1]
        
        # Time embedding: t can be either timestep (0-1000) or sigma (0-1)
        # Detect based on range: if max > 1, it's timesteps, else it's sigma
        t_float = t.float()
        if t_float.max() > 1.0:
            sigma_val = t_float / 1000.0  # Convert timestep to sigma
        else:
            sigma_val = t_float  # Already sigma
        timesteps = sigma_val.unsqueeze(-1).unsqueeze(-1).expand(-1, num_tokens, 1).clone().to(dtype)
        
        if cond_latent_frames > 0:
            tokens_per_frame = height * width
            cond_tokens = cond_latent_frames * tokens_per_frame
            cond_tokens = min(cond_tokens, num_tokens)  # safety clamp
            timesteps[:, :cond_tokens, :] = 0.0  # condition frames get sigma=0
        
        if not hasattr(self, '_ts_print_count'):
            self._ts_print_count = 0
        if self._ts_print_count < 2:
            sigma_gen = timesteps[0, -1, 0].item() if num_tokens > 0 else 0
            sigma_cond = timesteps[0, 0, 0].item() if num_tokens > 0 else 0
            print(f"[Model.forward] sigma_cond={sigma_cond:.4f}, sigma_gen={sigma_gen:.4f}, "
                  f"cond_frames={cond_latent_frames}, x={x_stacked.shape}, cam={getattr(self, 'enable_camera_control', False)}")
            self._ts_print_count += 1
        
        # Positions with FPS normalization (critical for temporal consistency)
        scale_factors = SpatioTemporalScaleFactors(time=8, width=32, height=32)
        latent_coords = patchifier.get_patch_grid_bounds(output_shape=target_shape, device=device)
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=scale_factors,
            causal_fix=True
        ).float()
        
        # Normalize time dimension by FPS (important for correct motion)
        if getattr(self, 'normalize_time_by_fps', True):
            positions[:, 0, ...] = positions[:, 0, ...] / fps
        positions = positions.to(dtype)
        
        # Context: stack if list
        if isinstance(context, list):
            # Pad to max length and stack
            max_len = max(c.shape[0] for c in context)
            context_padded = []
            for c in context:
                if c.shape[0] < max_len:
                    pad = torch.zeros(max_len - c.shape[0], c.shape[1], device=c.device, dtype=c.dtype)
                    c = torch.cat([c, pad], dim=0)
                context_padded.append(c)
            context_tensor = torch.stack(context_padded).to(dtype)  # [B, L, C]
        else:
            context_tensor = context.to(dtype)
        
        # Patch embedding
        hidden = self.patchify_proj(latent)
        
        tokens_per_frame = height * width
        cond_tokens = cond_latent_frames * tokens_per_frame if cond_latent_frames > 0 else 0
        
        if cond_latent_frames > 0 and cond_tokens < num_tokens:
            sigma_gen = timesteps[0, -1, 0:1]  # [1] sigma of the generated frames
            unique_sigmas = torch.stack([torch.zeros_like(sigma_gen), sigma_gen])  # [2, 1]
            unique_scaled = unique_sigmas * self.timestep_scale_multiplier
            unique_emb, unique_embedded = self.adaln_single(
                unique_scaled.squeeze(-1), hidden_dtype=hidden.dtype,
            )  # unique_emb: [2, 6*dim], unique_embedded: [2, dim]
            
            token_indices = torch.ones(num_tokens, dtype=torch.long, device=device)
            token_indices[:cond_tokens] = 0
            timestep_emb = unique_emb[token_indices].unsqueeze(0)  # [1, num_tokens, 6*dim]
            embedded_timestep = unique_embedded[token_indices].unsqueeze(0)  # [1, num_tokens, dim]
        else:
            single_sigma = timesteps[0, 0, 0:1]  # [1]
            single_scaled = single_sigma * self.timestep_scale_multiplier
            single_emb, single_embedded = self.adaln_single(
                single_scaled, hidden_dtype=hidden.dtype,
            )  # single_emb: [1, 6*dim], single_embedded: [1, dim]
            
            timestep_emb = single_emb.unsqueeze(0).expand(batch_size, num_tokens, -1)  # [B, num_tokens, 6*dim]
            embedded_timestep = single_embedded.unsqueeze(0).expand(batch_size, num_tokens, -1)  # [B, num_tokens, dim]

        # ===== Discrete action injection (add to timestep_emb via adaln_single) =====
        # FSDP requires ALL ranks to access the same parameters in forward pass.
        # Must always access action_embedder params even when action_labels is None.
        if (getattr(self, 'enable_camera_control', False) and
                getattr(self, 'enable_discrete_camera', False)):
            if action_labels is not None:
                action_emb = self.action_embedder(action_labels)  # [B, dim] or [B, T, dim]
                if action_emb.dim() == 3:
                    action_emb = action_emb.repeat_interleave(tokens_per_frame, dim=1)  # [B, T*H*W, dim]
                    if action_emb.shape[1] != num_tokens:
                        action_emb = action_emb.permute(0, 2, 1)
                        action_emb = F.interpolate(action_emb.float(), size=num_tokens, mode='linear', align_corners=True).to(dtype=action_emb.dtype)
                        action_emb = action_emb.permute(0, 2, 1)
                action_emb = self.discrete_action_dropout(action_emb)
                action_emb_mod = self.adaln_single.linear(
                    self.adaln_single.silu(action_emb)
                )  # [B, num_tokens, 6*dim] or [B, 6*dim]
                if not hasattr(self, '_act_diag_cnt'):
                    self._act_diag_cnt = 0
                if self._act_diag_cnt < 2:
                    _ratio = action_emb_mod.abs().mean() / (timestep_emb.abs().mean() + 1e-12)
                    print(f"[ActionDiag] action_emb_mod: shape={action_emb_mod.shape}, "
                          f"timestep_emb: shape={timestep_emb.shape}, "
                          f"mean_abs_ratio={_ratio:.4f}")
                    self._act_diag_cnt += 1
                if action_emb_mod.dim() == 2:
                    timestep_emb = timestep_emb + action_emb_mod.unsqueeze(1)
                else:
                    timestep_emb = timestep_emb + action_emb_mod
            else:
                # Dummy forward to keep FSDP params in sync across ranks
                dummy_labels = torch.zeros(1, dtype=torch.long, device=device)
                _ = self.action_embedder(dummy_labels)
                _ = self.discrete_action_dropout(torch.zeros(1, self.inner_dim, device=device, dtype=dtype))

        # Caption projection
        context_proj = self.caption_projection(context_tensor)
        context_proj = context_proj.view(batch_size, -1, hidden.shape[-1])

        # ===== Continuous camera signal processing =====
        # FSDP requires ALL ranks to access the same parameters in forward pass.
        # Must always access camera adapter params even when plucker_coords is None.
        plucker_emb = None
        if (getattr(self, 'enable_camera_control', False) and
                getattr(self, 'enable_continuous_camera', False)):
            if plucker_coords is not None:
                # All modes now receive pixel-resolution Plucker [B, 6, F, H_pixel, W_pixel]
                # WanCameraAdapter handles spatial downsample internally (PixelUnshuffle + strided Conv)

                if self.camera_injection_mode == "scale_shift":
                    # plucker_processor is now a WanCameraAdapter
                    plucker_emb = self.plucker_processor(
                        plucker_coords, num_frames, height, width)  # [B, L, dim]
                    # Store raw adapter output for aux rotation loss (before dropout)
                    if self.training and getattr(self, '_enable_cam_aux', False):
                        self._cam_features_for_aux = plucker_emb
                        self._cam_aux_shape = (num_frames, height, width)
                    plucker_emb = self.camera_dropout(plucker_emb)
                    # Sequence length safety check
                    if plucker_emb.shape[1] != num_tokens:
                        plucker_emb = plucker_emb.permute(0, 2, 1)
                        plucker_emb = torch.nn.functional.interpolate(
                            plucker_emb.float(), size=num_tokens, mode='linear', align_corners=True,
                        ).to(dtype=plucker_emb.dtype)
                        plucker_emb = plucker_emb.permute(0, 2, 1)
                    # Cross Normalization: align plucker_emb distribution to hidden
                    if getattr(self, '_camera_use_cross_norm', False):
                        _mean_h = hidden.detach().mean(dim=-1, keepdim=True)
                        _std_h = hidden.detach().std(dim=-1, keepdim=True)
                        _mean_c = plucker_emb.mean(dim=-1, keepdim=True)
                        _std_c = plucker_emb.std(dim=-1, keepdim=True)
                        plucker_emb = (plucker_emb - _mean_c) * (_std_h / (_std_c + 1e-12)) + _mean_h
                        plucker_emb = plucker_emb * self._cross_norm_scale
                elif self.camera_injection_mode == "additive":
                    # additive_camera_adapter is now a WanCameraAdapter
                    cam_features = self.additive_camera_adapter(
                        plucker_coords, num_frames, height, width)  # [B, L, C]
                    # Store raw adapter output for aux rotation loss (before dropout)
                    if self.training and getattr(self, '_enable_cam_aux', False):
                        self._cam_features_for_aux = cam_features
                        self._cam_aux_shape = (num_frames, height, width)
                    cam_features = self.camera_dropout(cam_features)
                    if cam_features.shape[1] != num_tokens:
                        cam_features = cam_features.permute(0, 2, 1)
                        cam_features = torch.nn.functional.interpolate(
                            cam_features.float(), size=num_tokens, mode='linear', align_corners=True,
                        ).to(dtype=cam_features.dtype)
                        cam_features = cam_features.permute(0, 2, 1)
                    if not hasattr(self, '_cam_diag_cnt'):
                        self._cam_diag_cnt = 0
                    if self._cam_diag_cnt < 2:
                        _ratio = cam_features.abs().mean() / (hidden.abs().mean() + 1e-12)
                        print(f"[CameraDiag] cam_features: mean_abs={cam_features.abs().mean():.6f}, "
                              f"hidden: mean_abs={hidden.abs().mean():.6f}, "
                              f"ratio={_ratio:.4f} (needs to exceed ~0.05 to have a visible effect)")
                        self._cam_diag_cnt += 1
                    hidden = hidden + cam_features
                    del cam_features  # free immediately
                elif self.camera_injection_mode == "wan_inject":
                    # wan_inject: plucker_coords is at pixel resolution [B, 6, F_pixel, H_pixel, W_pixel]
                    # WanCameraAdapter handles downsample + conv2d internally
                    cam_features = self.wan_camera_adapter(
                        plucker_coords, num_frames, height, width)  # [B, L, C]
                    # Store raw adapter output for aux rotation loss (before dropout)
                    if self.training and getattr(self, '_enable_cam_aux', False):
                        self._cam_features_for_aux = cam_features
                        self._cam_aux_shape = (num_frames, height, width)
                    cam_features = self.camera_dropout(cam_features)
                    if cam_features.shape[1] != num_tokens:
                        cam_features = cam_features.permute(0, 2, 1)
                        cam_features = torch.nn.functional.interpolate(
                            cam_features.float(), size=num_tokens, mode='linear', align_corners=True,
                        ).to(dtype=cam_features.dtype)
                        cam_features = cam_features.permute(0, 2, 1)
                    if not hasattr(self, '_cam_diag_cnt'):
                        self._cam_diag_cnt = 0
                    if self._cam_diag_cnt < 2:
                        _ratio = cam_features.abs().mean() / (hidden.abs().mean() + 1e-12)
                        print(f"[CameraDiag-WanInject] cam_features: mean_abs={cam_features.abs().mean():.6f}, "
                              f"hidden: mean_abs={hidden.abs().mean():.6f}, "
                              f"ratio={_ratio:.4f} (needs to exceed ~0.05 to have a visible effect)")
                        self._cam_diag_cnt += 1
                    hidden = hidden + cam_features
                    del cam_features  # free immediately
            else:
                # No camera data: dummy forward to keep FSDP params in sync across ranks
                # All modes use WanCameraAdapter now, min spatial = 32x32
                # F=1 pixel frame → temporal unshuffle repeats to S frames → F_lat=1
                if self.camera_injection_mode == "scale_shift":
                    dummy_pl = torch.zeros(1, 6, 1, 32, 32, device=device, dtype=dtype)
                    _ = self.plucker_processor(dummy_pl, 1, 1, 1)
                    _ = self.camera_dropout(torch.zeros(1, 1, self.inner_dim, device=device, dtype=dtype))
                elif self.camera_injection_mode == "additive":
                    dummy_pl = torch.zeros(1, 6, 1, 32, 32, device=device, dtype=dtype)
                    _ = self.additive_camera_adapter(dummy_pl, 1, 1, 1)
                    _ = self.camera_dropout(torch.zeros(1, 1, self.inner_dim, device=device, dtype=dtype))
                elif self.camera_injection_mode == "wan_inject":
                    dummy_pl = torch.zeros(1, 6, 1, 32, 32, device=device, dtype=dtype)
                    _ = self.wan_camera_adapter(dummy_pl, 1, 1, 1)
                    _ = self.camera_dropout(torch.zeros(1, 1, self.inner_dim, device=device, dtype=dtype))
                # Clear aux cache for dummy path
                if getattr(self, '_enable_cam_aux', False):
                    self._cam_features_for_aux = None
                    self._cam_aux_shape = None

        # Positional embeddings (RoPE)
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        freqs = precompute_freqs_cis(
            indices_grid=positions,
            dim=self.inner_dim,
            out_dtype=latent.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
            normalize_positions=self.normalize_rope_positions,
        )
        
        # ===== Context Parallel: pad + scatter sequence-dimension tensors =====
        _cp_orig_seq_len = None
        if is_cp_enabled():
            cp_size = get_cp_world_size()
            # Pad sequence to be divisible by cp_size (handles arbitrary T*H*W)
            hidden, _cp_orig_seq_len = pad_to_cp_divisible(hidden, dim=1)
            timestep_emb, _ = pad_to_cp_divisible(timestep_emb, dim=1)
            freqs_cos, _ = pad_to_cp_divisible(freqs[0], dim=2)
            freqs_sin, _ = pad_to_cp_divisible(freqs[1], dim=2)
            freqs = (freqs_cos, freqs_sin)
            if plucker_emb is not None:
                plucker_emb, _ = pad_to_cp_divisible(plucker_emb, dim=1)
            # Scatter
            hidden = scatter_sequence(hidden, dim=1)
            timestep_emb = scatter_sequence(timestep_emb, dim=1)
            freqs = (scatter_sequence(freqs[0], dim=2), scatter_sequence(freqs[1], dim=2))
            if plucker_emb is not None:
                plucker_emb = scatter_sequence(plucker_emb, dim=1)
            # Update video_shape T dimension for sparse attention
            if video_shape is not None:
                T_vs, H_vs, W_vs = video_shape
                # Use padded T: hidden.shape[1] = padded_seq / cp_size, T_padded = padded_seq / (H*W)
                padded_seq = hidden.shape[1] * cp_size
                T_padded = padded_seq // (H_vs * W_vs)
                video_shape = (T_padded // cp_size, H_vs, W_vs)

        # Transformer blocks with gradient checkpointing support (WAN-style)
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            block_kwargs = {
                "timesteps": timestep_emb,
                "freqs": freqs,
                "context": context_proj,
                "context_mask": None,
                "perturbations": None,
                "plucker_emb": plucker_emb,
                "video_shape": video_shape,  # For sparse attention
            }
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden,
                    **block_kwargs,
                    use_reentrant=False,
                )
            else:
                hidden = block(x=hidden, **block_kwargs)
        
        # ===== Context Parallel: gather sequence back + unpad =====
        if is_cp_enabled():
            hidden = gather_for_loss(hidden, dim=1)
            if _cp_orig_seq_len is not None:
                hidden = unpad_from_cp(hidden, _cp_orig_seq_len, dim=1)

        # Output processing
        out = self._process_output(self.scale_shift_table, self.norm_out, self.proj_out, hidden, embedded_timestep)
        
        # Unpatchify: [B, num_tokens, C] -> [B, C, F, H, W]
        out_bcfhw = patchifier.unpatchify(out, target_shape)
        
        return out_bcfhw

    def init_weights(self):
        """
        Initialize model parameters using Xavier initialization.
        """
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init video embeddings
        if self.model_type.is_video_enabled():
            # patchify_proj is the equivalent of patch_embedding
            nn.init.xavier_uniform_(self.patchify_proj.weight)
            # caption_projection is the text embedding
            for m in self.caption_projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=.02)
            # adaln_single contains time embedding
            for m in self.adaln_single.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=.02)
            # proj_out is the output head
            nn.init.zeros_(self.proj_out.weight)

    
# =============================================================================
# Utility Functions
# =============================================================================

def to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoising velocity to denoised sample.
    Formula: x0 = sample - velocity * sigma
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)


# =============================================================================
# X0Model Wrapper
# =============================================================================

class X0Model(nn.Module):
    """X0 model wrapper: expose a velocity model as an x0 model."""

    def __init__(self, velocity_model: LTX2Model):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self, x, t, context, seq_len, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass: outputs x0 (denoised samples) instead of velocity.
        
        Args:
            x: List of input video tensors [C, F, H, W]
            t: Timesteps [B]
            context: List of text embeddings [L, C]
            seq_len: Maximum sequence length
            
        Returns:
            x0 prediction [B, C, F, H, W]
        """
        velocity = self.velocity_model(x, t, context, seq_len, **kwargs)
        
        # Convert velocity to x0: x0 = xt - sigma * velocity
        sigma = t.float() / 1000.0
        while sigma.dim() < velocity.dim():
            sigma = sigma.unsqueeze(-1)
        
        # x is List[[C, F, H, W]], stack to [B, C, F, H, W]
        if isinstance(x, list):
            x_stacked = torch.stack(x)
        else:
            x_stacked = x
            
        x0 = x_stacked - sigma * velocity
        return x0

