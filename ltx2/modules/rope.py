import functools
import math
import os
from enum import Enum
from typing import Callable, Tuple

import numpy as np
import torch
from einops import rearrange


def _use_wan_rope() -> bool:
    """Enable Wan-style RoPE on all dimensions when LTX_USE_WAN_ROPE=1.
    The environment variable is set at startup from the corresponding argument."""
    return os.environ.get('LTX_USE_WAN_ROPE', '0') == '1'


def _use_wan_rope_t_only() -> bool:
    """LTX_USE_WAN_ROPE_T_ONLY=1: use Wan-style RoPE on the temporal dimension only (H/W stay LTX default).
    This buys length generalization while keeping the pretrained spatial representation."""
    return os.environ.get('LTX_USE_WAN_ROPE_T_ONLY', '0') == '1'


class LTXRopeType(Enum):
    """RoPE variants."""
    INTERLEAVED = "interleaved"
    SPLIT = "split"


def apply_rotary_emb(
    input_tensor: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
) -> torch.Tensor:
    """Apply RoPE.
    - LTX default: split mode (the head dimension is paired front-half with back-half).
    - LTX_USE_WAN_ROPE=1: Wan-style interleaved (adjacent elements form a complex pair).
    """
    if _use_wan_rope():
        return apply_interleaved_rotary_emb(input_tensor, *freqs_cis)
    return apply_split_rotary_emb(input_tensor, *freqs_cis)


def apply_interleaved_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    """Wan / standard RoPE: adjacent dimensions form a complex pair (a, b) -> (a + 1j*b),
    rotated by (cos + 1j*sin):
        new_even = a*cos - b*sin
        new_odd  = a*sin + b*cos

    Args:
        input_tensor: [batch, num_heads, num_tokens, head_dim]
        cos_freqs / sin_freqs: [batch, num_heads, num_tokens, head_dim//2]
    """
    needs_reshape = False
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        input_tensor = input_tensor.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    split_input = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    even = split_input[..., 0]   # [b, h, t, head_dim//2]
    odd = split_input[..., 1]    # [b, h, t, head_dim//2]

    new_even = even * cos_freqs - odd * sin_freqs
    new_odd = even * sin_freqs + odd * cos_freqs

    output = torch.stack([new_even, new_odd], dim=-1)  # [b, h, t, head_dim//2, 2]
    output = rearrange(output, "... d r -> ... (d r)")  # [b, h, t, head_dim]

    if needs_reshape:
        bb, hh, tt, _ = output.shape
        output = output.swapaxes(1, 2).reshape(bb, tt, -1)
    return output


def apply_split_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    """
    Apply split RoPE to the input tensor.
    
    Core 2D rotation:
        [x', y'] = [x·cos(θ) - y·sin(θ), x·sin(θ) + y·cos(θ)]
    
    Args:
        input_tensor: Q or K tensor of shape [batch_size, num_heads, num_tokens, head_dim]
                      e.g. [1, 32, 25515, 128]
        cos_freqs: cos(position × frequency), shape [batch_size, num_heads, num_tokens, head_dim//2]
                   e.g. [1, 32, 25515, 64]
        sin_freqs: sin(position × frequency), shape [batch_size, num_heads, num_tokens, head_dim//2]
                   e.g. [1, 32, 25515, 64]
    
    Returns:
        the rotated tensor, same shape as input_tensor
    """
    needs_reshape = False
    # print(input_tensor.shape, cos_freqs.shape, sin_freqs.shape)
    
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        input_tensor = input_tensor.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    # [batch_size, num_heads, num_tokens, head_dim] -> [batch_size, num_heads, num_tokens, 2, head_dim//2]
    split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
    
    first_half_input = split_input[..., :1, :]   # x: [batch, heads, tokens, 1, 64]
    second_half_input = split_input[..., 1:, :]  # y: [batch, heads, tokens, 1, 64]

    output = split_input * cos_freqs.unsqueeze(-2)  # [b, h, t, 2, 64]
    
    first_half_output = output[..., :1, :]   # x·cos: [b, h, t, 1, 64]
    second_half_output = output[..., 1:, :]  # y·cos: [b, h, t, 1, 64]

    first_half_output.addcmul_(-sin_freqs.unsqueeze(-2), second_half_input)  # x·cos - y·sin
    # y' = y·cos + x·sin
    second_half_output.addcmul_(sin_freqs.unsqueeze(-2), first_half_input)   # y·cos + x·sin

    output = rearrange(output, "... d r -> ... (d r)")
    
    if needs_reshape:
        output = output.swapaxes(1, 2).reshape(b, t, -1)

    return output


@functools.lru_cache(maxsize=5)
def generate_freq_grid_np(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta

    n_elem = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(start) / np.log(theta),
            np.log(end) / np.log(theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


@functools.lru_cache(maxsize=5)
def generate_freq_grid_wan(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    """Wan-style frequencies: indices[i] = 1 / theta^(2i/freq_dim).
    (LTX style uses theta^linspace(0,1) * pi/2; Wan style multiplies the absolute position directly)."""
    n_elem = 2 * positional_embedding_max_pos_count
    freq_dim = inner_dim // n_elem  # frequencies per cos/sin pair per dimension
    indices = 1.0 / (positional_embedding_theta ** (
        torch.arange(0, freq_dim, dtype=torch.float64) / float(freq_dim)
    ))
    return indices.to(torch.float32)


@functools.lru_cache(maxsize=5)
def generate_freq_grid_pytorch(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    theta = positional_embedding_theta # 10000.0
    start = 1 # 1
    end = theta # 10000.0
    n_elem = 2 * positional_embedding_max_pos_count # 3 * 2 = 6
    # print("n_elem:",n_elem)
    # print("start:",start)
    # print("end:",end)
    # print("theta:",theta)
    # print("positional_embedding_max_pos_count:",positional_embedding_max_pos_count)
    # print("positional_embedding_theta:",positional_embedding_theta)
    # print("inner_dim:",inner_dim)
    # print("positional_embedding_max_pos_count:",positional_embedding_max_pos_count)
    # print("positional_embedding_theta:",positional_embedding_theta)
    # print("inner_dim:",inner_dim)
    indices = theta ** (
        torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem,
            dtype=torch.float32,
        )
    )
    # print("indices:",indices)
    # print("indices.shape:",indices.shape)
    # print("indices max:",indices.max())
    # print("indices min:",indices.min())
    indices = indices.to(dtype=torch.float32)

    indices = indices * math.pi / 2

    return indices


def get_fractional_positions(
    indices_grid: torch.Tensor, 
    max_pos: list[int],
    normalize: bool = True,
) -> torch.Tensor:
    """
    Return position coordinates, optionally normalized.
    
    Args:
        indices_grid: [batch, n_dims, num_tokens] absolute position indices
        max_pos: [n_dims] maximum position per dimension, used for normalization
        normalize: normalize into the [0, 1] range
            - True (default): position / max_pos, used for standard inference
            - False: use absolute positions, used for streaming experiments
    
    Returns:
        fractional_positions: [batch, num_tokens, n_dims] position coordinates
    """
    n_pos_dims = indices_grid.shape[1]
    # indices_grid: [1, 3, 25515]
    # max_pos: [20, 2048, 2048]
    assert n_pos_dims == len(max_pos), (
        f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    )
    
    if normalize:
        fractional_positions = torch.stack(
            [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
            dim=-1,
        )
    else:
        fractional_positions = torch.stack(
            [indices_grid[:, i].float() for i in range(n_pos_dims)],
            dim=-1,
        )
    
    return fractional_positions


def generate_freqs(
    indices: torch.Tensor, 
    indices_grid: torch.Tensor, 
    max_pos: list[int], 
    use_middle_indices_grid: bool,
    normalize_positions: bool = True,
    time_yarn_config: dict = None,
) -> torch.Tensor:
    """
    Build RoPE frequencies.
    
    Args:
        indices: frequency basis [D]
        indices_grid: position grid
        max_pos: maximum position values
        use_middle_indices_grid: use the patch centre
        normalize_positions: normalize positions into [0, 1]
        time_yarn_config: YaRN configuration dict; None keeps the original path. Contains:
            - scale: extrapolation factor
            - train_frac_pos_max: maximum fractional position seen in training
            - beta_fast: high-frequency boundary (rotations within the training range)
            - beta_slow: low-frequency boundary
            - extrapolation_factor: extrapolation weight
    """
    if use_middle_indices_grid:
        assert len(indices_grid.shape) == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start, indices_grid_end = indices_grid[..., 0], indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    if _use_wan_rope():
        n_dims = indices_grid.shape[1]
        positions = indices_grid.float()  # [B, n_dims, num_tokens]
        indices = indices.to(device=positions.device)
        per_dim_freqs = []
        for i in range(n_dims):
            pos_i = positions[:, i, :].unsqueeze(-1)  # [B, num_tokens, 1]
            angle_i = pos_i * indices  # [B, num_tokens, freq_dim]
            per_dim_freqs.append(angle_i)
        freqs = torch.stack(per_dim_freqs, dim=2).transpose(-1, -2).flatten(2)
        return freqs

    if _use_wan_rope_t_only():
        # indices.shape = [freq_dim], freq_dim = inner_dim // (2 * n_dims)
        n_dims = indices_grid.shape[1]
        freq_dim = indices.shape[-1]
        inner_dim = freq_dim * (2 * n_dims)
        wan_indices_T = generate_freq_grid_wan(10000.0, n_dims, inner_dim).to(
            device=indices_grid.device
        )

        fractional_positions = get_fractional_positions(
            indices_grid, max_pos, normalize=normalize_positions
        )
        ltx_indices = indices.to(device=fractional_positions.device)
        # ltx_angle: [B, num_tokens, n_dims, freq_dim]
        ltx_angle = ltx_indices * (fractional_positions.unsqueeze(-1) * 2 - 1)

        T_pos = indices_grid[:, 0, :].float().unsqueeze(-1)  # [B, num_tokens, 1]
        T_angle = T_pos * wan_indices_T  # [B, num_tokens, freq_dim]

        angle_mixed = ltx_angle.clone()
        angle_mixed[:, :, 0, :] = T_angle
        freqs = angle_mixed.transpose(-1, -2).flatten(2)
        return freqs

    fractional_positions = get_fractional_positions(
        indices_grid, max_pos, normalize=normalize_positions
    )
    indices = indices.to(device=fractional_positions.device)
    
    if time_yarn_config is not None and time_yarn_config.get('scale', 1.0) > 1.0:
        s = time_yarn_config['scale']
        train_frac = time_yarn_config.get('train_frac_pos_max', 0.33)
        beta_fast = time_yarn_config.get('beta_fast', 2.0)
        beta_slow = time_yarn_config.get('beta_slow', 0.1)
        ext_factor = time_yarn_config.get('extrapolation_factor', 1.0)
        
        rotations = indices * train_frac / math.pi
        
        ramp = ((rotations - beta_slow) / (beta_fast - beta_slow + 1e-6)).clamp(0, 1)
        mask = ramp * ext_factor
        
        scale_per_freq = mask * 1.0 + (1.0 - mask) * (1.0 / s)
        time_indices = indices * scale_per_freq
        
        
        angle_factors = fractional_positions.unsqueeze(-1) * 2 - 1  # [B, N, 3, 1]
        time_freqs = time_indices * angle_factors[:, :, 0:1, :]      # [B, N, 1, D]
        spatial_freqs = indices * angle_factors[:, :, 1:, :]           # [B, N, 2, D]
        freqs = torch.cat([time_freqs, spatial_freqs], dim=2)          # [B, N, 3, D]
        freqs = freqs.transpose(-1, -2).flatten(2)
    else:
        freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    
    return freqs


def split_freqs_cis(freqs: torch.Tensor, pad_size: int, num_attention_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape freqs to be compatible with multi-head attention
    #   - batch_size = 1
    #   - freq_dim = 2048 = num_heads × (head_dim // 2) = 32 × 64
    batch_size = cos_freq.shape[0]   # 1
    num_tokens = cos_freq.shape[1]   # 25515

    #   - num_heads = 32
    cos_freq = cos_freq.reshape(batch_size, num_tokens, num_attention_heads, -1)  # [1, 25515, 32, 64]
    sin_freq = sin_freq.reshape(batch_size, num_tokens, num_attention_heads, -1)  # [1, 25515, 32, 64]

    # [batch_size, num_tokens, num_heads, head_dim//2] -> [batch_size, num_heads, num_tokens, head_dim//2]
    cos_freq = torch.swapaxes(cos_freq, 1, 2)  # [1, 32, 25515, 64]
    sin_freq = torch.swapaxes(sin_freq, 1, 2)  # [1, 32, 25515, 64]
    return cos_freq, sin_freq


def precompute_freqs_cis(
    indices_grid: torch.Tensor,
    dim: int,
    out_dtype: torch.dtype,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
    freq_grid_generator: Callable[[float, int, int, torch.device], torch.Tensor] = generate_freq_grid_pytorch,
    normalize_positions: bool = True,
    time_yarn_config: dict = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the cos/sin RoPE frequencies.
    
    Args:
        normalize_positions: normalize positions into [0, 1]
        time_yarn_config: YaRN configuration dict; None keeps the original behaviour
    """
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    if _use_wan_rope():
        indices = generate_freq_grid_wan(theta, indices_grid.shape[1], dim)
    else:
        indices = freq_grid_generator(theta, indices_grid.shape[1], dim)
    freqs = generate_freqs(
        indices, indices_grid, max_pos, use_middle_indices_grid,
        normalize_positions=normalize_positions,
        time_yarn_config=time_yarn_config,
    )
    #     freqs[token] = [
    # ]
    # print("freqs:",freqs.shape)## 1, 25515, 2046 
    expected_freqs = dim // 2  # 4096/2 = 2048
    current_freqs = freqs.shape[-1]  # 2046
    pad_size = expected_freqs - current_freqs
    cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    return cos_freq.to(out_dtype), sin_freq.to(out_dtype)
