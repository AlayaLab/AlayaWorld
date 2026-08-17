import os
from enum import Enum
from typing import Protocol

import torch

from ltx2.modules.rope import LTXRopeType, apply_rotary_emb

memory_efficient_attention = None
flash_attn_func = None
try:
    # Try flash-attn first (faster and more memory efficient)
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
    try:
        # Fallback to xformers if flash-attn not available
        from xformers.ops import memory_efficient_attention
    except ImportError:
        memory_efficient_attention = None


# =============================================================================
# Wan-style flash_attention (varlen API, used by model_wan.py)
# Imported from wan/modules/attention.py for compatibility.
# =============================================================================
import warnings as _warnings

_FLASH_ATTN_3_AVAILABLE = False
try:
    import flash_attn_interface as _flash_attn_interface
    def _is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "h200" in device_name or "hopper" in device_name
    _FLASH_ATTN_3_AVAILABLE = _is_hopper_gpu()
except ModuleNotFoundError:
    _flash_attn_interface = None

_FLASH_ATTN_2_AVAILABLE = False
try:
    import flash_attn as _flash_attn_mod
    _FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    _flash_attn_mod = None


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Wan-style flash attention using varlen API.

    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:vv] for u, vv in zip(k, k_lens)]))
        v = half(torch.cat([u[:vv] for u, vv in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not _FLASH_ATTN_3_AVAILABLE:
        _warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and _FLASH_ATTN_3_AVAILABLE:
        x = _flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif _FLASH_ATTN_2_AVAILABLE:
        x = _flash_attn_mod.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        # Fallback to PyTorch SDPA
        _warnings.warn('Flash attention not available, falling back to PyTorch SDPA.')
        # Reshape for SDPA: [total_tokens, nheads, dim] -> [B, nheads, L, dim]
        n_heads = q.size(1) if q.dim() == 3 else q.size(-2)
        # Simple fallback: reshape assuming uniform lengths
        q_r = q.unflatten(0, (b, lq)).transpose(1, 2)
        k_r = k.unflatten(0, (b, lk)).transpose(1, 2)
        v_r = v.unflatten(0, (b, lk)).transpose(1, 2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q_r.to(torch.bfloat16), k_r.to(torch.bfloat16), v_r.to(torch.bfloat16),
            dropout_p=dropout_p, is_causal=causal,
        ).transpose(1, 2)  # [B, L, nheads, dim]

    return x.type(out_dtype)


class AttentionCallable(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int,
        mask: torch.Tensor | None = None,
        window_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        b, S, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        # FreeLong++: convert window_size (left, right) into a banded boolean mask
        # so PYTORCH backend honors sliding-window like FA3 does. window_size=(-1,-1)
        # means full attention -> no extra mask.
        if window_size is not None and window_size != (-1, -1):
            left, right = window_size
            # build [S, S] bool mask: True = allow attend
            idx = torch.arange(S, device=q.device)
            d = idx.view(-1, 1) - idx.view(1, -1)  # row=q, col=k; d = q_pos - k_pos
            window_mask = (d <= left) & (-d <= right)  # |q - k| within window
            window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
            if mask is None:
                mask = window_mask
            else:
                # combine with existing mask (additive bias style or bool)
                if mask.dtype == torch.bool:
                    mask = mask & window_mask
                else:
                    # additive bias: mask out (set -inf) outside window
                    mask = mask.masked_fill(~window_mask, float('-inf'))

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        # Use math backend to avoid cuDNN compatibility issues
        # This is slower but more compatible with different CUDA/cuDNN versions
        try:
            # Prefer new API (PyTorch 2.10+) to avoid FutureWarning
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                ctx = sdpa_kernel(SDPBackend.MATH)
            except (ImportError, AttributeError):
                ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
            with ctx:
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        except Exception as e:
            # Fallback to manual attention if SDPA fails
            print(f"[Warning] SDPA failed, using manual attention: {e}")
            scale = dim_head ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            attn = attn.softmax(dim=-1)
            out = attn @ v
        
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class XFormersAttention(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad to a multiple of 8
            pad = 8 - mask.shape[-1] % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
            # doesn't this remove the padding again??
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
        window_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if flash_attn_func is None:
            raise RuntimeError("FlashAttention3 was selected but `flash-attn` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # flash_attn expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention3")

        # FreeLong++: sliding window via FA's native support. window_size is
        # (left, right) in TOKEN units. (-1, -1) = full attention (default).
        ws = window_size if window_size is not None else (-1, -1)

        # flash_attn_func expects bf16 or fp16
        _use_fa3 = (
            _FLASH_ATTN_3_AVAILABLE
            and _flash_attn_interface is not None
            and os.environ.get("ALAYA_USE_FA3", "0") == "1"
        )
        if _use_fa3:
            out = _flash_attn_interface.flash_attn_func(
                q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
                window_size=ws,
            )
            if isinstance(out, tuple):
                out = out[0]
        else:
            out = flash_attn_func(
                q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
                window_size=ws,
            )
        out = out.reshape(b, -1, heads * dim_head)
        return out.to(v.dtype)


# Global set to track which attention types have been logged



class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    DEFAULT = "default"

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int,
        mask: torch.Tensor | None = None,
        window_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        # FreeLong++: window_size honored by FA3 (native) and PYTORCH (banded mask).
        # XFormers ignores it (window_size construction would need attn_bias variant).
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()(q, k, v, heads, mask, window_size=window_size)
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            if mask is not None:
                if memory_efficient_attention is not None:
                    return XFormersAttention()(q, k, v, heads, mask)
                return PytorchAttention()(q, k, v, heads, mask, window_size=window_size)
            return FlashAttention3()(q, k, v, heads, mask, window_size=window_size)
        else:
            # Default behavior: Flash-Attn > XFormers > PyTorch
            if flash_attn_func is not None and mask is None:
                return FlashAttention3()(q, k, v, heads, mask, window_size=window_size)
            elif memory_efficient_attention is not None:
                return XFormersAttention()(q, k, v, heads, mask)
            else:
                return PytorchAttention()(q, k, v, heads, mask, window_size=window_size)

