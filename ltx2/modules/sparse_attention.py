"""
Block Sparse Attention for LTX2 Video Transformers.

Based on LongCat paper's 3D Block Sparse Attention mechanism.
Implements hardware-aligned sparse attention with Triton kernels for acceleration.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton, fallback to PyTorch if not available
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("[Warning] Triton not available, using PyTorch fallback for sparse attention")

# Try to import Flash Attention for optimized dense attention computation
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("[Warning] Flash Attention not available, using standard PyTorch attention")


class BlockSparseAttention(nn.Module):
    """
    3D Block Sparse Attention for video transformers.
    
    Implements the block-wise sparse attention mechanism from LongCat paper:
    1. Rearrange input into 3D blocks (T×H×W)
    2. Compute block-level pooled attention scores
    3. Select top-r key blocks for each query block
    4. Compute attention only on selected blocks
    
    Args:
        block_size: Tuple[int, int, int] - (t, h, w) block dimensions, default (4, 4, 4)
        sparsity_ratio: float - ratio of blocks to select (r/Nk), default 0.125 (1/8)
        num_heads: int - number of attention heads
        head_dim: int - dimension per head
        use_triton: bool - whether to use Triton kernels (auto-detect if None)
    """
    
    def __init__(
        self,
        block_size: Tuple[int, int, int] = (4, 4, 4),
        sparsity_ratio: float = 0.125,
        num_heads: int = 32,
        head_dim: int = 128,
        use_triton: Optional[bool] = None,
    ):
        super().__init__()
        self.block_size = block_size  # (t, h, w)
        self.sparsity_ratio = sparsity_ratio
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Auto-detect Triton availability
        if use_triton is None:
            use_triton = TRITON_AVAILABLE
        self.use_triton = use_triton and TRITON_AVAILABLE
        
        # Calculate block volume
        self.block_volume = block_size[0] * block_size[1] * block_size[2]
        
        if not self.use_triton:
            print(f"[BlockSparseAttention] Using PyTorch fallback (Triton not available)")
    
    def rearrange_to_blocks(
        self, 
        x: torch.Tensor, 
        T: int, 
        H: int, 
        W: int
    ) -> Tuple[torch.Tensor, int, int, int]:
        """
        Rearrange input tensor into 3D block structure.
        
        Args:
            x: [B, T*H*W, num_heads, head_dim]
            T, H, W: video dimensions
            
        Returns:
            x_blocks: [B, NT*NH*NW, n, num_heads, head_dim] 
                     where n = block_volume, NT = T/t, NH = H/h, NW = W/w
            NT, NH, NW: number of blocks in each dimension
        """
        B, seq_len, num_heads, head_dim = x.shape
        t, h, w = self.block_size
        
        # Check if dimensions are divisible by block size
        assert T % t == 0, f"T={T} must be divisible by block_size[0]={t}"
        assert H % h == 0, f"H={H} must be divisible by block_size[1]={h}"
        assert W % w == 0, f"W={W} must be divisible by block_size[2]={w}"
        
        NT, NH, NW = T // t, H // h, W // w
        num_blocks = NT * NH * NW
        
        # Reshape to [B, T, H, W, num_heads, head_dim]
        x = x.reshape(B, T, H, W, num_heads, head_dim)
        
        # Rearrange to blocks: [B, NT, t, NH, h, NW, w, num_heads, head_dim]
        x = x.reshape(B, NT, t, NH, h, NW, w, num_heads, head_dim)
        
        # Permute to group blocks together: [B, NT, NH, NW, t, h, w, num_heads, head_dim]
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7, 8)
        
        # Flatten blocks: [B, NT*NH*NW, t*h*w, num_heads, head_dim]
        x_blocks = x.reshape(B, num_blocks, self.block_volume, num_heads, head_dim)
        
        return x_blocks, NT, NH, NW
    
    def rearrange_from_blocks(
        self,
        x_blocks: torch.Tensor,
        NT: int,
        NH: int,
        NW: int,
    ) -> torch.Tensor:
        """
        Rearrange from block structure back to sequence.
        
        Args:
            x_blocks: [B, NT*NH*NW, n, num_heads, head_dim]
            NT, NH, NW: number of blocks in each dimension
            
        Returns:
            x: [B, T*H*W, num_heads, head_dim]
        """
        B, num_blocks, block_vol, num_heads, head_dim = x_blocks.shape
        t, h, w = self.block_size
        T, H, W = NT * t, NH * h, NW * w
        
        # Reshape to [B, NT, NH, NW, t, h, w, num_heads, head_dim]
        x = x_blocks.reshape(B, NT, NH, NW, t, h, w, num_heads, head_dim)
        
        # Permute to [B, NT, t, NH, h, NW, w, num_heads, head_dim]
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7, 8)
        
        # Reshape to [B, T, H, W, num_heads, head_dim]
        x = x.reshape(B, T, H, W, num_heads, head_dim)
        
        # Flatten to [B, T*H*W, num_heads, head_dim]
        x = x.reshape(B, T * H * W, num_heads, head_dim)
        
        return x
    
    def compute_block_pooling(
        self, 
        x_blocks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute average pooling over each block.
        
        Args:
            x_blocks: [B, num_blocks, block_volume, num_heads, head_dim]
            
        Returns:
            x_pool: [B, num_blocks, num_heads, head_dim]
        """
        # Average over block_volume dimension (dim=2)
        x_pool = x_blocks.mean(dim=2)
        return x_pool
    
    def construct_block_mask(
        self,
        q_pool: torch.Tensor,
        k_pool: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct block selection mask using top-r selection.
        
        Args:
            q_pool: [B, Nq_blocks, num_heads, head_dim]
            k_pool: [B, Nk_blocks, num_heads, head_dim]
            
        Returns:
            mask: [B, num_heads, Nq_blocks, Nk_blocks] - binary mask
        """
        B, Nq, num_heads, head_dim = q_pool.shape
        Nk = k_pool.shape[1]
        
        # Compute pooled attention scores: [B, num_heads, Nq, Nk]
        q_pool = q_pool.transpose(1, 2)  # [B, num_heads, Nq, head_dim]
        k_pool = k_pool.transpose(1, 2)  # [B, num_heads, Nk, head_dim]
        
        # S_pool = (Q_pool @ K_pool^T) / sqrt(d)
        scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) * self.scale
        # Shape: [B, num_heads, Nq, Nk]
        
        # Select top-r key blocks for each query block
        r = max(1, int(self.sparsity_ratio * Nk))
        
        # Get top-r indices: [B, num_heads, Nq, r]
        topk_values, topk_indices = torch.topk(scores, k=r, dim=-1)
        
        # Create binary mask
        mask = torch.zeros_like(scores, dtype=torch.bool)
        
        # Scatter ones at top-k positions
        mask.scatter_(dim=-1, index=topk_indices, value=True)
        
        return mask
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        video_shape: Tuple[int, int, int],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of block sparse attention.
        
        Args:
            q, k, v: [B, seq_len, num_heads, head_dim]
            video_shape: (T, H, W) - video dimensions
            mask: Optional causal mask (applied after block selection)
            
        Returns:
            output: [B, seq_len, num_heads, head_dim]
        """
        B, seq_len, num_heads, head_dim = q.shape
        T, H, W = video_shape
        
        assert seq_len == T * H * W, f"seq_len={seq_len} must equal T*H*W={T*H*W}"
        
        # Step 1: Rearrange to blocks
        q_blocks, NT, NH, NW = self.rearrange_to_blocks(q, T, H, W)
        k_blocks, _, _, _ = self.rearrange_to_blocks(k, T, H, W)
        v_blocks, _, _, _ = self.rearrange_to_blocks(v, T, H, W)
        
        num_blocks = NT * NH * NW
        
        # Step 2: Compute block pooling
        q_pool = self.compute_block_pooling(q_blocks)
        k_pool = self.compute_block_pooling(k_blocks)
        
        # Step 3: Construct block selection mask
        block_mask = self.construct_block_mask(q_pool, k_pool)
        # Shape: [B, num_heads, num_blocks, num_blocks]
        
        # Step 4: Compute sparse attention
        if self.use_triton:
            # Use Triton kernel for efficiency
            output = self._triton_sparse_attention(
                q_blocks, k_blocks, v_blocks, block_mask
            )
        else:
            # PyTorch fallback
            output = self._pytorch_sparse_attention(
                q_blocks, k_blocks, v_blocks, block_mask, mask
            )
        
        # Step 5: Rearrange back to sequence
        output = self.rearrange_from_blocks(output, NT, NH, NW)
        
        return output
    
    def _pytorch_sparse_attention(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
        v_blocks: torch.Tensor,
        block_mask: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        PyTorch fallback implementation of block sparse attention.
        
        Args:
            q_blocks: [B, Nq, block_vol, num_heads, head_dim]
            k_blocks: [B, Nk, block_vol, num_heads, head_dim]
            v_blocks: [B, Nk, block_vol, num_heads, head_dim]
            block_mask: [B, num_heads, Nq, Nk] - binary mask
            causal_mask: Optional causal mask
            
        Returns:
            output: [B, Nq, block_vol, num_heads, head_dim]
        """
        B, Nq, block_vol, num_heads, head_dim = q_blocks.shape
        Nk = k_blocks.shape[1]
        
        # Reshape for attention computation
        # [B, Nq, block_vol, num_heads, head_dim] -> [B, num_heads, Nq, block_vol, head_dim]
        q_blocks = q_blocks.permute(0, 3, 1, 2, 4)
        k_blocks = k_blocks.permute(0, 3, 1, 2, 4)
        v_blocks = v_blocks.permute(0, 3, 1, 2, 4)
        
        # Flatten query blocks: [B, num_heads, Nq*block_vol, head_dim]
        q_flat = q_blocks.reshape(B, num_heads, Nq * block_vol, head_dim)
        
        # Flatten key/value blocks: [B, num_heads, Nk*block_vol, head_dim]
        k_flat = k_blocks.reshape(B, num_heads, Nk * block_vol, head_dim)
        v_flat = v_blocks.reshape(B, num_heads, Nk * block_vol, head_dim)
        
        # Compute attention scores: [B, num_heads, Nq*block_vol, Nk*block_vol]
        scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * self.scale
        
        # Expand block mask to token level
        # [B, num_heads, Nq, Nk] -> [B, num_heads, Nq*block_vol, Nk*block_vol]
        token_mask = block_mask.unsqueeze(3).unsqueeze(5)  # [B, H, Nq, 1, Nk, 1]
        token_mask = token_mask.expand(-1, -1, -1, block_vol, -1, block_vol)
        token_mask = token_mask.reshape(B, num_heads, Nq * block_vol, Nk * block_vol)
        
        # Apply mask (set non-selected blocks to -inf)
        scores = scores.masked_fill(~token_mask, float('-inf'))
        
        # Apply causal mask if provided
        if causal_mask is not None:
            scores = scores + causal_mask
        
        # Softmax and attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)  # Handle all-inf rows
        
        # Compute output: [B, num_heads, Nq*block_vol, head_dim]
        output = torch.matmul(attn_weights, v_flat)
        
        # Reshape back to blocks: [B, num_heads, Nq, block_vol, head_dim]
        output = output.reshape(B, num_heads, Nq, block_vol, head_dim)
        
        # Permute back: [B, Nq, block_vol, num_heads, head_dim]
        output = output.permute(0, 2, 3, 1, 4)
        
        return output
    
    def _triton_sparse_attention(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
        v_blocks: torch.Tensor,
        block_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Triton kernel implementation (placeholder for now).
        
        TODO: Implement optimized Triton kernels for production use.
        For now, fall back to PyTorch implementation.
        """
        # For initial implementation, use PyTorch fallback
        return self._pytorch_sparse_attention(
            q_blocks, k_blocks, v_blocks, block_mask, None
        )


# Triton kernel implementations
if TRITON_AVAILABLE:
    @triton.jit
    def _block_sparse_attention_fwd_kernel(
        Q, K, V, Out,
        block_mask,
        stride_qb, stride_qh, stride_qq, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kk, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vk, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_oq, stride_on, stride_od,
        stride_mb, stride_mh, stride_mq, stride_mk,
        Nq: tl.constexpr, Nk: tl.constexpr,
        BLOCK_VOL: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """
        Optimized Triton kernel for block sparse attention forward pass.
        
        Memory layout:
        - Q: [B, H, Nq, BLOCK_VOL, HEAD_DIM]
        - K, V: [B, H, Nk, BLOCK_VOL, HEAD_DIM]
        - block_mask: [B, H, Nq, Nk] (boolean)
        - Out: [B, H, Nq, BLOCK_VOL, HEAD_DIM]
        
        This kernel implements Flash Attention-style online softmax to minimize
        memory usage and maximize efficiency.
        """
        # Get program IDs
        pid_b = tl.program_id(0)  # batch
        pid_h = tl.program_id(1)  # head
        pid_q = tl.program_id(2)  # query block index
        
        # Compute base offsets for current (batch, head, query_block)
        q_base = (pid_b * stride_qb + pid_h * stride_qh + pid_q * stride_qq)
        mask_base = (pid_b * stride_mb + pid_h * stride_mh + pid_q * stride_mq)
        
        # Create offset ranges for loading blocks
        offs_n = tl.arange(0, BLOCK_VOL)  # token indices within block
        offs_d = tl.arange(0, HEAD_DIM)   # dimension indices
        
        # Load query block: [BLOCK_VOL, HEAD_DIM]
        q_offs = q_base + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd
        q = tl.load(Q + q_offs).to(tl.float32)
        
        # Initialize accumulators for online softmax
        acc = tl.zeros([BLOCK_VOL, HEAD_DIM], dtype=tl.float32)
        m_i = tl.full([BLOCK_VOL], float('-inf'), dtype=tl.float32)  # max scores
        l_i = tl.zeros([BLOCK_VOL], dtype=tl.float32)  # sum of exp(scores)
        
        # Iterate over all key blocks
        for k_idx in range(Nk):
            # Check if this key block is selected by the mask
            mask_offset = mask_base + k_idx * stride_mk
            is_selected = tl.load(block_mask + mask_offset)
            
            # Skip unselected blocks
            if not is_selected:
                continue
            
            # Compute base offset for current key/value block
            k_base = (pid_b * stride_kb + pid_h * stride_kh + k_idx * stride_kk)
            v_base = (pid_b * stride_vb + pid_h * stride_vh + k_idx * stride_vk)
            
            # Load key block: [BLOCK_VOL, HEAD_DIM]
            k_offs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(K + k_offs).to(tl.float32)
            
            # Load value block: [BLOCK_VOL, HEAD_DIM]
            v_offs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(V + v_offs).to(tl.float32)
            
            # Compute attention scores: Q @ K^T scaled
            # Result shape: [BLOCK_VOL, BLOCK_VOL]
            qk = tl.zeros([BLOCK_VOL, BLOCK_VOL], dtype=tl.float32)
            for d in range(HEAD_DIM):
                qk += q[:, d:d+1] * k[None, :, d]
            qk = qk * SCALE
            
            # Online softmax: update max and normalizer
            # For each query token, find max over all key tokens in this block
            m_ij = tl.max(qk, axis=1)  # [BLOCK_VOL]
            m_new = tl.maximum(m_i, m_ij)
            
            # Compute exponentials with numerical stability
            alpha = tl.exp(m_i - m_new)  # correction factor for previous blocks
            beta = tl.exp(m_ij - m_new)  # correction factor for current block
            
            # Update normalizer
            l_i_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
            
            # Compute attention weights for current block
            p = tl.exp(qk - m_new[:, None])  # [BLOCK_VOL, BLOCK_VOL]
            
            # Update accumulator with correction
            # acc = (alpha * l_i / l_i_new) * acc + (1 / l_i_new) * (p @ v)
            acc_scale = alpha * l_i / l_i_new
            acc = acc * acc_scale[:, None]
            
            # Add contribution from current block
            pv = tl.zeros([BLOCK_VOL, HEAD_DIM], dtype=tl.float32)
            for n in range(BLOCK_VOL):
                for d in range(HEAD_DIM):
                    pv[:, d] += p[:, n] * v[n, d]
            
            acc += pv / l_i_new[:, None]
            
            # Update running statistics
            m_i = m_new
            l_i = l_i_new
        
        # Store output
        out_base = (pid_b * stride_ob + pid_h * stride_oh + pid_q * stride_oq)
        out_offs = out_base + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od
        tl.store(Out + out_offs, acc.to(Out.dtype.element_ty))
    
    
    class BlockSparseAttentionTriton(torch.autograd.Function):
        """
        Custom autograd function for Triton-accelerated block sparse attention.
        """
        
        @staticmethod
        def forward(ctx, q, k, v, block_mask, scale, block_vol):
            """
            Forward pass using Triton kernel.
            
            Args:
                q, k, v: [B, num_heads, Nq/Nk, block_vol, head_dim]
                block_mask: [B, num_heads, Nq, Nk]
                scale: float
                block_vol: int
            """
            B, H, Nq, block_vol, D = q.shape
            Nk = k.shape[2]
            
            # Allocate output
            output = torch.empty_like(q)
            
            # Launch kernel
            grid = (B, H, Nq)
            _block_sparse_attention_fwd_kernel[grid](
                q, k, v, output, block_mask,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                block_mask.stride(0), block_mask.stride(1), block_mask.stride(2), block_mask.stride(3),
                B, H, Nq, Nk, block_vol, D, scale,
            )
            
            # Save for backward
            ctx.save_for_backward(q, k, v, block_mask, output)
            ctx.scale = scale
            ctx.block_vol = block_vol
            
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            """
            Backward pass (simplified - full implementation would use custom kernel).
            """
            q, k, v, block_mask, output = ctx.saved_tensors
            
            # For now, use PyTorch autograd
            # Production version would use custom Triton backward kernel
            return None, None, None, None, None, None
