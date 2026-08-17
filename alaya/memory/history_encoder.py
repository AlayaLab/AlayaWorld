"""History encoder for video memory tokens.
=========================================================================

Implementation references:
    - arxiv 2512.23851 (Pretraining Frame Preservation for Lightweight
      Autoregressive Video History Embedding)
    - a public Wan 2.2 14B implementation

Inputs (two branches: high resolution + low resolution):
    latent: [B, 128, T_h, H_h, W_h] from the LTX VAE (32x spatial / 8x temporal compression);
                                          a typical 20s history is T_h=60, H_h=17, W_h=30

Outputs:
    mem_tokens: [B, N_mem_total, 4096]   (high-resolution tokens first, then low-resolution)
    mem_indices_grid: [B, 3, N_mem_total]
                                          (T, H, W) in the original latent coordinate system, used for RoPE

Two-branch design:
    High-resolution branch (re-representation):
        six-stage 3D convolution + 3D self-attention + 1x1 projection + output gate
        default compression (2, 2, 2): (60, 17, 30) -> (30, 8, 15) = 3600 tokens
    Low-resolution branch (coarse representation; removing it costs about 1.7 dB PSNR):
        temporal-spatial average pooling + 1x1 3D convolution + output gate
        default compression (2, 4, 4): (60, 17, 30) -> (30, 4, 7) = 840 tokens

Total memory tokens: 3600 + 840 = 4440
"""
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2], self.padding[2],   # W, both sides
            self.padding[1], self.padding[1],   # H, both sides
            2 * self.padding[0], 0,             # T, all padding on the left
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class _Conv3dBlock(nn.Module):
    """Causal 3D convolution + SiLU (no GroupNorm).
    Causal along T (padding on the left only) and symmetric along H/W."""
    def __init__(self, in_ch, out_ch, stride=(1, 1, 1), kernel_size=(3, 3, 3)):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)
        self.conv = CausalConv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class _SelfAttention3D(nn.Module):
    """3D self-attention over flattened (T, H, W) tokens.
    Causal mask over T: a token at frame t may attend only to frames <= t (spatially unrestricted).
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        spatial = H * W
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, T * spatial, C)
        residual = x_flat
        x_flat = self.norm(x_flat)
        qkv = self.qkv(x_flat).reshape(B, T * spatial, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        t_indices = torch.arange(T * spatial, device=x.device) // spatial   # [T*spatial]
        causal_mask = t_indices.unsqueeze(0) <= t_indices.unsqueeze(1)      # [seq, seq]
        attn_mask = torch.zeros(T * spatial, T * spatial, device=x.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~causal_mask, float('-inf'))

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, T * spatial, C)
        attn_out = self.proj(attn_out)
        out = residual + attn_out
        out = out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return out


class VideoHistoryEncoder(nn.Module):
    """Two-branch video history encoder.

    Args:
        in_channels: input latent channels
        out_channels: output channels (the transformer inner dimension)
        compress_t/h/w: compression of the high-resolution branch
        lr_compress_t/h/w: extra downsampling of the low-resolution branch
        gate_init: initial value of both output gates
        use_self_attn: add 3D self-attention after compression in the high-resolution branch
        use_lr_branch: enable the low-resolution branch; disable it to use the HR branch alone
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 4096,
        compress_t: int = 2,
        compress_h: int = 2,
        compress_w: int = 2,
        lr_compress_t: int = 2,
        lr_compress_h: int = 4,
        lr_compress_w: int = 4,
        gate_init: float = 0.0,
        use_self_attn: bool = True,
        use_lr_branch: bool = True,
        use_camera_pose: bool = False,    # accept camera poses as an extra input
        pose_emb_dim: int = 32,           # pose embedding channels (concatenated into the HR branch)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.compress_t = compress_t
        self.compress_h = compress_h
        self.compress_w = compress_w
        self.use_lr_branch = use_lr_branch
        self.lr_compress_t = lr_compress_t
        self.lr_compress_h = lr_compress_h
        self.lr_compress_w = lr_compress_w
        self.use_camera_pose = use_camera_pose
        self.pose_emb_dim = pose_emb_dim if use_camera_pose else 0

        # ★ Path C: pose embedder (c2w 12 floats → pose_emb_dim channels)
        if use_camera_pose:
            self.pose_embedder = nn.Sequential(
                nn.Linear(12, 64),
                nn.SiLU(),
                nn.Linear(64, pose_emb_dim),
            )
        else:
            self.pose_embedder = None

        # Stage 5: 256 → 512, stride=1                    (refine)
        # Stage 6: 512 → 512, stride=1                    (refine)
        # Stage 8: 512 → out_channels (1×1×1 projection)
        _stage1_in = in_channels + self.pose_emb_dim
        self.hr_stage1 = _Conv3dBlock(_stage1_in, 64, stride=(1, 1, 1))
        self.hr_stage2 = _Conv3dBlock(64, 128, stride=(compress_t, 1, 1))
        self.hr_stage3 = _Conv3dBlock(128, 256, stride=(1, compress_h, compress_w))
        self.hr_stage4 = _Conv3dBlock(256, 256, stride=(1, 1, 1))   # refine
        self.hr_stage5 = _Conv3dBlock(256, 512, stride=(1, 1, 1))
        self.hr_stage6 = _Conv3dBlock(512, 512, stride=(1, 1, 1))

        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.hr_attn = _SelfAttention3D(dim=512, num_heads=8)   # Stage 7: causal self-attn

        self.hr_proj = CausalConv3d(512, out_channels, kernel_size=1, stride=1, padding=0)

        self.output_gate = nn.Parameter(torch.full((1,), float(gate_init)))

        if use_lr_branch:
            self.register_buffer('lr_proj_weight', torch.zeros(out_channels, in_channels), persistent=False)
            self.register_buffer('lr_proj_bias', torch.zeros(out_channels), persistent=False)
            self._lr_proj_initialized = False

    def setup_lr_proj_from_patchify(self, patchify_proj: nn.Module):
        """Copy weights from the main transformer patchify projection into the LR branch buffer.

        Must be called before the main transformer is wrapped in FSDP, while the weight is still complete;
        calling it afterwards would copy a sharded weight and produce wrong results.
        """
        if not self.use_lr_branch:
            return
        with torch.no_grad():
            w = patchify_proj.weight.detach()
            self.lr_proj_weight = w.clone().to(self.lr_proj_weight.device)
            if patchify_proj.bias is not None:
                b = patchify_proj.bias.detach()
                self.lr_proj_bias = b.clone().to(self.lr_proj_bias.device)
            else:
                self.lr_proj_bias.zero_()
        self._lr_proj_initialized = True

    def _build_indices_grid(
        self, B: int, T_m: int, H_m: int, W_m: int,
        compress_t: int, compress_h: int, compress_w: int,
        device: torch.device,
        orig_t: int = None, orig_h: int = None, orig_w: int = None,
    ) -> torch.Tensor:
        """Compute the real (T, H, W) latent patch bounds of each memory token.

        Memory token i covers [i*compress, (i+1)*compress) of the original latent, centred at i*compress + compress/2.
        Returns [B, 3, N, 2] (start, end), matching the patchifier bounds format so the main model can run
        the usual pixel-coordinate and fps normalization and align RoPE centres with the generated segment.
        """
        t_start = torch.arange(T_m, device=device, dtype=torch.float32) * compress_t
        t_end = t_start + compress_t
        h_start = torch.arange(H_m, device=device, dtype=torch.float32) * compress_h
        h_end = h_start + compress_h
        w_start = torch.arange(W_m, device=device, dtype=torch.float32) * compress_w
        w_end = w_start + compress_w
        if orig_t is not None:
            t_end = t_end.clamp(max=float(orig_t))
        if orig_h is not None:
            h_end = h_end.clamp(max=float(orig_h))
        if orig_w is not None:
            w_end = w_end.clamp(max=float(orig_w))

        Ts, Hs, Ws = torch.meshgrid(t_start, h_start, w_start, indexing='ij')  # [T_m, H_m, W_m]
        Te, He, We = torch.meshgrid(t_end,   h_end,   w_end,   indexing='ij')

        starts = torch.stack([Ts.flatten(), Hs.flatten(), Ws.flatten()], dim=0)  # [3, N]
        ends   = torch.stack([Te.flatten(), He.flatten(), We.flatten()], dim=0)  # [3, N]
        bounds = torch.stack([starts, ends], dim=-1)                              # [3, N, 2]
        return bounds.unsqueeze(0).expand(B, -1, -1, -1).contiguous()             # [B, 3, N, 2]

    def forward(
        self,
        latent: torch.Tensor,
        patchify_proj: Optional[nn.Module] = None,
        past_c2w: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent: [B, in_channels, T_h, H_h, W_h] (e.g. [1, 128, 60, 17, 30])
            patchify_proj: [deprecated] the LR branch now uses an internal buffer; this argument is ignored
            past_c2w: [B, T_h, 4, 4] camera poses aligned with the latent T_h; required when use_camera_pose=True
                     and ignored otherwise.

        Returns:
            mem_tokens: [B, N_total, out_channels] (high-resolution tokens first, then low-resolution)
            mem_indices_grid: [B, 3, N_total]
        """
        if not hasattr(self, '_fwd_count'):
            self._fwd_count = 0
        self._fwd_count += 1
        _DBG_INTERVAL = 50
        try:
            import torch.distributed as _dist
            _dbg_rank = _dist.get_rank() if _dist.is_initialized() else 0
        except Exception:
            _dbg_rank = 0
        _do_dbg = (_dbg_rank == 0 and self._fwd_count % _DBG_INTERVAL == 1)

        B, C, T_h, H_h, W_h = latent.shape

        if self.use_camera_pose:
            assert past_c2w is not None, "past_c2w is required when use_camera_pose=True"
            assert past_c2w.shape[0] == B and past_c2w.shape[1] == T_h, \
                f"past_c2w shape {tuple(past_c2w.shape)} does not match [B={B}, T_h={T_h}, 4, 4]"
            _pose_flat = past_c2w[:, :, :3, :].reshape(B, T_h, 12).to(latent.dtype)
            _pose_emb = self.pose_embedder(_pose_flat)                    # [B, T_h, pose_emb_dim]
            if _do_dbg:
                _pe = _pose_emb.detach().float()
                _pf = _pose_flat.detach().float()
                print(
                    f"[HistEnc-DBG fwd={self._fwd_count}] pose(Path C) "
                    f"input(c2w_12) abs|μ|={_pf.abs().mean():.4f} σ={_pf.std():.4f} "
                    f"max={_pf.abs().max():.3f} | "
                    f"pose_emb out abs|μ|={_pe.abs().mean():.4f} σ={_pe.std():.4f} "
                    f"min={_pe.min():+.3f} max={_pe.max():+.3f} (dim={self.pose_emb_dim})",
                    flush=True,
                )
            _pose_grid = _pose_emb.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B, D, T_h, 1, 1]
            _pose_grid = _pose_grid.expand(B, self.pose_emb_dim, T_h, H_h, W_h)
            latent_with_pose = torch.cat([latent, _pose_grid], dim=1)     # [B, in_ch + D, T_h, H, W]
        else:
            latent_with_pose = latent

        x = self.hr_stage1(latent_with_pose)
        x = self.hr_stage2(x)        # Stage 2: compress T
        x = self.hr_stage3(x)        # Stage 3: compress H and W
        x = self.hr_stage4(x)        # Stage 4: refine
        x = self.hr_stage5(x)        # Stage 5: refine
        x = self.hr_stage6(x)        # Stage 6: refine
        if self.use_self_attn:
            x = self.hr_attn(x)      # Stage 7: causal self-attn
        x = self.hr_proj(x)          # Stage 8: 1×1 proj → [B, out_channels, T_hr, H_hr, W_hr]

        _, _, T_hr, H_hr, W_hr = x.shape
        hr_mem_tokens = x.permute(0, 2, 3, 4, 1).reshape(B, T_hr * H_hr * W_hr, self.out_channels)
        if _do_dbg:
            _hr_raw = hr_mem_tokens.detach().float()
            _hr_raw_stats = (_hr_raw.abs().mean().item(), _hr_raw.std().item(),
                             _hr_raw.min().item(), _hr_raw.max().item())
        hr_indices = self._build_indices_grid(
            B, T_hr, H_hr, W_hr,
            self.compress_t, self.compress_h, self.compress_w,
            device=latent.device,
            orig_t=T_h, orig_h=H_h, orig_w=W_h,
        )

        if not self.use_lr_branch:
            return hr_mem_tokens * self.output_gate, hr_indices

        if not self._lr_proj_initialized:
            raise RuntimeError(
                "the LR branch requires history_encoder.setup_lr_proj_from_patchify(transformer.patchify_proj) "
                "to be called before the main transformer is wrapped in FSDP (otherwise the weight is already sharded)"
            )

        _orig_dtype = latent.dtype
        latent_fp32 = latent.float() if latent.dtype != torch.float32 else latent
        target_T = (T_h + self.lr_compress_t - 1) // self.lr_compress_t
        target_H = (H_h + self.lr_compress_h - 1) // self.lr_compress_h
        target_W = (W_h + self.lr_compress_w - 1) // self.lr_compress_w
        lr_latent = F.interpolate(
            latent_fp32,
            size=(target_T, target_H, target_W),
            mode='trilinear',
            align_corners=False,
        )
        lr_latent = lr_latent.to(_orig_dtype)
        # [B, in_channels, T_lr, H_lr, W_lr] → [B, T_lr*H_lr*W_lr, in_channels]
        _, _, T_lr, H_lr, W_lr = lr_latent.shape
        lr_tokens = lr_latent.permute(0, 2, 3, 4, 1).reshape(B, T_lr * H_lr * W_lr, self.in_channels)
        lr_mem_tokens = F.linear(
            lr_tokens.to(self.lr_proj_weight.dtype),
            self.lr_proj_weight,
            self.lr_proj_bias,
        ).to(_orig_dtype)
        if _do_dbg:
            _lr_raw = lr_mem_tokens.detach().float()
            _lr_raw_stats = (_lr_raw.abs().mean().item(), _lr_raw.std().item(),
                             _lr_raw.min().item(), _lr_raw.max().item())
        lr_indices = self._build_indices_grid(
            B, T_lr, H_lr, W_lr,
            self.lr_compress_t, self.lr_compress_h, self.lr_compress_w,
            device=latent.device,
            orig_t=T_h, orig_h=H_h, orig_w=W_h,
        )

        assert hr_mem_tokens.shape == lr_mem_tokens.shape, (
            f"HR shape {tuple(hr_mem_tokens.shape)} != LR shape {tuple(lr_mem_tokens.shape)}. "
            f"add requires identical shapes - check that compress and lr_compress match in all three dimensions."
        )
        mem_sum = hr_mem_tokens + lr_mem_tokens
        mem_tokens = mem_sum * self.output_gate
        mem_indices = hr_indices  # after the add there is a single token set; HR and LR share the grid

        if _do_dbg:
            _sum_raw = mem_sum.detach().float()
            _sum_pre_stats = (_sum_raw.abs().mean().item(), _sum_raw.std().item())
            _mem_post = mem_tokens.detach().float()
            _mem_post_stats = (_mem_post.abs().mean().item(), _mem_post.std().item())
            print(
                f"[HistEnc-DBG fwd={self._fwd_count}] output_gate={self.output_gate.item():+.6f} "
                f"| HR raw abs|μ|={_hr_raw_stats[0]:.4f} σ={_hr_raw_stats[1]:.4f} "
                f"min={_hr_raw_stats[2]:+.3f} max={_hr_raw_stats[3]:+.3f} "
                f"| LR raw abs|μ|={_lr_raw_stats[0]:.4f} σ={_lr_raw_stats[1]:.4f} "
                f"min={_lr_raw_stats[2]:+.3f} max={_lr_raw_stats[3]:+.3f} "
                f"| sum(HR+LR) pre-gate abs|μ|={_sum_pre_stats[0]:.4f} σ={_sum_pre_stats[1]:.4f} "
                f"| mem post-gate abs|μ|={_mem_post_stats[0]:.4f} σ={_mem_post_stats[1]:.4f}",
                flush=True,
            )
        return mem_tokens, mem_indices

