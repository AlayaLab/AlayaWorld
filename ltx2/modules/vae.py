"""
Self-contained VAE implementation adapted from the public LTX-2 code.
"""
import math
from enum import Enum
from typing import Optional, Tuple, Union, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ['VideoEncoder', 'VideoDecoder']


# ============================================================
# ============================================================

class NormLayerType(Enum):
    GROUP_NORM = "group_norm"
    PIXEL_NORM = "pixel_norm"


class LogVarianceType(Enum):
    NONE = "none"
    UNIFORM = "uniform"
    PER_CHANNEL = "per_channel"
    CONSTANT = "constant"


class PaddingModeType(Enum):
    ZEROS = "zeros"
    REFLECT = "reflect"


# ============================================================
# ============================================================

class PixelNorm(nn.Module):
    """Per-pixel RMS normalization"""
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim, self.eps = dim, eps
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=self.dim, keepdim=True) + self.eps)


class PerChannelStatistics(nn.Module):
    """Per-channel statistics for latent normalization"""
    def __init__(self, latent_channels=128):
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))
    
    def normalize(self, x):
        return (x - self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)) / self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)
    
    def un_normalize(self, x):
        return x * self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x) + self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)


def patchify(x, patch_size_hw, patch_size_t=1):
    """Space-to-depth: (B,C,F,H,W) -> (B,C*p^2,F,H/p,W/p)"""
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    return rearrange(x, "b c (f p) (h q) (w r) -> b (c p r q) f h w", p=patch_size_t, q=patch_size_hw, r=patch_size_hw)


def unpatchify(x, patch_size_hw, patch_size_t=1):
    """Depth-to-space: inverse of patchify"""
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    return rearrange(x, "b (c p r q) f h w -> b c (f p) (h q) (w r)", p=patch_size_t, q=patch_size_hw, r=patch_size_hw)


# ============================================================
# ============================================================

def get_timestep_embedding(timesteps, dim, flip_sin_to_cos=False, downscale_freq_shift=1, scale=1, max_period=10000):
    half = dim // 2
    exp = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / (half - downscale_freq_shift)
    emb = timesteps[:, None].float() * torch.exp(exp)[None, :]
    emb = torch.cat([torch.sin(emb * scale), torch.cos(emb * scale)], -1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half:], emb[:, :half]], -1)
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_ch, time_dim, out_dim=None):
        super().__init__()
        self.linear_1 = nn.Linear(in_ch, time_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_dim, out_dim or time_dim)
    
    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim, size_emb_dim=0):
        super().__init__()
        self.time_proj = lambda t: get_timestep_embedding(t, 256, True, 0)
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
    
    def forward(self, timestep, hidden_dtype):
        return self.timestep_embedder(self.time_proj(timestep).to(hidden_dtype))


# ============================================================
# ============================================================

class CausalConv3d(nn.Module):
    """Causal 3D convolution: the temporal dimension only sees past frames."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=True,
                 spatial_padding_mode=PaddingModeType.ZEROS):
        super().__init__()
        self.time_kernel_size = kernel_size
        height_pad, width_pad = kernel_size // 2, kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size),
                              stride=stride, dilation=(dilation, 1, 1), padding=(0, height_pad, width_pad),
                              padding_mode=spatial_padding_mode.value, groups=groups, bias=bias)
    
    def forward(self, x, causal=True):
        if causal:
            pad = x[:, :, :1].repeat(1, 1, self.time_kernel_size - 1, 1, 1)
            x = torch.cat([pad, x], dim=2)
        else:
            pad_size = (self.time_kernel_size - 1) // 2
            x = torch.cat([x[:, :, :1].repeat(1, 1, pad_size, 1, 1), x, x[:, :, -1:].repeat(1, 1, pad_size, 1, 1)], dim=2)
        return self.conv(x)


def make_conv_nd(dims, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 causal=False, spatial_padding_mode=PaddingModeType.ZEROS, **kwargs):
    if dims == 3 and causal:
        return CausalConv3d(in_channels, out_channels, kernel_size, stride, dilation, groups, bias, spatial_padding_mode)
    elif dims == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                         padding_mode=spatial_padding_mode.value)
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                     padding_mode=spatial_padding_mode.value)


def make_linear_nd(dims, in_channels, out_channels, bias=True):
    return nn.Conv3d(in_channels, out_channels, 1, bias=bias) if dims == 3 else nn.Conv2d(in_channels, out_channels, 1, bias=bias)


# ============================================================
# ============================================================

class SpaceToDepthDownsample(nn.Module):
    """Downsample with space-to-depth followed by convolution."""
    def __init__(self, dims, in_channels, out_channels, stride, spatial_padding_mode=PaddingModeType.ZEROS):
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels
        self.conv = make_conv_nd(dims, in_channels, out_channels // math.prod(stride), 3, 1, causal=True, spatial_padding_mode=spatial_padding_mode)
    
    def forward(self, x, causal=True):
        if self.stride[0] == 2:
            x = torch.cat([x[:, :, :1], x], dim=2)
        x_in = rearrange(x, "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w", p1=self.stride[0], p2=self.stride[1], p3=self.stride[2])
        x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=self.group_size).mean(dim=2)
        x = self.conv(x, causal=causal)
        x = rearrange(x, "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w", p1=self.stride[0], p2=self.stride[1], p3=self.stride[2])
        return x + x_in


class DepthToSpaceUpsample(nn.Module):
    """Upsample with convolution followed by depth-to-space."""
    def __init__(self, dims, in_channels, stride, residual=False, out_channels_reduction_factor=1, spatial_padding_mode=PaddingModeType.ZEROS):
        super().__init__()
        self.stride = stride
        self.out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.conv = make_conv_nd(dims, in_channels, self.out_channels, 3, 1, causal=True, spatial_padding_mode=spatial_padding_mode)
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor
    
    def forward(self, x, causal=True):
        if self.residual:
            x_in = rearrange(x, "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)", p1=self.stride[0], p2=self.stride[1], p3=self.stride[2])
            x_in = x_in.repeat(1, math.prod(self.stride) // self.out_channels_reduction_factor, 1, 1, 1)
            if self.stride[0] == 2:
                x_in = x_in[:, :, 1:]
        x = self.conv(x, causal=causal)
        x = rearrange(x, "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)", p1=self.stride[0], p2=self.stride[1], p3=self.stride[2])
        if self.stride[0] == 2:
            x = x[:, :, 1:]
        return x + x_in if self.residual else x


# ============================================================
# ============================================================

class ResnetBlock3D(nn.Module):
    """3D residual block."""
    def __init__(self, dims, in_channels, out_channels=None, dropout=0.0, groups=32, eps=1e-6,
                 norm_layer=NormLayerType.PIXEL_NORM, inject_noise=False, timestep_conditioning=False,
                 spatial_padding_mode=PaddingModeType.ZEROS):
        super().__init__()
        out_channels = out_channels or in_channels
        self.inject_noise = inject_noise
        self.timestep_conditioning = timestep_conditioning
        
        self.norm1 = nn.GroupNorm(groups, in_channels, eps) if norm_layer == NormLayerType.GROUP_NORM else PixelNorm()
        self.conv1 = make_conv_nd(dims, in_channels, out_channels, 3, 1, 1, causal=True, spatial_padding_mode=spatial_padding_mode)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps) if norm_layer == NormLayerType.GROUP_NORM else PixelNorm()
        self.conv2 = make_conv_nd(dims, out_channels, out_channels, 3, 1, 1, causal=True, spatial_padding_mode=spatial_padding_mode)
        self.conv_shortcut = make_linear_nd(dims, in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.norm3 = nn.GroupNorm(1, in_channels, eps) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros(in_channels, 1, 1))
            self.per_channel_scale2 = nn.Parameter(torch.zeros(in_channels, 1, 1))
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels**0.5)
    
    def forward(self, x, causal=True, timestep=None, generator=None):
        h = self.norm1(x)
        if self.timestep_conditioning and timestep is not None:
            ada = self.scale_shift_table[None, ..., None, None, None].to(h) + timestep.reshape(x.shape[0], 4, -1, *timestep.shape[-3:])
            shift1, scale1, shift2, scale2 = ada.unbind(1)
            h = h * (1 + scale1) + shift1
        h = self.act(h)
        h = self.conv1(h, causal=causal)
        if self.inject_noise:
            noise = torch.randn(h.shape[-2:], device=h.device, dtype=h.dtype, generator=generator)[None]
            h = h + (noise * self.per_channel_scale1.to(h))[None, :, None]
        h = self.norm2(h)
        if self.timestep_conditioning and timestep is not None:
            h = h * (1 + scale2) + shift2
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h, causal=causal)
        if self.inject_noise:
            noise = torch.randn(h.shape[-2:], device=h.device, dtype=h.dtype, generator=generator)[None]
            h = h + (noise * self.per_channel_scale2.to(h))[None, :, None]
        return self.conv_shortcut(self.norm3(x)) + h


class UNetMidBlock3D(nn.Module):
    """UNet middle block."""
    def __init__(self, dims, in_channels, dropout=0.0, num_layers=1, resnet_eps=1e-6, resnet_groups=32,
                 norm_layer=NormLayerType.GROUP_NORM, inject_noise=False, timestep_conditioning=False,
                 attention_head_dim=None, spatial_padding_mode=PaddingModeType.ZEROS):
        super().__init__()
        self.timestep_conditioning = timestep_conditioning
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)
        self.res_blocks = nn.ModuleList([
            ResnetBlock3D(dims, in_channels, in_channels, dropout, resnet_groups, resnet_eps, norm_layer,
                         inject_noise, timestep_conditioning, spatial_padding_mode)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, causal=True, timestep=None, generator=None):
        ts_embed = None
        if self.timestep_conditioning and timestep is not None:
            ts_embed = self.time_embedder(timestep.flatten(), x.dtype).view(x.shape[0], -1, 1, 1, 1)
        for block in self.res_blocks:
            x = block(x, causal, ts_embed, generator)
        return x


# ============================================================
# ============================================================

def _make_encoder_block(name, config, in_ch, dims, norm, groups, padding_mode):
    out_ch = in_ch
    if name == "res_x":
        return UNetMidBlock3D(dims, in_ch, num_layers=config.get("num_layers", 1), resnet_groups=groups, norm_layer=norm, spatial_padding_mode=padding_mode), out_ch
    elif name == "res_x_y":
        out_ch = in_ch * config.get("multiplier", 2)
        return ResnetBlock3D(dims, in_ch, out_ch, groups=groups, norm_layer=norm, spatial_padding_mode=padding_mode), out_ch
    elif name == "compress_time":
        return make_conv_nd(dims, in_ch, out_ch, 3, (2,1,1), causal=True, spatial_padding_mode=padding_mode), out_ch
    elif name == "compress_space":
        return make_conv_nd(dims, in_ch, out_ch, 3, (1,2,2), causal=True, spatial_padding_mode=padding_mode), out_ch
    elif name == "compress_all":
        return make_conv_nd(dims, in_ch, out_ch, 3, (2,2,2), causal=True, spatial_padding_mode=padding_mode), out_ch
    elif name == "compress_all_res":
        out_ch = in_ch * config.get("multiplier", 2)
        return SpaceToDepthDownsample(dims, in_ch, out_ch, (2,2,2), padding_mode), out_ch
    elif name == "compress_space_res":
        out_ch = in_ch * config.get("multiplier", 2)
        return SpaceToDepthDownsample(dims, in_ch, out_ch, (1,2,2), padding_mode), out_ch
    elif name == "compress_time_res":
        out_ch = in_ch * config.get("multiplier", 2)
        return SpaceToDepthDownsample(dims, in_ch, out_ch, (2,1,1), padding_mode), out_ch
    raise ValueError(f"Unknown encoder block: {name}")


def _make_decoder_block(name, config, in_ch, dims, norm, ts_cond, groups, padding_mode):
    out_ch = in_ch
    if name == "res_x":
        return UNetMidBlock3D(dims, in_ch, num_layers=config.get("num_layers", 1), resnet_groups=groups, norm_layer=norm,
                             inject_noise=config.get("inject_noise", False), timestep_conditioning=ts_cond, spatial_padding_mode=padding_mode), out_ch
    elif name == "attn_res_x":
        return UNetMidBlock3D(dims, in_ch, num_layers=config.get("num_layers", 1), resnet_groups=groups, norm_layer=norm,
                             inject_noise=config.get("inject_noise", False), timestep_conditioning=ts_cond,
                             attention_head_dim=config.get("attention_head_dim"), spatial_padding_mode=padding_mode), out_ch
    elif name == "res_x_y":
        out_ch = in_ch // config.get("multiplier", 2)
        return ResnetBlock3D(dims, in_ch, out_ch, groups=groups, norm_layer=norm, inject_noise=config.get("inject_noise", False), spatial_padding_mode=padding_mode), out_ch
    elif name == "compress_time":
        mult = config.get("multiplier", 1)
        out_ch = in_ch // mult
        return DepthToSpaceUpsample(dims, in_ch, (2,1,1), out_channels_reduction_factor=mult, spatial_padding_mode=padding_mode), out_ch
    elif name == "compress_space":
        mult = config.get("multiplier", 1)
        out_ch = in_ch // mult
        return DepthToSpaceUpsample(dims, in_ch, (1,2,2), out_channels_reduction_factor=mult, spatial_padding_mode=padding_mode), out_ch
    elif name == "compress_all":
        mult = config.get("multiplier", 1)
        out_ch = in_ch // mult
        return DepthToSpaceUpsample(dims, in_ch, (2,2,2), config.get("residual", False), mult, padding_mode), out_ch
    raise ValueError(f"Unknown decoder block: {name}")


# ============================================================
# VideoEncoder
# ============================================================

class VideoEncoder(nn.Module):
    """
    Video VAE encoder.
    Input: (B, 3, F, H, W) where F must be 1 + 8*k
    Output: (B, 128, F', H', W') where F' = 1+(F-1)/8, H' = H/32, W' = W/32
    """
    def __init__(self, convolution_dimensions=3, in_channels=3, out_channels=128, encoder_blocks=None, patch_size=4,
                 norm_layer=NormLayerType.PIXEL_NORM, latent_log_var=LogVarianceType.UNIFORM,
                 encoder_spatial_padding_mode=PaddingModeType.ZEROS, base_channels=None):
        super().__init__()
        encoder_blocks = encoder_blocks or []
        self.patch_size = patch_size
        self.latent_log_var = latent_log_var
        self.per_channel_statistics = PerChannelStatistics(out_channels)

        feature_ch = base_channels if base_channels is not None else out_channels
        self.conv_in = make_conv_nd(convolution_dimensions, in_channels * patch_size**2, feature_ch, 3, 1, 1, causal=True, spatial_padding_mode=encoder_spatial_padding_mode)
        
        self.down_blocks = nn.ModuleList()
        for name, params in encoder_blocks:
            cfg = {"num_layers": params} if isinstance(params, int) else params
            block, feature_ch = _make_encoder_block(name, cfg, feature_ch, convolution_dimensions, norm_layer, 32, encoder_spatial_padding_mode)
            self.down_blocks.append(block)
        
        self.conv_norm_out = nn.GroupNorm(32, feature_ch, 1e-6) if norm_layer == NormLayerType.GROUP_NORM else PixelNorm()
        self.conv_act = nn.SiLU()
        conv_out_ch = out_channels + (1 if latent_log_var in (LogVarianceType.UNIFORM, LogVarianceType.CONSTANT) else (out_channels if latent_log_var == LogVarianceType.PER_CHANNEL else 0))
        self.conv_out = make_conv_nd(convolution_dimensions, feature_ch, conv_out_ch, 3, 1, 1, causal=True, spatial_padding_mode=encoder_spatial_padding_mode)
    
    def forward(self, x, return_posterior: bool = False):
        """
        Forward pass.
        
        Args:
            x: input video [B, C, F, H, W]
            return_posterior: also return the posterior distribution (for VAE training)
            
        Returns:
            when return_posterior=False: the normalized latent mean
            when return_posterior=True: (latent_mean, posterior_params, raw_mean)
                - latent_mean: normalized latent [B, C, F', H', W']
                - posterior_params: posterior parameters [B, 2*C, F', H', W'] (mean and logvar concatenated)
                - raw_mean: un-normalized mean [B, C, F', H', W']
        """
        x = patchify(x, self.patch_size, 1)
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.conv_act(self.conv_norm_out(x))
        x = self.conv_out(x)
        
        if self.latent_log_var == LogVarianceType.UNIFORM:
            means, logvar = x[:, :-1], x[:, -1:]
            x = torch.cat([means, logvar.repeat(1, means.shape[1], 1, 1, 1)], 1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            x = torch.cat([x[:, :-1], torch.full_like(x[:, :-1], -30)], 1)
        
        means, logvar = x.chunk(2, 1)
        normalized_means = self.per_channel_statistics.normalize(means)
        
        if return_posterior:
            return normalized_means, x, means
        
        return normalized_means
    
    def encode_with_posterior(self, x):
        """
        Encode and return the posterior distribution (for VAE training).
        
        Args:
            x: input video [B, C, F, H, W]
            
        Returns:
            (posterior, z, normalized_z)
                - posterior: the diagonal Gaussian posterior
                - z: latent sampled from the posterior (un-normalized)
                - normalized_z: the normalized latent
        """
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        
        x = patchify(x, self.patch_size, 1)
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.conv_act(self.conv_norm_out(x))
        x = self.conv_out(x)
        
        if self.latent_log_var == LogVarianceType.UNIFORM:
            means, logvar = x[:, :-1], x[:, -1:]
            posterior_params = torch.cat([means, logvar.repeat(1, means.shape[1], 1, 1, 1)], 1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            posterior_params = torch.cat([x[:, :-1], torch.full_like(x[:, :-1], -30)], 1)
        else:
            posterior_params = x
        
        posterior = DiagonalGaussianDistribution(posterior_params)
        
        z = posterior.sample()
        
        normalized_z = self.per_channel_statistics.normalize(z)
        
        return posterior, z, normalized_z


# ============================================================
# VideoDecoder
# ============================================================

class VideoDecoder(nn.Module):
    """
    Video VAE decoder.
    Input: (B, 128, F', H', W')
    Output: (B, 3, F, H, W) where F = 8*(F'-1)+1, H = 32*H', W = 32*W'
    """
    def __init__(self, convolution_dimensions=3, in_channels=128, out_channels=3, decoder_blocks=None, patch_size=4,
                 norm_layer=NormLayerType.PIXEL_NORM, causal=False, timestep_conditioning=False,
                 decoder_spatial_padding_mode=PaddingModeType.REFLECT, base_channels=None):
        super().__init__()
        decoder_blocks = decoder_blocks or []
        self.patch_size = patch_size
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self.per_channel_statistics = PerChannelStatistics(in_channels)
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        if base_channels is not None:
            feature_ch = base_channels * 8
        else:
            feature_ch = in_channels
            for name, params in reversed(decoder_blocks):
                cfg = params if isinstance(params, dict) else {}
                if name == "res_x_y":
                    feature_ch *= cfg.get("multiplier", 2)
                if name == "compress_all":
                    feature_ch *= cfg.get("multiplier", 1)

        self.conv_in = make_conv_nd(convolution_dimensions, in_channels, feature_ch, 3, 1, 1, causal=True, spatial_padding_mode=decoder_spatial_padding_mode)
        
        self.up_blocks = nn.ModuleList()
        for name, params in reversed(decoder_blocks):
            cfg = {"num_layers": params} if isinstance(params, int) else params
            block, feature_ch = _make_decoder_block(name, cfg, feature_ch, convolution_dimensions, norm_layer, timestep_conditioning, 32, decoder_spatial_padding_mode)
            self.up_blocks.append(block)
        
        self.conv_norm_out = nn.GroupNorm(32, feature_ch, 1e-6) if norm_layer == NormLayerType.GROUP_NORM else PixelNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(convolution_dimensions, feature_ch, out_channels * patch_size**2, 3, 1, 1, causal=True, spatial_padding_mode=decoder_spatial_padding_mode)
        
        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0))
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(feature_ch * 2, 0)
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, feature_ch))
    
    def forward(self, x, timestep=None, generator=None):
        bs = x.shape[0]
        if self.timestep_conditioning:
            noise = torch.randn(x.size(), generator=generator, dtype=x.dtype, device=x.device) * self.decode_noise_scale
            x = noise + (1 - self.decode_noise_scale) * x
        
        x = self.per_channel_statistics.un_normalize(x)
        
        if timestep is None and self.timestep_conditioning:
            timestep = torch.full((bs,), self.decode_timestep, device=x.device, dtype=x.dtype)
        
        x = self.conv_in(x, causal=self.causal)
        scaled_ts = timestep * self.timestep_scale_multiplier.to(x) if self.timestep_conditioning else None
        
        for block in self.up_blocks:
            if isinstance(block, UNetMidBlock3D):
                x = block(x, self.causal, scaled_ts if self.timestep_conditioning else None, generator)
            elif isinstance(block, ResnetBlock3D):
                x = block(x, self.causal, generator=generator)
            else:
                x = block(x, self.causal)
        
        x = self.conv_norm_out(x)
        
        if self.timestep_conditioning:
            ts_embed = self.last_time_embedder(scaled_ts.flatten(), x.dtype).view(bs, -1, 1, 1, 1)
            ada = self.last_scale_shift_table[None, ..., None, None, None].to(x) + ts_embed.reshape(bs, 2, -1, 1, 1, 1)
            shift, scale = ada.unbind(1)
            x = x * (1 + scale) + shift
        
        x = self.conv_act(x)
        x = self.conv_out(x, causal=self.causal)
        return unpatchify(x, self.patch_size, 1)


# ============================================================
# ============================================================

def create_video_encoder(config):
    """Build a VideoEncoder from a config."""
    cfg = config.get("vae", {})
    base_channels = cfg.get("encoder_base_channels", None)
    padding_mode = cfg.get("encoder_spatial_padding_mode", cfg.get("spatial_padding_mode", "zeros"))
    return VideoEncoder(
        convolution_dimensions=cfg.get("dims", 3),
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg.get("latent_channels", 128),
        encoder_blocks=cfg.get("encoder_blocks", []),
        patch_size=cfg.get("patch_size", 4),
        norm_layer=NormLayerType(cfg.get("norm_layer", "pixel_norm")),
        latent_log_var=LogVarianceType(cfg.get("latent_log_var", "uniform")),
        encoder_spatial_padding_mode=PaddingModeType(padding_mode),
        base_channels=base_channels,
    )


def create_video_decoder(config):
    """Build a VideoDecoder from a config."""
    cfg = config.get("vae", {})
    base_channels = cfg.get("decoder_base_channels", None)
    padding_mode = cfg.get("decoder_spatial_padding_mode", cfg.get("spatial_padding_mode", "reflect"))
    return VideoDecoder(
        convolution_dimensions=cfg.get("dims", 3),
        in_channels=cfg.get("latent_channels", 128),
        out_channels=cfg.get("out_channels", 3),
        decoder_blocks=cfg.get("decoder_blocks", []),
        patch_size=cfg.get("patch_size", 4),
        norm_layer=NormLayerType(cfg.get("norm_layer", "pixel_norm")),
        causal=cfg.get("causal_decoder", False),
        timestep_conditioning=cfg.get("timestep_conditioning", False),
        decoder_spatial_padding_mode=PaddingModeType(padding_mode),
        base_channels=base_channels,
    )


# ============================================================
# ============================================================

class LTX2VAE(nn.Module):
    """High-level LTX-2 VAE wrapper for the inference pipeline."""
    
    SCALE_FACTORS = (8, 32, 32)  # temporal, height, width
    
    def __init__(self, device=None, dtype=torch.bfloat16):
        super().__init__()
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._encoder = None
        self._decoder = None
        self.scale_factors = self.SCALE_FACTORS
    
    @property
    def device(self):
        return self._device
    
    def set_encoder(self, encoder):
        """Set up the encoder."""
        self._encoder = encoder
    
    def set_decoder(self, decoder):
        """Set up the decoder."""
        self._decoder = decoder
    
    def encode(self, pixel):
        """Encode pixels into latents."""
        if self._encoder is None:
            raise RuntimeError("Encoder not set")
        # [B, F, C, H, W] -> [B, C, F, H, W]
        if pixel.dim() == 5 and pixel.shape[2] == 3:
            pixel = pixel.permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            latent = self._encoder(pixel.to(self.dtype))
        # [B, C, F, H, W] -> [B, F, C, H, W]
        return latent.permute(0, 2, 1, 3, 4)
    
    def decode(self, latent, generator=None):
        """Decode latents into pixels."""
        if self._decoder is None:
            raise RuntimeError("Decoder not set")
        # [B, F, C, H, W] -> [B, C, F, H, W]
        if latent.dim() == 5 and latent.shape[2] != 3:
            latent = latent.permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            pixel = self._decoder(latent.to(self.dtype))
        # [B, C, F, H, W] -> [B, F, C, H, W]
        return pixel.permute(0, 2, 1, 3, 4)
