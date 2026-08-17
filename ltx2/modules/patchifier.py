"""
LTX-2 Patchifier Module (WAN-style)
Patchify / unpatchify helpers for video and audio latents.
"""
import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import einops
import torch


# =============================================================================
# =============================================================================

class SpatioTemporalScaleFactors(NamedTuple):
    """Spatio-temporal downsampling factors."""
    time: int
    width: int
    height: int
    
    @classmethod
    def default(cls) -> "SpatioTemporalScaleFactors":
        return cls(time=8, width=32, height=32)


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class VideoLatentShape(NamedTuple):
    """Video latent shape: (batch, channels, frames, height, width)."""
    batch: int
    channels: int
    frames: int
    height: int
    width: int
    
    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.height, self.width])
    
    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "VideoLatentShape":
        return VideoLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            height=shape[3],
            width=shape[4],
        )
    
    def mask_shape(self) -> "VideoLatentShape":
        return self._replace(channels=1)


class AudioLatentShape(NamedTuple):
    """Audio latent shape: (batch, channels, frames, mel_bins)."""
    batch: int
    channels: int
    frames: int
    mel_bins: int
    
    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.mel_bins])
    
    def mask_shape(self) -> "AudioLatentShape":
        return self._replace(channels=1, mel_bins=1)
    
    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "AudioLatentShape":
        return AudioLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            mel_bins=shape[3],
        )
    
    @staticmethod
    def from_duration(
        batch: int,
        duration: float,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        latents_per_second = float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)
        return AudioLatentShape(
            batch=batch,
            channels=channels,
            frames=round(duration * latents_per_second),
            mel_bins=mel_bins,
        )


@dataclass(frozen=True)
class LatentState:
    """Latent state during diffusion."""
    latent: torch.Tensor
    denoise_mask: torch.Tensor
    positions: torch.Tensor
    clean_latent: torch.Tensor
    
    def clone(self) -> "LatentState":
        return LatentState(
            latent=self.latent.clone(),
            denoise_mask=self.denoise_mask.clone(),
            positions=self.positions.clone(),
            clean_latent=self.clean_latent.clone(),
        )


# =============================================================================
# Video Patchifier
# =============================================================================

class VideoLatentPatchifier:
    """Video latent patchifier."""
    
    def __init__(self, patch_size: int = 1):
        self._patch_size = (1, patch_size, patch_size)
    
    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size
    
    def get_token_count(self, tgt_shape: VideoLatentShape) -> int:
        return math.prod(tgt_shape.to_torch_shape()[2:]) // math.prod(self._patch_size)
    
    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert [B, C, F, H, W] into [B, F*H*W, C*p1*p2*p3]."""
        latents = einops.rearrange(
            latents,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=self._patch_size[0],
            p2=self._patch_size[1],
            p3=self._patch_size[2],
        )
        return latents
    
    def unpatchify(self, latents: torch.Tensor, output_shape: VideoLatentShape) -> torch.Tensor:
        """Convert [B, F*H*W, C*p*q] back into [B, C, F, H, W]."""
        assert self._patch_size[0] == 1, "Temporal patch size must be 1"
        
        patch_grid_frames = output_shape.frames // self._patch_size[0]
        patch_grid_height = output_shape.height // self._patch_size[1]
        patch_grid_width = output_shape.width // self._patch_size[2]
        
        latents = einops.rearrange(
            latents,
            "b (f h w) (c p q) -> b c f (h p) (w q)",
            f=patch_grid_frames,
            h=patch_grid_height,
            w=patch_grid_width,
            p=self._patch_size[1],
            q=self._patch_size[2],
        )
        return latents
    
    def get_patch_grid_bounds(
        self,
        output_shape: VideoLatentShape,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return the boundary coordinates of every patch: [batch, 3, num_patches, 2]."""
        frames = output_shape.frames
        height = output_shape.height
        width = output_shape.width
        batch_size = output_shape.batch
        
        grid_coords = torch.meshgrid(
            torch.arange(start=0, end=frames, step=self._patch_size[0], device=device),
            torch.arange(start=0, end=height, step=self._patch_size[1], device=device),
            torch.arange(start=0, end=width, step=self._patch_size[2], device=device),
            indexing="ij",
        )
        
        patch_starts = torch.stack(grid_coords, dim=0)
        patch_size_delta = torch.tensor(
            self._patch_size,
            device=patch_starts.device,
            dtype=patch_starts.dtype,
        ).view(3, 1, 1, 1)
        
        patch_ends = patch_starts + patch_size_delta
        latent_coords = torch.stack((patch_starts, patch_ends), dim=-1)
        
        latent_coords = einops.repeat(
            latent_coords,
            "c f h w bounds -> b c (f h w) bounds",
            b=batch_size,
            bounds=2,
        )
        return latent_coords


# =============================================================================
# Audio Patchifier
# =============================================================================

class AudioPatchifier:
    """Audio latent patchifier."""
    
    def __init__(
        self,
        patch_size: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
        shift: int = 0,
    ):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.shift = shift
        self._patch_size = (1, patch_size, patch_size)
    
    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size
    
    def get_token_count(self, tgt_shape: AudioLatentShape) -> int:
        return tgt_shape.frames
    
    def _get_audio_latent_time_in_sec(
        self,
        start_latent: int,
        end_latent: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = torch.device("cpu")
        
        audio_latent_frame = torch.arange(start_latent, end_latent, dtype=dtype, device=device)
        audio_mel_frame = audio_latent_frame * self.audio_latent_downsample_factor
        
        if self.is_causal:
            causal_offset = 1
            audio_mel_frame = (audio_mel_frame + causal_offset - self.audio_latent_downsample_factor).clip(min=0)
        
        return audio_mel_frame * self.hop_length / self.sample_rate
    
    def _compute_audio_timings(
        self,
        batch_size: int,
        num_steps: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        resolved_device = device or torch.device("cpu")
        
        start_timings = self._get_audio_latent_time_in_sec(
            self.shift, num_steps + self.shift, torch.float32, resolved_device
        )
        start_timings = start_timings.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        
        end_timings = self._get_audio_latent_time_in_sec(
            self.shift + 1, num_steps + self.shift + 1, torch.float32, resolved_device
        )
        end_timings = end_timings.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
        
        return torch.stack([start_timings, end_timings], dim=-1)
    
    def patchify(self, audio_latents: torch.Tensor) -> torch.Tensor:
        """Convert [B, C, T, F] into [B, T, C*F]."""
        audio_latents = einops.rearrange(audio_latents, "b c t f -> b t (c f)")
        return audio_latents
    
    def unpatchify(self, audio_latents: torch.Tensor, output_shape: AudioLatentShape) -> torch.Tensor:
        """Convert [B, T, C*F] back into [B, C, T, F]."""
        audio_latents = einops.rearrange(
            audio_latents,
            "b t (c f) -> b c t f",
            c=output_shape.channels,
            f=output_shape.mel_bins,
        )
        return audio_latents
    
    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return the temporal boundaries of the audio patches: [batch, 1, time_steps, 2]."""
        return self._compute_audio_timings(output_shape.batch, output_shape.frames, device)


# =============================================================================
# =============================================================================

def get_pixel_coords(
    latent_coords: torch.Tensor,
    scale_factors: SpatioTemporalScaleFactors,
    causal_fix: bool = False,
) -> torch.Tensor:
    """
    Convert latent coordinates into pixel coordinates.
    
    Args:
        latent_coords: [batch, 3, num_patches, 2] latent boundaries
        scale_factors: spatio-temporal scale factors
        causal_fix: correct for causal encoding
    
    Returns:
        [batch, 3, num_patches, 2] pixel coordinates
    """
    broadcast_shape = [1] * latent_coords.ndim
    broadcast_shape[1] = -1
    scale_tensor = torch.tensor(scale_factors, device=latent_coords.device).view(*broadcast_shape)
    
    pixel_coords = latent_coords * scale_tensor
    
    if causal_fix:
        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + 1 - scale_factors[0]).clamp(min=0)
    
    return pixel_coords
