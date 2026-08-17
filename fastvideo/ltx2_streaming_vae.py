"""
LTX2 Streaming VAE Encoder
==========================

Reusable streaming VAE encoder: encodes very long videos in chunks to bound peak memory.

Key properties:
---------
1. streaming: a cache mechanism removes the need for overlapping windows
2. numerically equivalent to a single-shot encode (up to tiny floating point error)
3. memory friendly: peak memory is controlled by chunk_size
4. simple API

Choosing chunk_size:
-------------------
- must satisfy chunk_size = 1 + 8*k (k >= 0)
- valid values: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 257, ...
- the first chunk uses chunk_size frames (odd)
- later chunks use chunk_size - 1 frames (even, no space-to-depth padding)
- larger chunk_size: faster encoding, higher peak memory
- smaller chunk_size: lower memory, slightly slower

Reference points at 352x640:
-------------------------
- chunk_size=9:   lowest peak memory
- chunk_size=33:  balanced
- chunk_size=65:  faster
- chunk_size=129: high throughput
- chunk_size=257: maximum throughput (needs a lot of memory)

Example:
--------
```python
from fastvideo.ltx2_streaming_vae import StreamingVAEEncoder

# reuse an already loaded encoder
encoder = StreamingVAEEncoder(
    encoder=vae_encoder,  # a loaded VideoEncoder
    device="cuda:0",
    dtype=torch.bfloat16,
)

# streaming encode (chunk_size chosen automatically)
latent = encoder.encode(video)  # video: [B, C, T, H, W]

# explicit chunk_size
latent = encoder.encode(video, chunk_size=65)
```

Dependencies:
-----
- PyTorch
- einops
- ltx2/modules/vae (local module)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from einops import rearrange

from ltx2.modules.vae import (
    CausalConv3d,
    SpaceToDepthDownsample,
    ResnetBlock3D,
    UNetMidBlock3D,
    LogVarianceType,
    patchify,
    VideoEncoder,
)


# ============================================================================
# Constants
# ============================================================================

# Cache size: number of frames to cache (kernel_size - 1 for kernel_size=3)
CACHE_T = 2

# Valid chunk sizes (1 + 8*k for k >= 1)
VALID_CHUNK_SIZES = [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 
                    105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 
                    185, 193, 201, 209, 217, 225, 233, 241, 249, 257]


# ============================================================================
# Helper Functions
# ============================================================================

def validate_chunk_size(chunk_size: int) -> bool:
    """
    Validate chunk_size.
    
    A valid chunk_size satisfies chunk_size = 1 + 8*k (k >= 1),
    i.e. 9, 17, 25, 33, 41, ...
    """
    if chunk_size < 9:
        return False
    return (chunk_size - 1) % 8 == 0


def get_valid_chunk_sizes(max_size: int = 257) -> List[int]:
    """Return every valid chunk_size."""
    return [1 + 8 * k for k in range(1, (max_size - 1) // 8 + 1)]


def count_causal_conv3d(model: nn.Module) -> int:
    """Count the causal 3D convolutions in the model."""
    return sum(1 for m in model.modules() if isinstance(m, CausalConv3d))


def count_space_to_depth_blocks(model: nn.Module) -> int:
    """Count the space-to-depth downsample blocks in the model."""
    return sum(1 for m in model.modules() if isinstance(m, SpaceToDepthDownsample))


# ============================================================================
# Cache State Management
# ============================================================================

class CacheState:
    """
    Manage the cache state of a streaming encode.
    
    Cache layout:
    - feat_cache: input cache of each causal 3D convolution (the last CACHE_T frames)
    - s2d_cache: cache of the space-to-depth downsample blocks
    """
    
    def __init__(self, num_conv_layers: int, num_s2d_blocks: int):
        self.num_conv_layers = num_conv_layers
        self.num_s2d_blocks = num_s2d_blocks
        self.feat_cache: List[Optional[torch.Tensor]] = [None] * num_conv_layers
        self.s2d_cache: List[Optional[torch.Tensor]] = [None] * num_s2d_blocks
        self.conv_idx = [0]
        self.s2d_idx = [0]
        
    def reset(self):
        """Reset every cache."""
        self.feat_cache = [None] * self.num_conv_layers
        self.s2d_cache = [None] * self.num_s2d_blocks
        self.conv_idx = [0]
        self.s2d_idx = [0]
        
    def reset_index(self):
        """Reset the layer index (called when a new chunk starts)."""
        self.conv_idx = [0]
        self.s2d_idx = [0]


# ============================================================================
# Main Streaming Encoder Class
# ============================================================================

class StreamingVAEEncoder:
    """
    LTX2 VAE streaming encoder.
    
    Encodes long videos chunk by chunk while staying numerically equivalent via caching.
    
    Parameters
    ----------
    device : str or torch.device
        compute device
    dtype : torch.dtype
        dtype, defaults to torch.bfloat16
    encoder : nn.Module
        an already loaded VideoEncoder instance
        
    Attributes
    ----------
    encoder : nn.Module
        the VAE encoder
    device : torch.device
        compute device
    dtype : torch.dtype
        dtype
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        
        if encoder is None:
            raise ValueError("Must provide encoder")
        
        self.encoder = encoder
        self.encoder.eval()
        
        # Count layers for cache management
        self.num_conv_layers = count_causal_conv3d(self.encoder)
        self.num_s2d_blocks = count_space_to_depth_blocks(self.encoder)
        self._state = CacheState(self.num_conv_layers, self.num_s2d_blocks)
        
        # Get encoder attributes
        self.patch_size = self.encoder.patch_size
        self.latent_log_var = self.encoder.latent_log_var
        self.per_channel_statistics = self.encoder.per_channel_statistics
    
    def _clear_cache(self):
        """Clear every cache."""
        self._state.reset()
    
    def _apply_conv_with_cache(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply a causal 3D convolution and maintain its cache."""
        if not isinstance(conv, CausalConv3d):
            return conv(x)
        
        idx = self._state.conv_idx[0]
        cache = self._state.feat_cache[idx]
        time_kernel_size = conv.time_kernel_size
        
        # Save cache BEFORE processing
        if x.shape[2] >= CACHE_T:
            new_cache = x[:, :, -CACHE_T:, :, :].clone()
        else:
            if cache is not None and cache.shape[2] >= 1:
                combined = torch.cat([cache[:, :, -1:, :, :].to(x.device), x], dim=2)
                new_cache = combined[:, :, -CACHE_T:, :, :].clone()
            else:
                new_cache = x.clone()
        
        # Apply convolution with cache or padding
        if cache is not None:
            cache = cache.to(x.device)
            x_padded = torch.cat([cache, x], dim=2)
        else:
            first_frame_pad = x[:, :, :1, :, :].repeat(1, 1, time_kernel_size - 1, 1, 1)
            x_padded = torch.cat([first_frame_pad, x], dim=2)
        
        output = conv.conv(x_padded)
        
        # Update cache
        self._state.feat_cache[idx] = new_cache
        self._state.conv_idx[0] += 1
        
        return output
    
    def _forward_resnet_block(self, block: ResnetBlock3D, x: torch.Tensor) -> torch.Tensor:
        """ResnetBlock3D forward (streaming variant)."""
        hidden = block.norm1(x)
        hidden = block.act(hidden)
        hidden = self._apply_conv_with_cache(block.conv1, hidden)
        hidden = block.norm2(hidden)
        hidden = block.act(hidden)
        hidden = block.dropout(hidden)
        hidden = self._apply_conv_with_cache(block.conv2, hidden)
        
        # shortcut
        if isinstance(block.conv_shortcut, nn.Identity):
            shortcut = x
        else:
            shortcut = block.conv_shortcut(block.norm3(x))
        
        return shortcut + hidden
    
    def _forward_mid_block(self, block: UNetMidBlock3D, x: torch.Tensor) -> torch.Tensor:
        """UNetMidBlock3D forward."""
        for resnet in block.res_blocks:
            x = self._forward_resnet_block(resnet, x)
        return x
    
    def _forward_s2d_block(self, block: SpaceToDepthDownsample, x: torch.Tensor) -> torch.Tensor:
        """SpaceToDepthDownsample forward (streaming variant)."""
        stride = block.stride
        T = x.shape[2]
        
        # Temporal padding for odd frame count (when stride[0] == 2)
        if stride[0] == 2 and T % 2 == 1:
            x_padded = torch.cat([x[:, :, :1, :, :], x], dim=2)
        else:
            x_padded = x
        
        # Skip connection path (space-to-depth + group mean)
        x_in = rearrange(
            x_padded, "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=stride[0], p2=stride[1], p3=stride[2],
        )
        x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=block.group_size)
        x_in = x_in.mean(dim=2)
        
        # Conv path (with cache)
        x_conv = self._apply_conv_with_cache(block.conv, x_padded)
        x_conv = rearrange(
            x_conv, "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=stride[0], p2=stride[1], p3=stride[2],
        )
        
        return x_conv + x_in
    
    def _forward_down_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Downsample block forward."""
        if isinstance(block, UNetMidBlock3D):
            return self._forward_mid_block(block, x)
        elif isinstance(block, ResnetBlock3D):
            return self._forward_resnet_block(block, x)
        elif isinstance(block, SpaceToDepthDownsample):
            return self._forward_s2d_block(block, x)
        elif isinstance(block, CausalConv3d):
            return self._apply_conv_with_cache(block, x)
        else:
            return block(x)
    
    def _process_output(self, sample: torch.Tensor) -> torch.Tensor:
        """Post-process the encoder output and extract the normalized latent."""
        if self.latent_log_var == LogVarianceType.UNIFORM:
            means = sample[:, :-1, ...]
            logvar = sample[:, -1:, ...]
            num_channels = means.shape[1]
            repeat_shape = [1, num_channels] + [1] * (sample.ndim - 2)
            repeated_logvar = logvar.repeat(*repeat_shape)
            sample = torch.cat([means, repeated_logvar], dim=1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30
            sample = torch.cat(
                [sample, torch.ones_like(sample) * approx_ln_0], dim=1
            )
        
        means, _ = torch.chunk(sample, 2, dim=1)
        return self.per_channel_statistics.normalize(means)
    
    def _encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """Encode a single chunk."""
        self._state.reset_index()
        
        # Patchify
        sample = patchify(chunk, self.patch_size, 1)
        
        # Conv in (with cache)
        sample = self._apply_conv_with_cache(self.encoder.conv_in, sample)
        
        # Down blocks
        for down_block in self.encoder.down_blocks:
            sample = self._forward_down_block(down_block, sample)
        
        # Output layers
        sample = self.encoder.conv_norm_out(sample)
        sample = self.encoder.conv_act(sample)
        sample = self._apply_conv_with_cache(self.encoder.conv_out, sample)
        
        return self._process_output(sample)
    
    def encode(
        self,
        video: torch.Tensor,
        chunk_size: int = 65,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Encode a video in streaming fashion.
        
        Parameters
        ----------
        video : torch.Tensor
            input video of shape [B, C, T, H, W] where T = 1 + 8*k
        chunk_size : int
            frames per chunk; must satisfy chunk_size = 1 + 8*k (k >= 1)
            valid values: 9, 17, 25, 33, 41, 49, 57, 65, ...
            - larger is faster but uses more memory
            - suggested: 33 (balanced), 65 (faster), 129 (high throughput)
        verbose : bool
            print progress information
            
        Returns
        -------
        torch.Tensor
            the encoded latent of shape [B, C, T_latent, H_latent, W_latent]
            
        Raises
        ------
        ValueError
            when chunk_size or the frame count is invalid
        """
        if not validate_chunk_size(chunk_size):
            valid_sizes = get_valid_chunk_sizes(chunk_size + 16)[:10]
            raise ValueError(
                f"Invalid chunk_size={chunk_size}. Must be 1 + 8*k (k >= 1). "
                f"Valid values: {valid_sizes}..."
            )
        
        B, C, T, H, W = video.shape
        
        if ((T - 1) % 8) != 0:
            raise ValueError(
                f"Invalid frame count T={T}. Must be 1 + 8*k (e.g., 1, 9, 17, 25, ...)"
            )
        
        expected_latent_T = 1 + (T - 1) // 8
        
        if verbose:
            print(f"[StreamingVAE] Input: {T} frames -> {expected_latent_T} latent frames")
            print(f"[StreamingVAE] chunk_size={chunk_size}")
        
        self._clear_cache()
        latent_chunks = []
        
        # Short video: single pass
        if T <= chunk_size:
            with torch.no_grad():
                return self._encode_chunk(video)
        
        pos = 0
        chunk_idx = 0
        
        with torch.no_grad():
            while pos < T:
                if chunk_idx == 0:
                    # First chunk: use chunk_size (odd)
                    chunk_end = min(chunk_size, T)
                else:
                    # Subsequent chunks: use chunk_size - 1 (even)
                    chunk_end = min(pos + chunk_size - 1, T)
                
                actual_frames = chunk_end - pos
                if actual_frames < 1:
                    break
                
                chunk = video[:, :, pos:chunk_end, :, :]
                
                if verbose:
                    parity = "odd" if actual_frames % 2 == 1 else "even"
                    print(f"  Chunk {chunk_idx}: frames [{pos}:{chunk_end}] ({actual_frames}f, {parity})")
                
                latent_chunk = self._encode_chunk(chunk)
                latent_chunks.append(latent_chunk)
                
                pos = chunk_end
                chunk_idx += 1
        
        latent = torch.cat(latent_chunks, dim=2)
        
        if verbose:
            print(f"[StreamingVAE] Done! Latent shape: {latent.shape}")
        
        return latent
    
    def encode_full(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode in one shot with the original encoder (for comparison).
        
        Parameters
        ----------
        video : torch.Tensor
            input video of shape [B, C, T, H, W]
            
        Returns
        -------
        torch.Tensor
            the encoded latent
        """
        with torch.no_grad():
            return self.encoder(video)
    
    @staticmethod
    def get_recommended_chunk_size(
        num_frames: int,
        target_memory_gb: float = 8.0,
        resolution: Tuple[int, int] = (352, 640),
    ) -> int:
        """
        Suggest a chunk_size from the frame count and a memory budget.
        
        Parameters
        ----------
        num_frames : int
            total number of frames
        target_memory_gb : float
            memory budget in GB
        resolution : Tuple[int, int]
            video resolution (H, W)
            
        Returns
        -------
        int
            the suggested chunk_size
        """
        H, W = resolution
        pixels_per_frame = H * W
        
        # Empirical memory estimation (rough)
        # ~0.1 GB per chunk for 352x640, scales with resolution
        base_memory = 0.1 * (pixels_per_frame / (352 * 640))
        
        # Calculate max chunk size based on target memory
        max_chunk = int(target_memory_gb / base_memory)
        
        # Find nearest valid chunk size
        valid_sizes = get_valid_chunk_sizes(max(max_chunk + 16, 257))
        
        for size in reversed(valid_sizes):
            if size <= max_chunk:
                return size
        
        return 9  # Minimum valid chunk size


# ============================================================================
# Utility Functions
# ============================================================================

def measure_peak_memory(
    encoder: StreamingVAEEncoder,
    video: torch.Tensor,
    chunk_size: int,
) -> float:
    """
    Measure peak memory for a given chunk_size.
    
    Returns
    -------
    float
        peak memory in GB
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    _ = encoder.encode(video, chunk_size=chunk_size, verbose=False)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    torch.cuda.empty_cache()
    
    return peak_memory


def benchmark_chunk_sizes(
    encoder: StreamingVAEEncoder,
    video: torch.Tensor,
    chunk_sizes: Optional[List[int]] = None,
) -> dict:
    """
    Benchmark several chunk_size values.
    
    Parameters
    ----------
    encoder : StreamingVAEEncoder
        streaming encoder instance
    video : torch.Tensor
        test video
    chunk_sizes : List[int], optional
        chunk_size values to test, default [9, 17, 33, 65, 129, 257]
        
    Returns
    -------
    dict
        results per chunk_size
    """
    import time
    
    if chunk_sizes is None:
        chunk_sizes = [9, 17, 33, 65, 129, 257]
    
    results = {}
    
    for chunk_size in chunk_sizes:
        if not validate_chunk_size(chunk_size):
            continue
            
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        try:
            # Warmup
            _ = encoder.encode(video, chunk_size=chunk_size, verbose=False)
            torch.cuda.empty_cache()
            
            # Measure
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            
            _ = encoder.encode(video, chunk_size=chunk_size, verbose=False)
            
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            
            results[chunk_size] = {
                "time_seconds": elapsed,
                "peak_memory_gb": peak_mem,
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results[chunk_size] = {"error": "OOM"}
            else:
                results[chunk_size] = {"error": str(e)}
        
        torch.cuda.empty_cache()
    
    return results
