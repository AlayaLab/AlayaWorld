"""
LTX2 Camera Control Module

Provides optional camera control injection for video generation:
1. Continuous signal (Plucker embedding):
   - scale_shift mode: Per-block scale/shift modulation (ref: lingbot-world)
   - additive mode: Add camera features after patch embedding
2. Discrete signal (Action labels):
   - 81-class action space (9 translations x 9 rotations, ref: HunyuanVideo hycam)
   - Injected via timestep embedding (adaln_single)
3. Training dropout (10%) for both signals (distributed-safe)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple
import numpy as np

_LTX_QUIET = os.environ.get('LTX_QUIET_INFERENCE', '0') == '1'

_DIAG_DETAIL = 0
_DIAG_INTERVAL = 1000000
_prepare_call_cnt = 0
_adapter_call_cnt = 0


# =============================================================================
# Utility Functions
# =============================================================================

def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """1D sinusoidal positional encoding (ref: HunyuanVideo).
    Supports [N] -> [N, dim] and [B, T] -> [B, T, dim]."""
    orig_shape = position.shape
    position_flat = position.flatten().type(torch.float64)
    freqs = torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2))
    sinusoid = torch.outer(position_flat, freqs)
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=-1)
    x = x.view(*orig_shape, dim)
    return x.to(position.dtype)


import random as _random

# Shared RNG for FSDP-safe dropout decisions (no NCCL communication)
# Must be initialized via init_shared_dropout_rng() before each training step
_shared_dropout_rng: Optional[_random.Random] = None


def init_shared_dropout_rng(seed: int):
    """Initialize shared dropout RNG with a seed that's consistent across all ranks.

    Call this BEFORE entering the FSDP-wrapped model forward, using a seed
    that has been broadcast from rank 0 to all ranks.
    """
    global _shared_dropout_rng
    _shared_dropout_rng = _random.Random(seed)


def distributed_safe_dropout_decision(dropout_prob: float, device: torch.device) -> bool:
    """FSDP-safe dropout decision. All ranks get the same result without communication.

    WARNING: Do NOT use dist.broadcast inside FSDP forward pass!
    NCCL does not support concurrent operations on the same communicator from
    different CUDA streams (FSDP comm stream vs compute stream), causing hangs.

    Instead, uses a shared RNG initialized with a synchronized seed before
    entering the model forward.
    """
    global _shared_dropout_rng
    if _shared_dropout_rng is not None:
        return _shared_dropout_rng.random() < dropout_prob
    # Fallback: no dropout (safe default if RNG not initialized)
    return False


def custom_meshgrid(*args):
    """torch.meshgrid with 'ij' indexing (torch>=2.0.0)."""
    return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device):
    """Plucker embedding computation (from wan_video_camera_controller.py / CameraCtrl).

    Args:
        K: [B, V, 4] intrinsic vectors [fx, fy, cx, cy] in pixel space
        c2w: [B, V, 4, 4] camera-to-world matrices
        H: pixel height
        W: pixel width
        device: torch device

    Returns:
        plucker: [B, V, H, W, 6] Plucker coordinates (o×d, d)
    """
    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B, V, 1

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    return plucker


class _ResidualBlock(nn.Module):
    """Residual block for WanCameraAdapter (same as Wan's CameraControlAdapter)."""
    def __init__(self, dim, zero_init: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        if zero_init:
            nn.init.zeros_(self.conv1.weight)
            nn.init.zeros_(self.conv2.weight)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class WanCameraAdapter(nn.Module):
    """Pixel-resolution Plucker -> token-level camera features.

    Aligned with VideoX-Fun pipeline architecture:
        Temporal PixelUnshuffle → Spatial PixelUnshuffle → strided Conv2d → ResidualBlock → output_scale

    Time dimension handling (VideoX-Fun convention):
        - Input: [B, 6, F_pixel, H, W] at full pixel frame rate
        - Temporal unshuffle: repeat first frame S times, then reshape
          [B, 6, F_pixel, H, W] → [B, 6*S, F_latent, H, W]
        - This is LOSSLESS: no interpolation, all pixel-frame camera info preserved

    For LTX2 (VAE temporal_stride=8, spatial_stride=32):
        - Temporal unshuffle: 6 * 8 = 48 channels, F_pixel → F_latent frames
        - Spatial PixelUnshuffle(8): 48 * 64 = 3072 channels
        - Conv2d(k=4, s=4): 3072 → out_dim, 4x spatial downsample
        - Total spatial: 8 * 4 = 32x (matches VAE)

    Zero-initialization: output_scale starts at 0 (same as ref), so adapter
    outputs zero at init, preserving base model behavior.
    """

    def __init__(self, in_channels: int = 6, out_dim: int = 4096,
                 downscale_factor: int = 8, conv_kernel: int = 4, conv_stride: int = 4,
                 num_residual_blocks: int = 1, vae_temporal_stride: int = 8,
                 split_plucker: bool = False,
                 output_scale_init: float = 0.0,
                 disable_output_scale: bool = False,
                 zero_init: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.in_channels = in_channels
        self.downscale_factor = downscale_factor
        self.conv_stride = conv_stride
        self.vae_temporal_stride = vae_temporal_stride
        self.total_spatial_ratio = downscale_factor * conv_stride  # 8 * 4 = 32
        self.split_plucker = split_plucker
        self.zero_init = zero_init

        init_mode = "ZERO" if zero_init else "Xavier"
        print(f"[WanCameraAdapter] ========== Weight Init Mode: {init_mode} ==========")

        # After temporal unshuffle: in_channels * vae_temporal_stride (6 * 8 = 48)
        ch_temporal = in_channels * vae_temporal_stride  # 48


        # Original single-branch path
        # After spatial PixelUnshuffle: ch_temporal * downscale_factor^2 (48 * 64 = 3072)
        ch_after_unshuffle = ch_temporal * downscale_factor ** 2  # 3072

        # Architecture: Spatial PixelUnshuffle → Conv2d (strided) → ResidualBlock
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv = nn.Conv2d(ch_after_unshuffle, out_dim,
                                kernel_size=conv_kernel, stride=conv_stride, padding=0)
        if zero_init:
            nn.init.zeros_(self.conv.weight)
        else:
            nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.residual_blocks = nn.Sequential(
            *[_ResidualBlock(out_dim, zero_init=zero_init) for _ in range(num_residual_blocks)]
        )

        # Learnable output scale (init=0 for stable training start)
        if not disable_output_scale:
            self.output_scale = nn.Parameter(torch.full((1,), output_scale_init))
        else:
            self.output_scale = None

        # Init summary logging
        total_params = sum(p.numel() for p in self.parameters())
        all_zero = all(p.data.abs().max().item() == 0.0 for p in self.parameters())
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                w_abs_mean = module.weight.data.abs().mean().item()
                w_max = module.weight.data.abs().max().item()
                layer_type = "Conv2d" if isinstance(module, nn.Conv2d) else "Linear"
                print(f"[WanCameraAdapter] init {name}: {layer_type}{list(module.weight.shape)} ({init_mode}) "
                      f"| abs_mean={w_abs_mean:.8f}, max={w_max:.8f}")
        print(f"[WanCameraAdapter] ========== Init Summary: mode={init_mode}, "
              f"total_params={total_params:,}, all_zero={all_zero} ==========")

    def forward(
        self,
        plucker_pixel: torch.Tensor,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        """
        Args:
            plucker_pixel: [B, 6, F_pixel, H_pixel, W_pixel] at full pixel frame rate
            latent_frames, latent_height, latent_width: target latent dimensions

        Returns:
            [B, F_lat*H_lat*W_lat, out_dim]
        """
        global _adapter_call_cnt
        _adapter_call_cnt += 1
        _cnt = _adapter_call_cnt
        _detail = (_cnt <= _DIAG_DETAIL)
        _do_log = _detail or (_cnt % _DIAG_INTERVAL == 0)
        B, C, F_in, H_in, W_in = plucker_pixel.shape
        S = self.vae_temporal_stride  # 8
        r = self.total_spatial_ratio  # 32

        # if _detail:
        #     print(f"[Adapter-DIAG] #{_cnt} input: [{B}, {C}, {F_in}, {H_in}, {W_in}], "
        #           f"target latent=({latent_frames}, {latent_height}, {latent_width}), "
        #           f"input norm={plucker_pixel.norm().item():.4f}")

        # Step 1: Temporal pixel unshuffle (VideoX-Fun convention)
        # Repeat first frame S times + rest frames
        first_repeated = plucker_pixel[:, :, 0:1].repeat(1, 1, S, 1, 1)  # [B, 6, S, H, W]
        rest = plucker_pixel[:, :, 1:]  # [B, 6, F-1, H, W]
        x = torch.cat([first_repeated, rest], dim=2)  # [B, 6, F_in+S-1, H, W]

        F_after_unshuffle = x.shape[2]  # frames after repeating the first frame S times and concatenating the rest
        F_needed = latent_frames * S   # target frame count = latent_frames * vae_stride

        if F_after_unshuffle > F_needed:
            x = x[:, :, :F_needed]
            if not _LTX_QUIET:  # noqa
                print(f"[CamAdapter] trim: F_in={F_in} -> unshuffle={F_after_unshuffle} -> keep the first {F_needed}")
        elif F_after_unshuffle < F_needed:
            pad_n = F_needed - F_after_unshuffle
            x = torch.cat([x, x[:, :, -1:].expand(-1, -1, pad_n, -1, -1)], dim=2)
            if not _LTX_QUIET:  # noqa
                print(f"[CamAdapter] pad: F_in={F_in} -> unshuffle={F_after_unshuffle} -> repeat the last frame")

        F_lat = latent_frames  # keep this consistent with latent_frames

        # Reshape: [B, C, F_needed, H, W] → [B, C*S, F_lat, H, W]
        x = x.view(B, C, F_lat, S, H_in, W_in)       # [B, 6, F_lat, 8, H, W]
        x = x.permute(0, 1, 3, 2, 4, 5)               # [B, 6, 8, F_lat, H, W]
        x = x.reshape(B, C * S, F_lat, H_in, W_in)    # [B, 48, F_lat, H, W]

        ch = C * S  # 48

        if _detail:
            print(f"[Adapter-DIAG] #{_cnt} temporal unshuffle: "
                  f"[{B},{C},{F_in},{H_in},{W_in}] → [{B},{ch},{F_lat},{H_in},{W_in}], "
                  f"norm={x.norm().item():.4f}")

        # Step 2: Spatial alignment (ensure H, W match latent * total_spatial_ratio)
        H_target = latent_height * r
        W_target = latent_width * r
        _spatial_interp = (H_in != H_target or W_in != W_target)
        if _spatial_interp:
            if _detail:
                print(f"[Adapter-DIAG] #{_cnt} spatial interpolate: "
                      f"({H_in},{W_in}) → ({H_target},{W_target})")
            x = F.interpolate(
                x.float(),
                size=(F_lat, H_target, W_target),
                mode='trilinear',
                align_corners=True,
            ).to(dtype=plucker_pixel.dtype)
            H_in, W_in = H_target, W_target

        # Step 3: Per-frame spatial PixelUnshuffle + Conv2d + ResBlock
        T = F_lat
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, ch, H_in, W_in)


        x = self.pixel_unshuffle(x)    # [B*T, 3072, H/8, W/8]
        x = self.conv(x)               # [B*T, out_dim, H/32, W/32]
        x = self.residual_blocks(x)    # [B*T, out_dim, H_lat, W_lat]

        if _detail:
            print(f"[Adapter-DIAG] #{_cnt} after conv+resblock: {list(x.shape)}, "
                  f"norm={x.norm().item():.4f}")

        # Reshape back to video
        x = x.view(B, T, self.out_dim, latent_height, latent_width)
        x = x.permute(0, 2, 1, 3, 4)  # [B, out_dim, T, H_lat, W_lat]

        # Learnable output scale (init=0 for stable training)
        if self.output_scale is not None:
            x = x * self.output_scale

        # Flatten to token sequence: [B, out_dim, T*H*W] → [B, T*H*W, out_dim]
        x = x.flatten(2).transpose(1, 2)

        return x


class SimplePatchCameraAdapter(nn.Module):
    """Lingbot-world style: interpolate plucker to latent res → Linear → residual MLP.

    Compared to WanCameraAdapter (PixelUnshuffle + Conv2d + ResBlock, ~500M params),
    this is a lightweight adapter (~33M params) that directly projects 6D Plucker
    coordinates to model dim. The output magnitude is naturally controlled by Xavier
    init on a low-dimensional input (6 → dim), avoiding the explosion seen with
    WanCameraAdapter's Xavier-initialized heavy convolutions.

    Architecture (matches lingbot-world):
        interpolate(plucker, latent_res) → flatten → Linear(6, dim) → residual MLP
    """

    def __init__(self, in_channels: int = 6, out_dim: int = 4096):
        super().__init__()
        self.out_dim = out_dim
        self.in_channels = in_channels

        self.patch_proj = nn.Linear(in_channels, out_dim)
        self.mlp_layer1 = nn.Linear(out_dim, out_dim)
        self.mlp_layer2 = nn.Linear(out_dim, out_dim)

        # Xavier weight + zero bias (lingbot-world convention)
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)
        nn.init.xavier_uniform_(self.mlp_layer1.weight)
        nn.init.zeros_(self.mlp_layer1.bias)
        nn.init.xavier_uniform_(self.mlp_layer2.weight)
        nn.init.zeros_(self.mlp_layer2.bias)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[SimplePatchCameraAdapter] Linear({in_channels}, {out_dim}) + residual MLP, "
              f"total_params={total_params:,}, Xavier init")

    def forward(
        self,
        plucker_pixel: torch.Tensor,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        """
        Args:
            plucker_pixel: [B, 6, F_pixel, H_pixel, W_pixel]
            latent_frames, latent_height, latent_width: target latent dimensions

        Returns:
            [B, F_lat*H_lat*W_lat, out_dim]
        """
        # Interpolate to latent resolution
        plucker_latent = F.interpolate(
            plucker_pixel.float(),
            size=(latent_frames, latent_height, latent_width),
            mode='trilinear', align_corners=True,
        ).to(dtype=plucker_pixel.dtype)  # [B, 6, F_lat, H_lat, W_lat]

        # Flatten to tokens: [B, L, 6]
        plucker_tokens = plucker_latent.flatten(2).transpose(1, 2)

        # Project + residual MLP (lingbot-world convention)
        x = self.patch_proj(plucker_tokens)  # [B, L, dim]
        h = self.mlp_layer2(F.silu(self.mlp_layer1(x)))
        x = x + h  # residual
        return x


def get_relative_c2w(c2w_matrices: np.ndarray) -> np.ndarray:
    """Compute relative camera poses (first frame as reference).

    CameraCtrl convention: relative_i = w2c_first @ c2w_i
    - R_relative: rotates from camera-i's frame to camera-0's frame
    - t_relative: camera-i's origin in camera-0's frame

    Args:
        c2w_matrices: [F, 4, 4]

    Returns:
        Relative poses [F, 4, 4] (first frame is identity)
    """
    w2c_first = np.linalg.inv(c2w_matrices[0])
    relative_c2w = [np.eye(4, dtype=np.float32)]
    for i in range(1, len(c2w_matrices)):
        relative_c2w.append((w2c_first @ c2w_matrices[i]).astype(np.float32))
    return np.array(relative_c2w, dtype=np.float32)


# HunyuanVideo action mapping: one-hot [4] -> label 0~8
HUNYUAN_ACTION_MAPPING = {
    (0, 0, 0, 0): 0,  # static
    (1, 0, 0, 0): 1,  # forward
    (0, 1, 0, 0): 2,  # backward
    (0, 0, 1, 0): 3,  # right
    (0, 0, 0, 1): 4,  # left
    (1, 0, 1, 0): 5,  # forward + right
    (1, 0, 0, 1): 6,  # forward + left
    (0, 1, 1, 0): 7,  # backward + right
    (0, 1, 0, 1): 8,  # backward + left
}


def _hunyuan_one_hot_to_label(one_hot: np.ndarray) -> int:
    key = tuple(int(x) for x in one_hot)
    return HUNYUAN_ACTION_MAPPING.get(key, 0)


def c2w_to_action_labels(
    c2w_matrices: np.ndarray,
    vae_temporal_stride: int = 8,
    translation_threshold: float = 0.0001,
    rotation_threshold: float = 0.001,
    classify_mode: str = "threshold",
) -> torch.Tensor:
    """Convert c2w to 81-class action labels (9 trans × 9 rot).

    Args:
        c2w_matrices: [N, 4, 4] extrinsics at pixel-frame level
        vae_temporal_stride: VAE temporal downsampling stride
        translation_threshold: translation threshold
        rotation_threshold: rotation threshold
        classify_mode:
            "threshold" - max-norm normalization and per-component comparison (default)
            "direction_angle" - adaptive threshold with direction-angle classification

    Returns:
        action_labels: [latent_num] tensor (0~80)
    """
    from scipy.spatial.transform import Rotation as R

    num_frames = len(c2w_matrices)
    stride = vae_temporal_stride
    latent_num = (num_frames - 1) // stride + 1

    latent_frame_indices = [min(i * stride, num_frames - 1) for i in range(latent_num)]
    c2ws_sampled = np.array([c2w_matrices[i] for i in latent_frame_indices])

    trans_one_hot = np.zeros((latent_num, 4), dtype=np.int32)
    rotate_one_hot = np.zeros((latent_num, 4), dtype=np.int32)

    if classify_mode == "direction_angle":
        relative_c2w = np.zeros_like(c2ws_sampled)
        relative_c2w[0] = c2ws_sampled[0]
        if len(c2ws_sampled) > 1:
            C_inv = np.linalg.inv(c2ws_sampled[:-1])
            relative_c2w[1:] = np.einsum('nij,njk->nik', C_inv, c2ws_sampled[1:])

        move_thresh = translation_threshold
        rot_thresh_deg = rotation_threshold * (180.0 / np.pi)  # to degrees
        if latent_num > 1:
            step_norms = np.linalg.norm(relative_c2w[1:, :3, 3], axis=1)
            nonzero = step_norms[step_norms > 1e-12]
            if nonzero.size > 0:
                move_thresh = max(1e-6, 0.05 * np.median(nonzero))

        for i in range(1, latent_num):
            move_dirs = relative_c2w[i, :3, 3]
            move_norms = np.linalg.norm(move_dirs)

            if move_norms > move_thresh:
                move_norm_dirs = move_dirs / move_norms
                trans_angles_deg = np.degrees(np.arccos(np.clip(move_norm_dirs, -1.0, 1.0)))
            else:
                trans_angles_deg = np.zeros(3)

            R_rel = relative_c2w[i, :3, :3]
            rot_angles_deg = R.from_matrix(R_rel).as_euler("xyz", degrees=True)

            if move_norms > move_thresh:
                if trans_angles_deg[2] < 60:       # +z: forward
                    trans_one_hot[i, 0] = 1
                elif trans_angles_deg[2] > 120:    # -z: backward
                    trans_one_hot[i, 1] = 1
                if trans_angles_deg[0] < 60:       # +x: right
                    trans_one_hot[i, 2] = 1
                elif trans_angles_deg[0] > 120:    # -x: left
                    trans_one_hot[i, 3] = 1

            if rot_angles_deg[0] > rot_thresh_deg:
                rotate_one_hot[i, 0] = 1
            elif rot_angles_deg[0] < -rot_thresh_deg:
                rotate_one_hot[i, 1] = 1
            if rot_angles_deg[1] > rot_thresh_deg:
                rotate_one_hot[i, 2] = 1
            elif rot_angles_deg[1] < -rot_thresh_deg:
                rotate_one_hot[i, 3] = 1

    else:
        eps = 1e-10
        c2w0_inv = np.linalg.inv(c2ws_sampled[0])
        c2ws_aligned = np.array([c2w0_inv @ c for c in c2ws_sampled])
        max_norm = np.linalg.norm(c2ws_aligned[:, :3, 3], axis=1).max()
        if max_norm > eps:
            c2ws_aligned[:, :3, 3] /= max_norm

        trans_thresh = translation_threshold * stride
        rot_thresh = rotation_threshold * stride

        for i in range(1, latent_num):
            T_rel = np.linalg.inv(c2ws_aligned[i - 1]) @ c2ws_aligned[i]
            t_rel = T_rel[:3, 3]
            R_rel = T_rel[:3, :3]
            x_move, _, z_move = t_rel
            rot_euler = R.from_matrix(R_rel).as_euler("xyz", degrees=False)

            if z_move > trans_thresh: trans_one_hot[i, 0] = 1
            if z_move < -trans_thresh: trans_one_hot[i, 1] = 1
            if x_move > trans_thresh: trans_one_hot[i, 2] = 1
            if x_move < -trans_thresh: trans_one_hot[i, 3] = 1

            if rot_euler[0] > rot_thresh: rotate_one_hot[i, 0] = 1
            elif rot_euler[0] < -rot_thresh: rotate_one_hot[i, 1] = 1
            if rot_euler[1] > rot_thresh: rotate_one_hot[i, 2] = 1
            elif rot_euler[1] < -rot_thresh: rotate_one_hot[i, 3] = 1

    trans_labels = np.array([_hunyuan_one_hot_to_label(trans_one_hot[i]) for i in range(latent_num)])
    rotate_labels = np.array([_hunyuan_one_hot_to_label(rotate_one_hot[i]) for i in range(latent_num)])
    action_labels = trans_labels * 9 + rotate_labels
    return torch.tensor(action_labels, dtype=torch.long)


def c2w_to_action_vectors(
    c2w_normalized: np.ndarray,
    vae_temporal_stride: int = 8,
) -> torch.Tensor:
    """Extract 13-dimensional action vectors from an already normalized c2w sequence.

    The input must be the dataset-normalized c2w (first frame identity, relic/max scaling),
    the same extrinsics used by the continuous Plucker embedding; no second normalization is applied.

    13-dimensional velocity vector (all values >= 0):
      translation (camera-frame delta P):
        [0] dolly_in    (+z forward)
        [1] dolly_out   (-z backward)
        [2] truck_right  (+x right)
        [3] truck_left   (-x left)
        [4] pedestal_up  (+y up)
        [5] pedestal_down (-y down)
      rotation (xyz Euler angles):
        [6]  tilt_up      (+pitch)
        [7]  tilt_down    (-pitch)
        [8]  pan_right    (+yaw)
        [9]  pan_left     (-yaw)
        [10] roll_cw      (+roll)
        [11] roll_ccw     (-roll)
      static:
        [12] static       (1.0 when there is almost no motion between frames, else 0.0)

    Args:
        c2w_normalized: [N, 4, 4] normalized extrinsics (first frame identity)
        vae_temporal_stride: VAE temporal downsampling stride

    Returns:
        action_vectors: [latent_num, 13] float tensor
    """
    from scipy.spatial.transform import Rotation as R

    num_frames = len(c2w_normalized)
    stride = vae_temporal_stride
    latent_num = (num_frames - 1) // stride + 1

    # Sample latent frame indices
    latent_frame_indices = [min(i * stride, num_frames - 1) for i in range(latent_num)]
    c2ws = np.array([c2w_normalized[i] for i in latent_frame_indices])

    eps = 1e-10
    rot_mags = []
    for i in range(1, latent_num):
        T_rel = np.linalg.inv(c2ws[i - 1]) @ c2ws[i]
        rot_euler = R.from_matrix(T_rel[:3, :3]).as_euler("xyz", degrees=False)
        rot_mags.append(np.linalg.norm(rot_euler))
    rot_mags = np.array(rot_mags)
    nonzero_rot = rot_mags > eps
    rot_dbar = rot_mags[nonzero_rot].mean() if np.any(nonzero_rot) else 1.0
    if rot_dbar < eps:
        rot_dbar = 1.0

    action_vecs = np.zeros((latent_num, 13), dtype=np.float32)
    _static_thresh = 1e-5  # threshold below which a frame counts as static

    for i in range(1, latent_num):
        T_rel = np.linalg.inv(c2ws[i - 1]) @ c2ws[i]
        t_rel = T_rel[:3, 3]  # per-frame translation in camera coordinates (already normalized)
        R_rel = T_rel[:3, :3]
        rot_euler = R.from_matrix(R_rel).as_euler("xyz", degrees=False)
        rot_euler = rot_euler / rot_dbar  # normalized rotation (divided by the mean rotation step)

        x, y, z = t_rel
        action_vecs[i, 0] = max(0.0, z)     # dolly_in    (+z)
        action_vecs[i, 1] = max(0.0, -z)    # dolly_out   (-z)
        action_vecs[i, 2] = max(0.0, x)     # truck_right (+x)
        action_vecs[i, 3] = max(0.0, -x)    # truck_left  (-x)
        action_vecs[i, 4] = max(0.0, y)     # pedestal_up (+y)
        action_vecs[i, 5] = max(0.0, -y)    # pedestal_down (-y)

        pitch, yaw, roll = rot_euler
        action_vecs[i, 6] = max(0.0, pitch)    # tilt_up    (+pitch)
        action_vecs[i, 7] = max(0.0, -pitch)   # tilt_down  (-pitch)
        action_vecs[i, 8] = max(0.0, yaw)      # pan_right  (+yaw)
        action_vecs[i, 9] = max(0.0, -yaw)     # pan_left   (-yaw)
        action_vecs[i, 10] = max(0.0, roll)     # roll_cw    (+roll)
        action_vecs[i, 11] = max(0.0, -roll)    # roll_ccw   (-roll)

        total_motion = np.linalg.norm(t_rel) + np.linalg.norm(rot_euler)
        action_vecs[i, 12] = 1.0 if total_motion < _static_thresh else 0.0

    action_vecs[0, 12] = 1.0

    return torch.tensor(action_vecs, dtype=torch.float32)


def prepare_camera_inputs(
    K_ctrl: torch.Tensor,
    c2w_ctrl: torch.Tensor,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
    dtype: torch.dtype,
    mode: str = "default",
    pixel_height: int = 0,
    pixel_width: int = 0,
    original_height: float = 0.0,
    original_width: float = 0.0,
    compute_action_labels: bool = True,
    c2w_raw: torch.Tensor = None,  # raw (un-normalized) extrinsics, used for 81-class labels
    camera_norm_mode: str = "relic",
    discrete_mode: str = "81cls",  # "81cls" (81-class labels) or "13dim" (13-dim velocity)
    action_classify_mode: str = "direction_angle",  # "threshold" or "direction_angle"
    # legacy params (ignored, kept for API compat)
    auto_rot_thresh: bool = True,
    rot_thresh_min_deg: float = 0.5,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """Prepare camera control inputs from dataset outputs.

    The dataset has already:
    1. normalized the intrinsics into [0,1]
    2. normalized the extrinsics (first frame identity, max(||t||) about 1.0)

    This function:
    - rescales the normalized intrinsics to the target pixel resolution with aspect-ratio correction
    - computes Plucker coordinates via ray_condition (batched)
    - optionally computes discrete action labels

    Args:
        K_ctrl: normalized [0,1] intrinsics [3,3] or [B,3,3]
        c2w_ctrl: normalized extrinsics [N,4,4] or [B,N,4,4] (first frame identity)
        latent_frames, latent_height, latent_width: Latent dimensions
        device, dtype: Target device and dtype
        mode: 'wan_inject' (pixel-res Plucker)
        pixel_height, pixel_width: Pixel resolution for Plucker computation
        original_height, original_width: source resolution, used for aspect-ratio correction
        compute_action_labels: also compute discrete action labels (default True);
            set False to skip the scipy dependency and return action_labels=None

    Returns:
        (action_labels, plucker_coords, action_vectors)
        action_labels: [B, latent_num] tensor (0..80) or None
        plucker_coords: [B, 6, F, H_pixel, W_pixel]
        action_vectors: [B, latent_num, 8] float tensor or None
    """
    global _prepare_call_cnt
    K_np = K_ctrl.cpu().numpy() if isinstance(K_ctrl, torch.Tensor) else K_ctrl
    c2w_np = c2w_ctrl.cpu().numpy() if isinstance(c2w_ctrl, torch.Tensor) else c2w_ctrl

    if K_np.ndim == 2:
        K_np = K_np[np.newaxis]
    if c2w_np.ndim == 2:
        c2w_np = c2w_np[np.newaxis]
    if c2w_np.ndim == 3 and K_np.ndim == 3 and K_np.shape[0] != c2w_np.shape[0]:
        K_np = np.broadcast_to(K_np, (c2w_np.shape[0],) + K_np.shape[-2:])
    if c2w_np.ndim == 4:
        batch_size = c2w_np.shape[0]
    else:
        c2w_np = c2w_np[np.newaxis]
        batch_size = 1
    if K_np.ndim == 2:
        K_np = K_np[np.newaxis]

    action_labels_list = [] if (compute_action_labels and discrete_mode == "81cls") else None
    action_vectors_list = [] if (compute_action_labels and discrete_mode != "81cls") else None
    K_vec_list = []     # [B, V, 4] for ray_condition
    c2w_all_list = []   # [B, V, 4, 4] for ray_condition

    H, W = pixel_height, pixel_width

    for b in range(batch_size):
        c2w_b = c2w_np[b]
        K_b = K_np[b] if K_np.shape[0] > 1 else K_np[0]
        num_frames = c2w_b.shape[0]

        if compute_action_labels:
            # if discrete_mode == "81cls":
            if c2w_raw is not None:
                c2w_raw_np = c2w_raw.cpu().numpy() if isinstance(c2w_raw, torch.Tensor) else c2w_raw
                c2w_raw_b = c2w_raw_np[b] if c2w_raw_np.ndim == 4 else (c2w_raw_np if c2w_raw_np.ndim == 3 else c2w_b)
            else:
                c2w_raw_b = c2w_b
            _al = c2w_to_action_labels(c2w_raw_b, vae_temporal_stride=8, classify_mode=action_classify_mode)
            action_labels_list.append(_al)
            # else:
            #     _av = c2w_to_action_vectors(c2w_b, vae_temporal_stride=8)
            #     action_vectors_list.append(_av)

        c2w_for_plucker = c2w_b.astype(np.float32)

        fx_n = float(K_b[0, 0])
        fy_n = float(K_b[1, 1])
        cx_n = float(K_b[0, 2])
        cy_n = float(K_b[1, 2])

        if original_width > 0 and original_height > 0:
            sample_wh_ratio = W / H
            pose_wh_ratio = original_width / original_height
            if pose_wh_ratio > sample_wh_ratio:
                resized_ori_w = H * pose_wh_ratio
                fx_n = resized_ori_w * fx_n / W
            else:
                resized_ori_h = W / pose_wh_ratio
                fy_n = resized_ori_h * fy_n / H

        K_vec = np.array([fx_n * W, fy_n * H, cx_n * W, cy_n * H], dtype=np.float32)
        K_vec_frames = np.tile(K_vec[None, :], (num_frames, 1))  # [V, 4]
        K_vec_list.append(K_vec_frames)
        c2w_all_list.append(c2w_for_plucker)

        if _prepare_call_cnt < _DIAG_DETAIL and b == 0:
            _t_range = np.linalg.norm(c2w_for_plucker[:, :3, 3], axis=1)
            if not _LTX_QUIET:  # noqa
                print(f"[Plucker-DIAG] batch={b}, "
                  f"c2w_frames={c2w_for_plucker.shape[0]}, "
                  f"t_norm(min/med/max)={_t_range.min():.4f}/{np.median(_t_range):.4f}/{_t_range.max():.4f}")
            if not _LTX_QUIET:  # noqa
                print(f"[Plucker-DIAG] K_vec: fx={K_vec[0]:.1f} fy={K_vec[1]:.1f} "
                  f"cx={K_vec[2]:.1f} cy={K_vec[3]:.1f}")

    # ── Batched Plucker via ray_condition (wan convention) ──
    # K_batch = torch.as_tensor(np.stack(K_vec_list, axis=0), dtype=torch.float32)
    # c2w_batch = torch.as_tensor(np.stack(c2w_all_list, axis=0), dtype=torch.float32)
    # plucker_bhw6 = ray_condition(K_batch, c2w_batch, H, W, device='cpu')
    # plucker_coords = plucker_bhw6.permute(0, 4, 1, 2, 3).contiguous()
    # plucker_coords = plucker_coords.to(device=device, dtype=dtype)
    K_batch = torch.as_tensor(np.stack(K_vec_list, axis=0), dtype=torch.float32, device=device)   # [B, V, 4]
    c2w_batch = torch.as_tensor(np.stack(c2w_all_list, axis=0), dtype=torch.float32, device=device)  # [B, V, 4, 4]
    plucker_bhw6 = ray_condition(K_batch, c2w_batch, H, W, device=device)  # [B, V, H, W, 6]
    plucker_coords = plucker_bhw6.permute(0, 4, 1, 2, 3).contiguous()    # [B, 6, V, H, W]
    plucker_coords = plucker_coords.to(dtype=dtype)  # fp32 -> bf16, releasing the fp32 intermediates
    del plucker_bhw6, K_batch, c2w_batch
    torch.cuda.empty_cache()  # reclaim the fp32 intermediates immediately

    action_labels = None
    action_vectors = None
    if compute_action_labels:
        if discrete_mode == "81cls" and action_labels_list:
            action_labels = torch.stack(action_labels_list, dim=0).to(device=device)
            al_len = action_labels.shape[1]
            if al_len != latent_frames and al_len > 0 and latent_frames > 0:
                resample_idx = torch.linspace(0, al_len - 1, latent_frames).long()
                action_labels = action_labels[:, resample_idx]
        elif action_vectors_list:
            action_vectors = torch.stack(action_vectors_list, dim=0).to(device=device, dtype=dtype)
            av_len = action_vectors.shape[1]
            if av_len != latent_frames and av_len > 0 and latent_frames > 0:
                resample_idx = torch.linspace(0, av_len - 1, latent_frames).long()
                action_vectors = action_vectors[:, resample_idx]

    _prepare_call_cnt += 1
    _do_log = (_prepare_call_cnt <= _DIAG_DETAIL) or (_prepare_call_cnt % _DIAG_INTERVAL == 0)
    _res_tag = f"{plucker_coords.shape[3]}x{plucker_coords.shape[4]}" if plucker_coords.dim() == 5 else "?"
    _pnorm = plucker_coords.norm().item()
    _pmin = plucker_coords.min().item()
    _pmax = plucker_coords.max().item()
    _has_nan = torch.isnan(plucker_coords).any().item()
    _has_inf = torch.isinf(plucker_coords).any().item()

    if _do_log:
        _av_info = f"{action_vectors.shape} (first_frame={action_vectors[0,1,:6].tolist()[:6]})" if action_vectors is not None else "None"
        if not _LTX_QUIET:  # noqa
            print(f"[Plucker-DIAG] prepare #{_prepare_call_cnt} (mode={mode}): "
              f"plucker={plucker_coords.shape} ({_res_tag}), "
              f"norm={_pnorm:.2f}, range=[{_pmin:.4f}, {_pmax:.4f}], "
              f"nan={_has_nan}, inf={_has_inf}, "
              f"action_vectors={_av_info}")

    if _has_nan or _has_inf:
        if not _LTX_QUIET:  # noqa
            print(f"[Plucker-DIAG] !!ALERT!! prepare #{_prepare_call_cnt}: "
              f"plucker contains nan={_has_nan} inf={_has_inf}")

    return action_labels, plucker_coords, action_vectors


# =============================================================================
# nn.Module Components
# =============================================================================

# PluckerEmbeddingProcessor and AdditiveCameraAdapter have been removed.
# All camera injection modes (scale_shift, additive, wan_inject) now use
# WanCameraAdapter with pixel-resolution Plucker for lossless spatial downsampling.


class VelocityActionEmbedder(nn.Module):
    """Embed RELIC 13-dim continuous action velocity vectors.

    Input [B, T, 13]: each value is the speed along one motion axis (>= 0).
    6 translation + 6 rotation + 1 static (see c2w_to_action_vectors)

    Each axis is sinusoidally encoded, concatenated and projected to output_dim by an MLP.
    Xavier init + bias=0。
    """
    ACTION_DIM = 13  # 6 translation + 6 rotation + 1 static

    def __init__(self, dim: int, output_dim: int = 0, freq_dim_per_axis: int = 32):
        super().__init__()
        self.freq_dim_per_axis = freq_dim_per_axis
        _total_freq = self.ACTION_DIM * freq_dim_per_axis
        _out = output_dim if output_dim > 0 else dim
        self.mlp = nn.Sequential(
            nn.Linear(_total_freq, dim), nn.SiLU(), nn.Linear(dim, _out),
        )
        # Xavier init + bias=0
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, action_vectors: torch.Tensor) -> torch.Tensor:
        """[B, T, 13] -> [B, T, output_dim]"""
        if action_vectors.dim() == 2:
            action_vectors = action_vectors.unsqueeze(0)  # [T, 13] → [1, T, 13]

        B, T, D = action_vectors.shape
        assert D == self.ACTION_DIM, f"Expected {self.ACTION_DIM} dims, got {D}"
        freq = self.freq_dim_per_axis
        # [B, T, 8] -> [B, T, 8, freq] -> [B, T, 8*freq]
        embeddings = []
        for axis in range(self.ACTION_DIM):
            axis_val = action_vectors[..., axis]  # [B, T]
            emb = sinusoidal_embedding_1d(freq, axis_val)  # [B, T, freq]
            embeddings.append(emb)
        combined = torch.cat(embeddings, dim=-1)  # [B, T, 8*freq]
        return self.mlp(combined.to(self.mlp[0].weight.dtype))


class DiscreteActionEmbedder81(nn.Module):
    """Embed 81-class action labels via sinusoidal encoding + MLP.

    Architecture:
    sinusoidal(freq_dim) → Linear(freq_dim→dim) → SiLU → Linear(dim→dim)
    The last layer is zero-initialized. The output dimension is inner_dim and requires the external

    Args:
        dim: output dimension (inner_dim)
        freq_dim: sinusoidal encoding dimension
        num_actions: number of classes
    """

    def __init__(self, dim: int, freq_dim: int = 256, num_actions: int = 81):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.mlp[2].weight)
        if self.mlp[2].bias is not None:
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, action_labels: torch.Tensor) -> torch.Tensor:
        """[B, T] int (0~80) -> [B, T, dim]"""
        action_emb = sinusoidal_embedding_1d(self.freq_dim, action_labels.float())
        return self.mlp(action_emb.to(self.mlp[0].weight.dtype))


DiscreteActionEmbedder = VelocityActionEmbedder


class CameraControlDropout(nn.Module):
    """Training-time dropout for continuous camera signal (default 10%)."""

    def __init__(self, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout_prob = dropout_prob

    def forward(self, camera_signal: torch.Tensor) -> torch.Tensor:
        if self.training and distributed_safe_dropout_decision(self.dropout_prob, camera_signal.device):
            return torch.zeros_like(camera_signal)
        return camera_signal


class DiscreteActionDropout(nn.Module):
    """Training-time dropout for discrete action embedding (default 10%)."""

    def __init__(self, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout_prob = dropout_prob

    def forward(self, action_emb: torch.Tensor) -> torch.Tensor:
        if self.training and distributed_safe_dropout_decision(self.dropout_prob, action_emb.device):
            return torch.zeros_like(action_emb)
        return action_emb


class RotationPredictionHead(nn.Module):
    """Auxiliary head: predict per-frame relative Euler angles from camera features.

    Provides direct rotation supervision signal to improve rotation encoding
    in the camera adapter. Small (~4M params), not wrapped in FSDP.
    """

    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.pool_proj = nn.Linear(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # predict 3 Euler angles (xyz)
        )

    def forward(self, cam_features: torch.Tensor, num_frames: int, height: int, width: int) -> torch.Tensor:
        """
        Args:
            cam_features: [B, T*H*W, dim] camera adapter output (with computation graph)
            num_frames: number of latent frames (T)
            height: latent height (H)
            width: latent width (W)

        Returns:
            [B, T, 3] predicted Euler angles per latent frame
        """
        B = cam_features.shape[0]
        # Reshape to [B, T, H*W, dim] and spatial average pool
        x = cam_features.view(B, num_frames, height * width, -1).mean(dim=2)  # [B, T, dim]
        x = self.pool_proj(x)  # [B, T, hidden_dim]
        return self.mlp(x)  # [B, T, 3]


def extract_rotation_gt(c2w_ctrl: torch.Tensor, latent_frames: int, vae_temporal_stride: int = 8) -> torch.Tensor:
    """Extract per-latent-frame Euler angles from c2w matrices as ground truth.

    Args:
        c2w_ctrl: [B, N, 4, 4] or [N, 4, 4] camera-to-world matrices (normalized, first=identity)
        latent_frames: number of latent frames to sample
        vae_temporal_stride: temporal stride of VAE (default 8)

    Returns:
        [B, latent_frames, 3] Euler angles (xyz, radians) for each latent frame
    """
    from scipy.spatial.transform import Rotation as R

    c2w_np = c2w_ctrl.detach().cpu().numpy()
    if c2w_np.ndim == 3:
        c2w_np = c2w_np[np.newaxis]  # [1, N, 4, 4]

    B, N = c2w_np.shape[0], c2w_np.shape[1]
    result = np.zeros((B, latent_frames, 3), dtype=np.float32)

    for b in range(B):
        # Sample at latent frame rate
        for t in range(latent_frames):
            # First latent frame maps to pixel frame 0 (identity -> euler=0)
            # Subsequent latent frames map to pixel frames at stride intervals
            if t == 0:
                pf = 0
            else:
                pf = min((t - 1) * vae_temporal_stride + 1, N - 1)

            # Compute relative rotation from previous latent frame
            if t == 0:
                # First frame: identity rotation -> euler = [0, 0, 0]
                result[b, t] = 0.0
            else:
                pf_prev = 0 if t == 1 else min((t - 2) * vae_temporal_stride + 1, N - 1)
                # Relative rotation: R_prev^-1 @ R_curr
                R_prev = c2w_np[b, pf_prev, :3, :3]
                R_curr = c2w_np[b, pf, :3, :3]
                R_rel = np.linalg.inv(R_prev) @ R_curr
                euler = R.from_matrix(R_rel).as_euler('xyz')  # radians
                result[b, t] = euler.astype(np.float32)

    return torch.from_numpy(result)


# =============================================================================
# =============================================================================

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

_SQRT2_INV = 1.0 / math.sqrt(2.0)

_LABEL_TO_VEC = {
    0: (0.0, 0.0),                          # static
    1: (0.0, -1.0),                          # forward (up)
    2: (0.0, 1.0),                           # backward (down)
    3: (1.0, 0.0),                           # right
    4: (-1.0, 0.0),                          # left
    5: (_SQRT2_INV, -_SQRT2_INV),            # forward + right
    6: (-_SQRT2_INV, -_SQRT2_INV),           # forward + left
    7: (_SQRT2_INV, _SQRT2_INV),             # backward + right
    8: (-_SQRT2_INV, _SQRT2_INV),            # backward + left
}


def action_label_to_vectors(action_label: int):
    """Split a single action label (0..80) into 2D translation and rotation vectors."""
    trans_label = action_label // 9
    rot_label = action_label % 9
    return _LABEL_TO_VEC.get(trans_label, (0.0, 0.0)), _LABEL_TO_VEC.get(rot_label, (0.0, 0.0))


def _add_circle_rgba(img, center, radius, bgr, alpha, thickness=-1):
    _cv2.circle(img, center, max(int(radius), 1),
                (bgr[0], bgr[1], bgr[2], float(alpha)),
                thickness, _cv2.LINE_AA)


def _radial_glow(img, center, r_outer, bgr, a_outer, power=2.0, step=2):
    r_outer = max(int(r_outer), 1)
    for rr in range(r_outer, 0, -step):
        t = rr / r_outer
        _add_circle_rgba(img, center, rr, bgr, a_outer * (t ** power), thickness=-1)


def draw_joystick(frame, center, radius, vec2, label=None):
    """Draw a blue energy-ring joystick on a frame (BGR uint8)."""
    if not _HAS_CV2:
        return
    h, w = frame.shape[:2]
    cx, cy = center
    x = float(np.clip(vec2[0], -1.0, 1.0))
    y = float(np.clip(vec2[1], -1.0, 1.0))
    knob_x = int(round(cx + x * radius * 0.85))
    knob_y = int(round(cy + y * radius * 0.85))

    overlay = np.zeros((h, w, 4), dtype=np.float32)
    blue = (255, 170, 50)
    blue_hot = (255, 210, 90)
    base_dark = (18, 22, 30)
    base_mid = (35, 45, 65)
    knob_core = (235, 245, 255)
    knob_edge = (165, 190, 215)

    _radial_glow(overlay, (cx, cy), radius * 0.95, base_dark, a_outer=0.75, power=1.6)
    _radial_glow(overlay, (cx, cy), radius * 0.78, base_mid, a_outer=0.40, power=2.2)
    _add_circle_rgba(overlay, (cx, cy), radius * 0.88, (0, 0, 0), 0.24, thickness=2)
    _add_circle_rgba(overlay, (cx, cy), radius * 0.70, (0, 0, 0), 0.14, thickness=2)
    _radial_glow(overlay, (cx, cy), radius * 1.22, blue, a_outer=0.26, power=2.6)
    _add_circle_rgba(overlay, (cx, cy), radius * 1.00, blue, 0.66, thickness=3)
    _add_circle_rgba(overlay, (cx, cy), radius * 0.98, blue_hot, 0.44, thickness=1)
    sx, sy = int(radius * 0.05), int(radius * 0.06)
    _radial_glow(overlay, (knob_x + sx, knob_y + sy), radius * 0.42, (0, 0, 0), a_outer=0.22, power=2.2)
    _radial_glow(overlay, (knob_x, knob_y), radius * 0.40, knob_edge, a_outer=0.62, power=1.8)
    _radial_glow(overlay, (knob_x, knob_y), radius * 0.33, knob_core, a_outer=0.80, power=2.6)
    _add_circle_rgba(overlay, (knob_x, knob_y), radius * 0.38, (255, 255, 255), 0.20, thickness=2)
    _add_circle_rgba(overlay, (knob_x, knob_y), radius * 0.39, (0, 0, 0), 0.14, thickness=2)
    _add_circle_rgba(overlay, (knob_x - int(radius * 0.12), knob_y - int(radius * 0.12)),
                     radius * 0.11, (255, 255, 255), 0.35, thickness=-1)
    if label:
        _cv2.putText(frame, label, (cx - int(radius * 0.8), cy - int(radius * 1.15)),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, _cv2.LINE_AA)
    alpha = overlay[:, :, 3:4]
    frame[:] = np.clip(
        frame.astype(np.float32) * (1.0 - alpha) + overlay[:, :, :3] * alpha, 0, 255
    ).astype(np.uint8)


def add_joystick_overlay(video_frames, action_labels, vae_temporal_stride: int = 8, smooth_alpha: float = 0.3):
    """Draw a dual-joystick HUD on the video frames from action_labels.

    Args:
        video_frames: list of [H, W, 3] uint8 numpy (RGB)
        action_labels: [latent_frames] or [B, latent_frames] int tensor/list (0~80)
        vae_temporal_stride: VAE temporal downsampling stride
        smooth_alpha: smoothing factor of the joystick position

    Returns:
        list of [H, W, 3] uint8 numpy (RGB)
    """
    if not _HAS_CV2:
        print("[CameraControl] WARNING: cv2 not available, joystick overlay skipped")
        return video_frames
    num_pixel_frames = len(video_frames)
    if num_pixel_frames == 0:
        return video_frames

    if hasattr(action_labels, 'cpu'):
        action_labels = action_labels.cpu().tolist()
    elif hasattr(action_labels, 'tolist'):
        action_labels = action_labels.tolist()
    if isinstance(action_labels[0], (list, tuple)):
        action_labels = action_labels[0]
    num_latent = len(action_labels)

    # latent-rate -> pixel-rate
    pixel_labels = []
    for pf in range(num_pixel_frames):
        if pf == 0:
            li = 0
        else:
            li = min((pf - 1) // vae_temporal_stride + 1, num_latent - 1)
        pixel_labels.append(action_labels[li])

    trans_vecs = []
    rot_vecs = []
    for al in pixel_labels:
        tv, rv = action_label_to_vectors(int(al))
        trans_vecs.append(tv)
        rot_vecs.append(rv)

    h, w = video_frames[0].shape[:2]
    short = min(w, h)
    radius = int(np.clip(short * 0.08, 24, 120))
    margin = int(np.clip(short * 0.04, 12, 80))
    left_center = (margin + radius, h - margin - radius)
    right_center = (w - margin - radius, h - margin - radius)

    ts = np.array([0.0, 0.0], dtype=np.float32)
    rs = np.array([0.0, 0.0], dtype=np.float32)

    result_frames = []
    for i in range(num_pixel_frames):
        frame_bgr = video_frames[i][:, :, ::-1].copy()
        tv_target = np.array(trans_vecs[i], dtype=np.float32)
        rv_target = np.array(rot_vecs[i], dtype=np.float32)
        ts = (1.0 - smooth_alpha) * ts + smooth_alpha * tv_target
        rs = (1.0 - smooth_alpha) * rs + smooth_alpha * rv_target
        draw_joystick(frame_bgr, left_center, radius, ts, label="Move")
        draw_joystick(frame_bgr, right_center, radius, rs, label="Rotate")
        result_frames.append(frame_bgr[:, :, ::-1].copy())

    return result_frames
