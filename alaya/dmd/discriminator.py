"""GAN discriminator for DMD2-style distillation (faithful HELIOS hook-based port).

The discriminator head (a 3D convolution stack) hangs on several intermediate blocks of the
critic / fake-score transformer and reads the hidden states of the target segment.

Differences from the reference implementation, adapted to this stack:
  - the model has no gan_mode return value, so a forward hook captures the block output instead
  - the token sequence is [sink | mem | spatial | nearby | target] and the target is always the
    trailing K*H*W tokens, so slicing hidden[:, -K*H*W:] and reshaping recovers the target
  - no spatial crop augmentation: cropping the target latent would break the index alignment
  - the first head layer is a 1x1 convolution for channel compression (the inner dimension is large)

Gradient checkpointing: the blocks run through torch.utils.checkpoint. The hook captures the block
output, which is a checkpoint boundary tensor and is not freed, so building the logits graph during
the forward pass is correct; a re-trigger during recompute is harmless because the capture is no longer read.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn


def _num_groups(channels: int, max_groups: int = 32) -> int:
    for g in (max_groups, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class Discriminator3DHead(nn.Module):
    """[B, in_ch, K, H, W] -> [B, 1] scalar logit: 3D convolutions, global pooling and a Linear.

    Adaptive average pooling accepts any spatial or temporal size.
    """

    def __init__(self, in_ch: int, hidden: int = 768, *, dtype=None, device=None):
        super().__init__()

        def conv_block(ci: int, co: int, stride):
            return nn.Sequential(
                nn.Conv3d(ci, co, kernel_size=3, stride=stride, padding=1),
                nn.GroupNorm(_num_groups(co), co),
                nn.SiLU(),
            )

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hidden, kernel_size=1),  # channel compression (the inner dimension can be large)
            nn.GroupNorm(_num_groups(hidden), hidden),
            nn.SiLU(),
            conv_block(hidden, hidden, (1, 2, 2)),  # spatial downsample
            conv_block(hidden, hidden, (1, 2, 2)),  # spatial downsample
            conv_block(hidden, hidden, (2, 2, 2)),  # temporal and spatial downsample
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(hidden, 1),
        )
        if dtype is not None or device is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GanDiscriminator(nn.Module):
    """Discriminator attached to several critic transformer blocks through forward hooks.

    Usage:
        disc = GanDiscriminator(score_model, hooks=[5,15,25,35], inner_dim=..., cond_map_dim=768)
        with disc.capturing():
            forward_velocity(score_model, noisy, sigma, cond)   # one critic forward
        logits = disc.compute_logits(K, H_lat, W_lat)           # list[ [B,1] ]
    """

    def __init__(
        self,
        score_model: nn.Module,
        *,
        hooks: list[int],
        inner_dim: int,
        cond_map_dim: int = 768,
        dtype=None,
        device=None,
    ):
        super().__init__()
        n_blocks = len(score_model.blocks)
        valid = []
        for h in hooks:
            if not (0 <= int(h) < n_blocks):
                raise ValueError(f"gan_hooks block index {h} out of range [0,{n_blocks})")
            valid.append(int(h))
        if not valid:
            raise ValueError("gan_hooks is empty; need >=1 block index for the hook discriminator")
        self.hook_indices = valid
        self.heads = nn.ModuleDict(
            {str(i): Discriminator3DHead(inner_dim, cond_map_dim, dtype=dtype, device=device) for i in valid}
        )
        self._captured: dict[int, torch.Tensor] = {}
        self._capturing = False
        self._handles = []
        self._register(score_model)

    def _register(self, score_model: nn.Module) -> None:
        for i in self.hook_indices:
            block = score_model.blocks[i]

            def make_hook(idx: int):
                def hook(module, inputs, output):
                    if self._capturing:
                        self._captured[idx] = output[0] if isinstance(output, tuple) else output
                    return None
                return hook

            self._handles.append(block.register_forward_hook(make_hook(i)))

    @contextmanager
    def capturing(self):
        """Enter capture mode: clear the cache and set the flag; on exit clear the flag but keep the cache."""
        self._captured = {}
        self._capturing = True
        try:
            yield
        finally:
            self._capturing = False

    def compute_logits(self, K: int, H: int, W: int) -> list[torch.Tensor]:
        """Reshape the captured target-segment suffix into [B,D,K,H,W] and run the discriminator head."""
        n_tgt = K * H * W
        logits = []
        for i in self.hook_indices:
            feat = self._captured.get(i)
            if feat is None:
                raise RuntimeError(f"gan hook block {i} produced no capture; was capturing() active during forward?")
            tgt = feat[:, -n_tgt:, :]  # the target is always the sequence suffix
            B, S, D = tgt.shape
            if S != n_tgt:
                raise RuntimeError(f"gan capture target tokens {S} != K*H*W={n_tgt} (block {i})")
            grid = tgt.reshape(B, K, H, W, D).permute(0, 4, 1, 2, 3).contiguous()  # [B,D,K,H,W]
            logits.append(self.heads[str(i)](grid))
        return logits

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.heads.parameters() if p.requires_grad]

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
