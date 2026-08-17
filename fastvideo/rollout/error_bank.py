"""
ErrorBank: error recycling in the style of Stable Video Infinity (SVI).

Reference: vita-epfl/Stable-Video-Infinity

SVI defines three inject paths; two are implemented here:
    A. latent inject (90%): δ → target_clean → add_noise → model
       - strict K matching plus a strict current-timestep grid sample
       - target_clean here corresponds to SVI's latents
    B. y inject (90%) - here this is the history inject:
       - SVI uses y = image_emb["y"] as cross-attention conditioning
       - the history memory tokens here are also cross-attention conditioning, so the semantics match
       - SVI fixes the last 3 frames; here the tail length is autoregression-aware (1..59 frames)
       with a linspace(0.5, 1.0) decay
       - concatenated across K with a strict current-timestep grid (the timestep locks the noise scale)
    C. noise inject (1% in SVI): not implemented (marginal benefit)

clean_prob overrides every path and injects nothing.

Single-buffer design:
    Reconstructing x0 = x_t - sigma * v yields a single kind of residual, so the latent inject and the
    history inject share one buffer and one residual definition.
    (SVI keeps two buffers because it has self_corr=True/False scheduler step paths; in practice both
     the latent and the y injection read from y_error_buffer, so it is effectively shared.)

Two-level bucketing by key = (K, timestep_grid_idx):
    K∈{2,4,8} (task-specific), grid∈[0, num_grids) by current sigma
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

import torch


class ErrorBank:
    """Residual buffer bucketed by (K, timestep grid), following SVI."""

    def __init__(
        self,
        buffer_k: int = 500,
        num_grids: int = 50,
        warmup_iter: int = 50,
        latent_prob: float = 0.9,
        clean_prob: float = 0.2,
        clean_buffer_update_prob: float = 0.1,
        replacement_strategy: str = 'l2_batch',
        gamma: float = 1.0,
        history_prob: float = 0.9,
        spatial_prob: float = 0.9,
        nearby_prob: float = 0.9,
        error_modulate_factor: float = 0.2,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            buffer_k: capacity per (K, grid) bucket
            num_grids: number of grids the timestep range [0,1] is split into
            warmup_iter: do not sample until this many pushes have happened
            latent_prob: probability of triggering a latent inject
            clean_prob: probability of overriding everything and injecting nothing
            clean_buffer_update_prob: probability of skipping the buffer update
            replacement_strategy: 'l2_batch' (replace the most similar) | 'random' | 'fifo'
            gamma: injection strength scalar
            history_prob: probability of triggering a history inject
            error_modulate_factor: post-sample intensity jitter of +/- factor
            device/dtype: residual storage (CPU + bf16 by default to save GPU memory)
        """
        self.buffer_k = int(buffer_k)
        self.num_grids = int(num_grids)
        self.warmup_iter = int(warmup_iter)
        self.latent_prob = float(latent_prob)
        self.clean_prob = float(clean_prob)
        self.clean_buffer_update_prob = float(clean_buffer_update_prob)
        self.replacement_strategy = str(replacement_strategy)
        self.gamma = float(gamma)
        self.history_prob = float(history_prob)
        self.spatial_prob = float(spatial_prob)
        self.nearby_prob = float(nearby_prob)
        self.error_modulate_factor = float(error_modulate_factor)
        self.device = device
        self.dtype = dtype

        if self.replacement_strategy not in ('l2_batch', 'random', 'fifo'):
            raise ValueError(
                f"replacement_strategy must be l2_batch/random/fifo, got {replacement_strategy}"
            )

        self._buckets: dict[tuple[int, int], list] = defaultdict(list)
        self._n_push = 0

    def _grid_idx(self, timestep: float) -> int:
        """timestep ∈ [0,1] (FlowMatch sigma) → grid index ∈ [0, num_grids)."""
        t = max(0.0, min(1.0, float(timestep)))
        return min(int(t * self.num_grids), self.num_grids - 1)

    def push(self, residual: torch.Tensor, timestep: float) -> None:
        """Collect the target residual delta = x0_clean - x_target into the matching (K, grid) bucket.

        Args:
            residual: [B, C, K, H, W] or [C, K, H, W]
            timestep: current sigma in [0,1]
        """
        if random.random() < self.clean_buffer_update_prob:
            return

        grid_idx = self._grid_idx(timestep)

        if residual.dim() == 5:
            for b in range(residual.shape[0]):
                _r = residual[b]
                K = int(_r.shape[1])
                self._push_one(_r, K, grid_idx)
                self._n_push += 1
        elif residual.dim() == 4:
            K = int(residual.shape[1])
            self._push_one(residual, K, grid_idx)
            self._n_push += 1
        else:
            raise ValueError(f"residual must be 4D or 5D, got {residual.dim()}D")

    def _push_one(self, r_chw: torch.Tensor, K: int, grid_idx: int) -> None:
        key = (K, grid_idx)
        bucket = self._buckets[key]
        r_cpu = r_chw.detach().to(dtype=self.dtype, device='cpu')

        if len(bucket) < self.buffer_k:
            bucket.append(r_cpu)
            return

        if self.replacement_strategy == 'l2_batch':
            stacked = torch.stack(bucket, dim=0).float()
            diff = stacked - r_cpu.unsqueeze(0).float()
            dists = diff.flatten(1).pow(2).sum(dim=1)
            idx = int(dists.argmin().item())
            bucket[idx] = r_cpu
        elif self.replacement_strategy == 'random':
            bucket[random.randint(0, len(bucket) - 1)] = r_cpu
        else:
            bucket.pop(0)
            bucket.append(r_cpu)

    def is_warm(self) -> bool:
        return self._n_push >= self.warmup_iter

    def roll_inject_flags(self, ignore_clean: bool = False) -> dict:
        """Roll the latent and history inject flags independently in one call.

        Args:
            ignore_clean: True skips clean_prob because the caller already made that decision;
                          False respects the clean_prob override.
        """
        if not self.is_warm():
            return {'latent': False, 'history': False, 'spatial': False, 'nearby': False}
        if (not ignore_clean) and random.random() < self.clean_prob:
            return {'latent': False, 'history': False, 'spatial': False, 'nearby': False}
        return {
            'latent':  random.random() < self.latent_prob,
            'history': random.random() < self.history_prob,
            'spatial': random.random() < self.spatial_prob,
            'nearby':  random.random() < self.nearby_prob,
        }

    def _apply_modulate(self, δ: torch.Tensor) -> torch.Tensor:
        """Apply the intensity jitter: delta *= U(1-f, 1+f)."""
        if self.error_modulate_factor > 0:
            mod = random.uniform(1.0 - self.error_modulate_factor,
                                 1.0 + self.error_modulate_factor)
            δ = δ * mod
        return δ

    def sample_one(
        self,
        K: int,
        timestep: float,
        target_shape: tuple,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        """Latent inject path: strict K match plus the current timestep grid.

        Applied to target_clean.

        Args:
            K: target K of the current step (shape-matched to target_clean)
            timestep: current sigma in [0,1]
            target_shape: (B, C, K, H, W) or (C, K, H, W)
        """
        if not self.is_warm():
            return None

        gi = self._grid_idx(timestep)
        key = (K, gi)
        bucket = self._buckets.get(key)
        if not bucket:
            return None

        r = bucket[random.randint(0, len(bucket) - 1)]   # CPU [C, K, H, W]

        if len(target_shape) == 5:
            expect_C, expect_K, expect_H, expect_W = target_shape[1:]
        elif len(target_shape) == 4:
            expect_C, expect_K, expect_H, expect_W = target_shape
        else:
            return None
        if (
            r.shape[0] != expect_C or r.shape[1] != expect_K
            or r.shape[2] != expect_H or r.shape[3] != expect_W
        ):
            return None

        δ = r.to(device=device, dtype=dtype or self.dtype)
        δ = self._apply_modulate(δ)

        if len(target_shape) == 5:
            δ = δ.unsqueeze(0).expand(target_shape[0], *δ.shape).contiguous()
        return δ

    def sample_X(self, p_decay: float = 0.1, max_X: int = 59) -> int:
        """Sample the history inject length X in [1, max_X] from an exponential distribution (small X is likely)."""
        X = int(random.expovariate(p_decay)) + 1
        return max(1, min(max_X, X))

    def sample_x_frames(
        self,
        X: int,
        target_shape: tuple,
        device: torch.device,
        timestep: float,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        """History inject path: concatenate X frames across K within the current timestep grid.

        Applied to the last X frames of the history, mimicking the drift accumulated at the end of a rollout.
        The timestep is locked so the noise scale matches the latent inject.
        Buckets are concatenated across K because the history tail length X is not tied to the current K.

        Args:
            X: number of frames to concatenate
            target_shape: (B, C, X, H, W) or (C, X, H, W), used for shape checks and broadcasting
            timestep: current sigma in [0,1] (locks the timestep grid)
        """
        if not self.is_warm():
            return None

        gi = self._grid_idx(timestep)
        cand_keys = [k for k, b in self._buckets.items()
                     if k[1] == gi and len(b) > 0]
        if not cand_keys:
            return None

        if len(target_shape) == 5:
            expect_C, _, expect_H, expect_W = target_shape[1:]
        elif len(target_shape) == 4:
            expect_C, _, expect_H, expect_W = target_shape
        else:
            return None

        chunks = []
        remaining = X
        max_iters = X + 10
        iters = 0
        while remaining > 0 and iters < max_iters:
            iters += 1
            key = random.choice(cand_keys)
            K_chosen = key[0]
            bucket = self._buckets[key]
            r = bucket[random.randint(0, len(bucket) - 1)]   # CPU [C, K_chosen, H, W]
            if r.shape[0] != expect_C or r.shape[2] != expect_H or r.shape[3] != expect_W:
                continue
            take = min(K_chosen, remaining)
            start = random.randint(0, K_chosen - take) if K_chosen > take else 0
            chunks.append(r[:, start:start + take])
            remaining -= take

        if remaining > 0 or not chunks:
            return None
        δ = torch.cat(chunks, dim=1).to(device=device, dtype=dtype or self.dtype)
        δ = self._apply_modulate(δ)

        if len(target_shape) == 5:
            δ = δ.unsqueeze(0).expand(target_shape[0], *δ.shape).contiguous()
        return δ

    def __len__(self) -> int:
        return sum(len(b) for b in self._buckets.values())

    def stats(self) -> dict:
        per_K: dict[int, int] = defaultdict(int)
        per_grid: dict[int, int] = defaultdict(int)
        for (K, gi), b in self._buckets.items():
            per_K[K] += len(b)
            per_grid[gi] += len(b)
        return {
            'n_push': self._n_push,
            'total_size': len(self),
            'n_buckets': len(self._buckets),
            'per_K': dict(per_K),
            'per_grid_top5': dict(
                sorted(per_grid.items(), key=lambda x: -x[1])[:5]
            ),
            'buffer_k': self.buffer_k,
            'num_grids': self.num_grids,
            'warm': self.is_warm(),
        }
