"""Next-Forcing depth=1 Multi-Chunk Prediction (MCP) head for DMD distillation.

A forward hook on the gradient-carrying generator forward captures gen-token hidden states from
the hook layers, fuses them with an MLP, concatenates the fused feature with a noised embedding of
the next chunk, and predicts the flow-matching velocity of that next chunk
(frames [target+K, target+2K)) with a small transformer head. The loss flows back through the fusion
MLP into the generator's intermediate layers, giving dense temporal supervision. depth=1 predicts one chunk ahead.

Hook / capturing / FSDP / gradient-checkpoint handling matches alaya/dmd/discriminator.py:
  - the hook captures block outputs (checkpoint boundary tensors, which are not freed), so building
    the loss graph during forward back-propagates correctly; a re-trigger during recompute is harmless.
  - only the selected step of few-step sampling carries gradients and run_generator returns right after,
    so wrapping run_generator in capturing() leaves the captured features from that
    gradient-carrying forward.
  - hooks are registered on the raw transformer blocks before FSDP wrapping; FSDP with
    use_orig_params=True still triggers them when it calls the inner block forward.

Token layout: [sink | mem | spatial | nearby | target]. The target is always the trailing K*H*W tokens
after patchify, so hidden[:, -K*H*W:] is the current chunk.

Simplifications: the head has no text cross-attention and takes no future action, because the fused
feature already carries the main model's text/action conditioning. Keeping depth-1 prediction
action-free also avoids future-action leakage. RoPE uses the next chunk positions (current target + K).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltx2.modules.model_ltx_2_3 import LTX23SelfAttention, FeedForward, Timesteps, rms_norm
from ltx2.modules.patchifier import (
    VideoLatentPatchifier,
    VideoLatentShape,
    SpatioTemporalScaleFactors,
    get_pixel_coords,
)
from ltx2.modules.rope import (
    precompute_freqs_cis,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
)


class _MCPBlock(nn.Module):
    """Lightweight denoising block: AdaLN-zero (sigma modulation) + self-attention (RoPE) + FeedForward.

    Reuses the main model's self-attention (RoPE, qk-norm, same attention backend) and FeedForward.
    AdaLN parameters come from a per-block Linear on the sigma embedding, zero-initialized so the block starts as identity.
    """

    def __init__(self, *, dim, heads, dim_head, rope_type, attention_function, apply_gated_attention, norm_eps):
        super().__init__()
        self.norm_eps = norm_eps
        self.attn = LTX23SelfAttention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            context_dim=None,
            rope_type=rope_type,
            norm_eps=norm_eps,
            attention_function=attention_function,
            apply_gated_attention=apply_gated_attention,
            enable_sparse_attention=False,
        )
        self.ff = FeedForward(dim, dim_out=dim)
        # AdaLN-zero: sigma_emb -> (shift/scale/gate) x (msa, mlp) = 6*dim, zero-init for an identity start.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, sigma_emb: torch.Tensor, freqs) -> torch.Tensor:
        # sigma_emb: [B, dim] sinusoidal -> [B, 6*dim] -> six [B,1,dim] tensors broadcast over tokens.
        mod = self.ada(sigma_emb).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        h = rms_norm(x, eps=self.norm_eps) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(h, pe=freqs)
        h = rms_norm(x, eps=self.norm_eps) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.ff(h)
        return x


class NextForcingHead(nn.Module):
    """depth=1 next-chunk prediction head. Hooks several generator blocks and predicts the next chunk.

    Usage (train_one_step_dmd, generator step only):
        with mcp.capturing():
            x0_fake = run_generator(transformer, noise, cond, sigma_list, grad_step=g)
        loss, log = mcp.compute_loss(x0_next=..., noise=..., sigma=..., next_indices_grid=..., fps=..., K=..., H=..., W=...)
        gen_total = dmd_loss + w * loss
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        hook_layers: list[int],
        num_blocks: int,
        fuse_hidden_mult: float = 1.0,
        dtype=None,
        device=None,
    ):
        super().__init__()
        n_blocks = len(transformer.blocks)
        valid = []
        for h in hook_layers:
            if not (0 <= int(h) < n_blocks):
                raise ValueError(f"next_forcing.hook_layers index {h} out of range [0,{n_blocks})")
            valid.append(int(h))
        if not valid:
            raise ValueError("next_forcing.hook_layers is empty")
        self.hook_indices = valid

        # Read RoPE / attention hyper-parameters from the main model so the head matches the generator
        D = int(transformer.inner_dim)
        heads = int(transformer.num_attention_heads)
        dim_head = D // heads
        self.inner_dim = D
        self.num_heads = heads
        self.rope_type = transformer.rope_type
        self.theta = float(transformer.positional_embedding_theta)
        self.max_pos = list(transformer.positional_embedding_max_pos)
        self.use_middle_indices_grid = bool(transformer.use_middle_indices_grid)
        self.normalize_rope_positions = bool(transformer.normalize_rope_positions)
        self.normalize_time_by_fps = bool(getattr(transformer, "normalize_time_by_fps", True))
        self._freq_grid_generator = (
            generate_freq_grid_np if bool(transformer.double_precision_rope) else generate_freq_grid_pytorch
        )
        attention_function = transformer.blocks[0].attn1.attention_function
        apply_gated_attention = bool(transformer.apply_gated_attention)
        norm_eps = float(getattr(transformer.blocks[0], "norm_eps", 1e-6))
        in_channels = int(transformer.patchify_proj.in_features)
        out_channels = int(transformer.proj_out.out_features)
        # The next-chunk input and the flow-matching target share the same latent channel count.
        # Assert early: if in_channels != out_channels the projection and the MSE target would mismatch.
        if in_channels != out_channels:
            raise ValueError(
                f"NextForcingHead assumes latent in_channels==out_channels, got {in_channels} vs {out_channels}"
            )

        # Fusion MLP: concatenate the gen-token features of all hooked layers into one feature
        n_hooks = len(valid)
        fuse_hidden = max(1, int(round(fuse_hidden_mult * D)))
        self.fuse = nn.Sequential(
            nn.Linear(n_hooks * D, fuse_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(fuse_hidden, D, bias=True),
        )
        # Noised next-chunk latent -> token embedding (same patchify space as the main model, separate weights)
        self.next_patch_proj = nn.Linear(in_channels, D, bias=True)
        # Concatenate the fused feature with the next-chunk embedding to form the head input
        self.in_proj = nn.Linear(2 * D, D, bias=True)
        # Sinusoidal sigma embedding consumed by each block's AdaLN Linear
        self.sigma_embed = Timesteps(D, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.blocks = nn.ModuleList(
            _MCPBlock(
                dim=D,
                heads=heads,
                dim_head=dim_head,
                rope_type=self.rope_type,
                attention_function=attention_function,
                apply_gated_attention=apply_gated_attention,
                norm_eps=norm_eps,
            )
            for _ in range(int(num_blocks))
        )
        self.norm_out = nn.LayerNorm(D, elementwise_affine=False, eps=norm_eps)
        self.proj_out = nn.Linear(D, out_channels, bias=True)
        self.out_channels = out_channels

        self._patchifier = VideoLatentPatchifier(patch_size=1)
        self._captured: dict[int, torch.Tensor] = {}
        self._capturing = False
        self._handles = []
        self._register(transformer)

        if dtype is not None or device is not None:
            self.to(device=device, dtype=dtype)

    # ----------------------------------------------------------------- hooks
    def _register(self, transformer: nn.Module) -> None:
        for i in self.hook_indices:
            block = transformer.blocks[i]

            def make_hook(idx: int):
                def hook(module, inputs, output):
                    if self._capturing:
                        self._captured[idx] = output[0] if isinstance(output, tuple) else output
                    return None
                return hook

            self._handles.append(block.register_forward_hook(make_hook(i)))

    @contextmanager
    def capturing(self):
        """Enter capture mode: clear the cache and set the flag; on exit clear the flag but keep the cache for compute_loss."""
        self._captured = {}
        self._capturing = True
        try:
            yield
        finally:
            self._capturing = False

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------ rope freqs
    def _build_freqs(self, next_indices_grid: torch.Tensor, fps: float, dtype, device):
        """RoPE frequencies for the next-chunk tokens. next_indices_grid: [B,3,N,2] in latent coordinates.

        Mirrors the main model's positions-to-frequencies path (pixel coords + fps normalization +
        precompute_freqs_cis); only the temporal index differs (current target + K).
        """
        scale_factors = SpatioTemporalScaleFactors(time=8, width=32, height=32)
        coords = get_pixel_coords(
            latent_coords=next_indices_grid.to(device=device, dtype=torch.float32),
            scale_factors=scale_factors,
            causal_fix=True,
        ).float()
        if self.normalize_time_by_fps:
            coords[:, 0, ...] = coords[:, 0, ...] / fps
        positions = coords.to(dtype)
        return precompute_freqs_cis(
            indices_grid=positions,
            dim=self.inner_dim,
            out_dtype=dtype,
            theta=self.theta,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_heads,
            rope_type=self.rope_type,
            freq_grid_generator=self._freq_grid_generator,
            normalize_positions=self.normalize_rope_positions,
        )

    # --------------------------------------------------------------- forward
    def compute_loss(
        self,
        *,
        x0_next: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
        next_indices_grid: torch.Tensor,
        fps: float,
        K: int,
        H: int,
        W: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Flow-matching MSE: predict the next-chunk velocity from the captured generator features.

        x0_next: clean next-chunk ground truth [B,C,K,H,W] (detached).
        noise:   noise added to the next chunk (same shape as x0_next).
        sigma:   [1] tensor, the noise level for the next chunk (sampled by the trainer).
        """
        n_gen = int(K * H * W)
        # Every hook must have fired during this generator forward, otherwise the capture is incomplete。
        if len(self._captured) < len(self.hook_indices):
            raise RuntimeError(
                f"next_forcing: captured {len(self._captured)}/{len(self.hook_indices)} hook blocks "
                f"(got {sorted(self._captured)}, want {self.hook_indices}); hooks did not fire or capturing() did not wrap the forward?"
            )
        # 1) fuse the gen-token features of all hooked layers
        feats = []
        for i in self.hook_indices:
            f = self._captured.get(i)
            if f is None:
                raise RuntimeError(f"next_forcing hook block {i} produced no capture; was capturing() active?")
            if not f.requires_grad:
                # A non-gradient forward was captured (wrong grad step, or gradient checkpointing with use_reentrant=True)。
                raise RuntimeError(
                    f"next_forcing capture block {i} has requires_grad=False - a non-gradient forward was captured, "
                    "so this supervision would not reach the generator (check grad_step and use_reentrant=False)."
                )
            tgt = f[:, -n_gen:, :]
            if tgt.shape[1] != n_gen:
                raise RuntimeError(f"next_forcing capture target tokens {tgt.shape[1]} != K*H*W={n_gen} (block {i})")
            feats.append(tgt)
        h_fuse = self.fuse(torch.cat(feats, dim=-1))  # [B, n_gen, D]
        dtype = h_fuse.dtype
        device = h_fuse.device

        # 2) noise and embed the next chunk
        x0_next = x0_next.to(device=device, dtype=dtype)
        noise = noise.to(device=device, dtype=dtype)
        sigma_f = float(sigma.reshape(-1)[0].item())
        noisy = (1.0 - sigma_f) * x0_next + sigma_f * noise
        B, C = noisy.shape[0], noisy.shape[1]
        next_tokens = self._patchifier.patchify(noisy).to(dtype)  # [B, n_gen, C]
        next_emb = self.next_patch_proj(next_tokens)              # [B, n_gen, D]

        # 3) concatenate and run the denoising head (sigma-AdaLN + self-attention + FFN) x num_blocks
        x = self.in_proj(torch.cat([h_fuse, next_emb], dim=-1))   # [B, n_gen, D]
        t = (sigma.reshape(1).to(device=device, dtype=torch.float32) * 1000.0).expand(B)
        sigma_emb = self.sigma_embed(t).to(dtype)                 # [B, D]
        freqs = self._build_freqs(next_indices_grid, fps, dtype, device)
        for blk in self.blocks:
            x = blk(x, sigma_emb, freqs)

        # 4) project to the velocity field and unpatchify back to [B,C,K,H,W]
        out = self.proj_out(self.norm_out(x))                     # [B, n_gen, out_channels]
        v_pred = self._patchifier.unpatchify(
            out, VideoLatentShape(batch=B, channels=self.out_channels, frames=K, height=H, width=W)
        )

        # flow-matching target: v = noise - x0 (same convention as the critic loss)
        target = (noise - x0_next).to(dtype=v_pred.dtype)
        loss = torch.mean((v_pred.float() - target.float()) ** 2)
        return loss, {"mcp_sigma": sigma_f, "mcp_loss": float(loss.item())}
