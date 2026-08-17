# Copyright 2024-2025 LTX-2 Refactored (WAN-style)
"""LTX-2 shared configuration following WAN style."""
import torch
from easydict import EasyDict

# ------------------------ LTX-2 shared config ------------------------#
ltx2_shared_cfg = EasyDict()

# text encoder (Gemma)
ltx2_shared_cfg.text_encoder_model = 'gemma-3-12b-it'
ltx2_shared_cfg.text_encoder_dtype = torch.bfloat16
ltx2_shared_cfg.text_len = 1024
ltx2_shared_cfg.text_dim = 3840  # Gemma caption channels

# transformer
ltx2_shared_cfg.param_dtype = torch.bfloat16
ltx2_shared_cfg.patch_size = (1, 1, 1)
ltx2_shared_cfg.in_dim = 128
ltx2_shared_cfg.dim = 4096  # 32 heads * 128 head_dim
ltx2_shared_cfg.ffn_dim = 16384  # 4x dim
ltx2_shared_cfg.freq_dim = 256
ltx2_shared_cfg.out_dim = 128
ltx2_shared_cfg.num_heads = 32
ltx2_shared_cfg.num_layers = 48
ltx2_shared_cfg.window_size = (-1, -1)
ltx2_shared_cfg.qk_norm = True
ltx2_shared_cfg.cross_attn_norm = True
ltx2_shared_cfg.eps = 1e-6

# VAE scale factors
ltx2_shared_cfg.vae_stride = (8, 32, 32)  # temporal, height, width
ltx2_shared_cfg.latent_channels = 128

# inference
ltx2_shared_cfg.num_train_timesteps = 1000
ltx2_shared_cfg.sample_fps = 24
ltx2_shared_cfg.sample_neg_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

# scheduler
ltx2_shared_cfg.timestep_shift = 2.05
ltx2_shared_cfg.base_shift = 0.95
ltx2_shared_cfg.base_shift_anchor = 1024
ltx2_shared_cfg.max_shift_anchor = 4096

# 8-step distilled schedule with official sigma values (0-1 range)
# These are the optimized sigma values for distilled 8-step inference
# NOT uniform! Optimized by LTX-2 distillation training
ltx2_shared_cfg.distilled_sigma_values = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

# positional embedding
ltx2_shared_cfg.positional_embedding_theta = 10000.0
ltx2_shared_cfg.positional_embedding_max_pos = [20, 2048, 2048]

# audio transformer
ltx2_shared_cfg.audio_in_dim = 128
ltx2_shared_cfg.audio_dim = 2048  # 32 heads * 64 head_dim
ltx2_shared_cfg.audio_ffn_dim = 8192  # 4x audio_dim
ltx2_shared_cfg.audio_out_dim = 128
ltx2_shared_cfg.audio_num_heads = 32
ltx2_shared_cfg.audio_text_dim = 3840  # Same as video text_dim
ltx2_shared_cfg.audio_positional_embedding_max_pos = [20]

# model type
ltx2_shared_cfg.model_type = 'audio_video'  # 'audio_video', 'video_only', 'audio_only'


# ------------------------ LTX-2 19B config ------------------------#
ltx2_19B = EasyDict(__name__='Config: LTX-2 19B')
ltx2_19B.update(ltx2_shared_cfg)

# model paths
ltx2_19B.model_checkpoint = 'LTX-2/ltx-2-19b-dev.safetensors'
ltx2_19B.gemma_path = 'LTX-2/google/gemma-3-12b-it-qat-q4_0-unquantized'

# VAE
ltx2_19B.vae_checkpoint = 'Wan2.1_VAE.pth'


# Helper class for backward compatibility
class SpatioTemporalScaleFactors:
    """Spatiotemporal downscaling between pixel and latent space."""
    
    def __init__(self, time=8, height=32, width=32):
        self.time = time
        self.height = height
        self.width = width
    
    @classmethod
    def default(cls):
        return cls(time=8, width=32, height=32)


# Scale factors instance
ltx2_shared_cfg.scale_factors = SpatioTemporalScaleFactors.default()
