"""
Self-contained text encoder adapted from the public LTX-2 code.
Gemma-based text encoder.
"""
from typing import NamedTuple
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from .model import FeedForward, LTXRopeType, precompute_freqs_cis, rms_norm
# Import Attention from model_ltx_2_3 which supports apply_gated_attention (to_gate_logits)
# The LTX 2.0 Attention in model.py lacks this feature, causing silent weight drops for 2.3 checkpoints
try:
    from .model_ltx_2_3 import Attention
except ImportError:
    from .model import Attention

__all__ = ['AVGemmaTextEncoderModel', 'LTXVGemmaTokenizer']


# ============================================================
# Tokenizer
# ============================================================

class LTXVGemmaTokenizer:
    """Tokenizer wrapper."""
    def __init__(self, tokenizer_path, max_length=1024):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, model_max_length=max_length)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
    
    def tokenize_with_weights(self, text):
        encoded = self.tokenizer(text.strip(), padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        return {"gemma": [(t.item(), a.item()) for t, a in zip(encoded.input_ids[0], encoded.attention_mask[0])]}


# ============================================================
# Feature Extractor
# ============================================================

class GemmaFeaturesExtractorProjLinear(nn.Module):
    """Feature extractor: project multi-layer hidden states to the target dimension.

    V1 (19B): aggregate_embed Linear(188160, 3840, bias=False)
    V2 (22B): video_aggregate_embed Linear(188160, 4096, bias=True)
    """
    def __init__(self, out_dim=3840, bias=False, use_video_key=False):
        super().__init__()
        in_dim = 3840 * 49  # gemma hidden_size * (num_hidden_layers + 1)
        if use_video_key:
            self.video_aggregate_embed = nn.Linear(in_dim, out_dim, bias=bias)
        else:
            self.aggregate_embed = nn.Linear(in_dim, out_dim, bias=bias)
        self._use_video_key = use_video_key

    def forward(self, x):
        if self._use_video_key:
            return self.video_aggregate_embed(x)
        return self.aggregate_embed(x)


# ============================================================
# Embeddings Connector
# ============================================================

class _BasicTransformerBlock1D(nn.Module):
    """1D transformer block (matches the official implementation)."""
    def __init__(self, dim, heads, dim_head, rope_type=LTXRopeType.SPLIT, apply_gated_attention=False):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim, context_dim=None, heads=heads, dim_head=dim_head,
            rope_type=rope_type, apply_gated_attention=apply_gated_attention,
        )
        self.ff = FeedForward(dim, dim)

    def forward(self, x, attention_mask=None, pe=None):
        norm_x = rms_norm(x).squeeze(1)
        x = self.attn1(norm_x, mask=attention_mask, pe=pe) + x
        if x.ndim == 4:
            x = x.squeeze(1)
        x = self.ff(rms_norm(x)) + x
        if x.ndim == 4:
            x = x.squeeze(1)
        return x


class Embeddings1DConnector(nn.Module):
    """1D embedding connector for text embeddings (matches the official implementation)."""
    def __init__(self, attention_head_dim=128, num_attention_heads=32, num_layers=8,
                 positional_embedding_theta=10000.0, positional_embedding_max_pos=None,
                 num_learnable_registers=128, rope_type=LTXRopeType.SPLIT,
                 apply_gated_attention=True):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos or [1]
        self.rope_type = rope_type
        self.num_learnable_registers = num_learnable_registers

        self.transformer_1d_blocks = nn.ModuleList([
            _BasicTransformerBlock1D(
                self.inner_dim, num_attention_heads, attention_head_dim,
                rope_type, apply_gated_attention=apply_gated_attention,
            )
            for _ in range(num_layers)
        ])
        
        if num_learnable_registers:
            self.learnable_registers = nn.Parameter(torch.rand(num_learnable_registers, self.inner_dim, dtype=torch.bfloat16) * 2 - 1)
    
    def _replace_padded_with_learnable_registers(self, hidden_states, attention_mask):
        n = self.num_learnable_registers
        registers = torch.tile(self.learnable_registers, (hidden_states.shape[1] // n, 1))
        mask_binary = (attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) >= -9000.0).int()
        
        non_zero = hidden_states[:, mask_binary.squeeze().bool(), :]
        pad_len = hidden_states.shape[1] - non_zero.shape[1]
        adjusted = nn.functional.pad(non_zero, (0, 0, 0, pad_len), value=0)
        flipped = torch.flip(mask_binary, dims=[1])
        hidden_states = flipped * adjusted + (1 - flipped) * registers
        attention_mask = torch.zeros_like(attention_mask)
        return hidden_states, attention_mask
    
    def forward(self, hidden_states, attention_mask=None):
        if self.num_learnable_registers:
            hidden_states, attention_mask = self._replace_padded_with_learnable_registers(hidden_states, attention_mask)
        
        indices_grid = torch.arange(hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device)[None, None, :]
        pe = precompute_freqs_cis(indices_grid, self.inner_dim, hidden_states.dtype, self.positional_embedding_theta,
                                  self.positional_embedding_max_pos, False, self.num_attention_heads, self.rope_type)
        
        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, attention_mask, pe)
        
        return rms_norm(hidden_states), attention_mask


# ============================================================
# Encoder Output
# ============================================================

class AVGemmaEncoderOutput(NamedTuple):
    video_encoding: torch.Tensor
    audio_encoding: torch.Tensor
    attention_mask: torch.Tensor


# ============================================================
# AV Gemma Text Encoder
# ============================================================

def _norm_and_concat_padded_batch(encoded_text, sequence_lengths, padding_side="right"):
    """V1 normalization: per-segment mean/range normalization."""
    b, t, d, l = encoded_text.shape
    device = encoded_text.device

    token_indices = torch.arange(t, device=device)[None, :]
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    else:
        mask = token_indices >= (t - sequence_lengths[:, None])
    mask = rearrange(mask, "b t -> b t 1 1")

    eps = 1e-6
    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_ = x_max - x_min

    normed = 8 * (encoded_text - mean) / (range_ + eps)
    normed = normed.reshape(b, t, -1)

    mask_flat = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
    return normed.masked_fill(~mask_flat, 0.0)


def _norm_and_concat_per_token_rms(encoded_text, attention_mask):
    """V2 normalization: per-token RMS normalization.

    Each token is RMS-normalized independently, then all layers are concatenated.

    Args:
        encoded_text: [B, T, D, L] - hidden states from all layers
        attention_mask: [B, T] - binary attention mask
    Returns:
        [B, T, D*L] - normalized and concatenated tensor
    """
    B, T, D, L = encoded_text.shape
    variance = torch.mean(encoded_text ** 2, dim=2, keepdim=True)  # [B, T, 1, L]
    normed = encoded_text * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(B, T, D * L)
    mask_3d = attention_mask.bool().unsqueeze(-1)  # [B, T, 1]
    return torch.where(mask_3d, normed, torch.zeros_like(normed))


def _rescale_norm(x, target_dim, source_dim):
    """Rescale normalization: x * sqrt(target_dim / source_dim)"""
    import math
    return x * math.sqrt(target_dim / source_dim)


class AVGemmaTextEncoderModel(nn.Module):
    """Audio-video text encoder."""
    def __init__(self, feature_extractor_linear, embeddings_connector, audio_embeddings_connector=None,
                 tokenizer=None, model=None, dtype=torch.bfloat16,
                 use_v2_norm=False, gemma_embedding_dim=3840):
        super().__init__()
        self._gemma_root = None
        self.tokenizer = tokenizer
        self.model = model
        self.feature_extractor_linear = feature_extractor_linear.to(dtype=dtype)
        self.embeddings_connector = embeddings_connector.to(dtype=dtype)
        self.audio_embeddings_connector = audio_embeddings_connector.to(dtype=dtype) if audio_embeddings_connector is not None else None
        self.use_v2_norm = use_v2_norm
        self.gemma_embedding_dim = gemma_embedding_dim
    
    def _convert_to_additive_mask(self, attention_mask, dtype):
        return (attention_mask - 1).to(dtype).reshape(attention_mask.shape[0], 1, -1, attention_mask.shape[-1]) * torch.finfo(dtype).max
    
    def _run_feature_extractor(self, hidden_states, attention_mask, padding_side="right"):
        encoded = torch.stack(hidden_states, dim=-1)  # [B, T, D, L]
        if self.use_v2_norm:
            # V2 (22B): per-token RMS normalization + rescale
            normed = _norm_and_concat_per_token_rms(encoded, attention_mask)
            normed = normed.to(encoded.dtype)
            # Rescale: x * sqrt(target_dim / source_dim) to maintain magnitude after projection
            v_dim = self.feature_extractor_linear.video_aggregate_embed.out_features \
                if hasattr(self.feature_extractor_linear, 'video_aggregate_embed') \
                else self.feature_extractor_linear.aggregate_embed.out_features
            normed = _rescale_norm(normed, v_dim, self.gemma_embedding_dim)
        else:
            # V1 (19B): per-segment mean/range normalization
            seq_lengths = attention_mask.sum(dim=-1)
            normed = _norm_and_concat_padded_batch(encoded, seq_lengths, padding_side)
            normed = normed.to(encoded.dtype)
        return self.feature_extractor_linear(normed)
    
    def _preprocess_text(self, text, padding_side="left"):
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=self.model.device)
        attention_mask = torch.tensor([[t[1] for t in token_pairs]], device=self.model.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        projected = self._run_feature_extractor(outputs.hidden_states, attention_mask, padding_side)
        return projected, attention_mask
    
    def _run_connectors(self, encoded_input, attention_mask):
        connector_mask = self._convert_to_additive_mask(attention_mask, encoded_input.dtype)
        encoded, encoded_mask = self.embeddings_connector(encoded_input, connector_mask)
        attention_mask = (encoded_mask < 0.000001).to(torch.int64).reshape(encoded.shape[0], encoded.shape[1], 1)
        encoded = encoded * attention_mask
        if self.audio_embeddings_connector is not None:
            encoded_audio, _ = self.audio_embeddings_connector(encoded_input, connector_mask)
        else:
            encoded_audio = encoded  # video-only mode
        return encoded, encoded_audio, attention_mask.squeeze(-1)
    
    def forward(self, text, padding_side="left"):
        encoded_inputs, attention_mask = self._preprocess_text(text, padding_side)
        video_enc, audio_enc, mask = self._run_connectors(encoded_inputs, attention_mask)
        return AVGemmaEncoderOutput(video_enc, audio_enc, mask)


# ============================================================
# ============================================================

def create_av_text_encoder(config):
    """Build the text encoder from a config."""
    cfg = config.get("transformer", {})
    rope_type = LTXRopeType(cfg.get("rope_type", "interleaved"))
    pe_max_pos = cfg.get("connector_positional_embedding_max_pos", [1])
    
    feature_extractor = GemmaFeaturesExtractorProjLinear()
    embeddings_connector = Embeddings1DConnector(num_attention_heads=30, attention_head_dim=128, positional_embedding_max_pos=pe_max_pos, rope_type=rope_type)
    audio_connector = Embeddings1DConnector(num_attention_heads=30, attention_head_dim=128, positional_embedding_max_pos=pe_max_pos, rope_type=rope_type)
    
    return AVGemmaTextEncoderModel(feature_extractor, embeddings_connector, audio_connector)


def find_file(root, pattern):
    """Locate a file."""
    matches = list(Path(root).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Not found: {pattern} in {root}")
    return str(matches[0].parent)


# ============================================================
# ============================================================

class LTX2TextEncoder(nn.Module):
    """High-level text-encoder wrapper for the inference pipeline."""
    
    def __init__(self, device=None, dtype=torch.bfloat16):
        super().__init__()
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._encoder = None
        self._encode_fn = None
    
    @property
    def device(self):
        return self._device
    
    def set_encoder(self, encoder, encode_fn=None):
        """Set up the encoder."""
        self._encoder = encoder
        self._encode_fn = encode_fn
    
    def forward(self, text_prompts):
        """Encode text prompts."""
        if self._encoder is None:
            raise RuntimeError("Encoder not set")
        
        results = []
        for prompt in text_prompts:
            output = self._encoder(prompt)
            # output.video_encoding is [1, seq_len, dim], squeeze batch dim
            results.append({
                "prompt_embeds": output.video_encoding.squeeze(0),  # [seq_len, dim]
                "audio_embeds": output.audio_encoding.squeeze(0) if output.audio_encoding is not None else None,
                "attention_mask": output.attention_mask.squeeze(0) if output.attention_mask is not None else None,
            })
        
        # Stack batch: [B, seq_len, dim]
        batch = {
            "prompt_embeds": torch.stack([r["prompt_embeds"] for r in results]),
        }
        if results[0]["audio_embeds"] is not None:
            batch["audio_embeds"] = torch.stack([r["audio_embeds"] for r in results])
        return batch
