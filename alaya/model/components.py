from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn


@dataclass
class ModelComponents:
    transformer: nn.Module
    vae_encoder: object
    vae_decoder: nn.Module
    text_encoder: nn.Module
    encode_text: Callable
    lora_manager: object | None = None
    score_model: nn.Module | None = None
    critic_lora: object | None = None
    gan_discriminator: nn.Module | None = None
    next_forcing_head: nn.Module | None = None
