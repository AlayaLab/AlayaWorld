from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import safetensors.torch


class LoRAForwardManager:
    """LoRA via forward hooks, kept outside the base transformer's state_dict."""

    def __init__(self, trainable: bool) -> None:
        self.trainable = trainable
        self.enabled = False
        self.lora_dict: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.hooks = []
        self._registered = False

    def init_for_training(
        self,
        model: nn.Module,
        target_keywords: list[str],
        rank: int,
        alpha: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> int:
        scaling = alpha / float(rank)
        self.lora_dict.clear()

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(keyword in name for keyword in target_keywords):
                continue

            lora_a = torch.empty(rank, module.in_features, dtype=dtype, device=device)
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
            lora_a = lora_a * scaling
            lora_b = torch.zeros(module.out_features, rank, dtype=dtype, device=device)
            if self.trainable:
                lora_a = nn.Parameter(lora_a)
                lora_b = nn.Parameter(lora_b)
            self.lora_dict[name] = (lora_a, lora_b)
        return len(self.lora_dict)

    def register_hooks(self, model: nn.Module) -> int:
        if self._registered:
            return 0
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self.lora_dict:
                self.hooks.append(module.register_forward_hook(self._make_hook(name)))
                count += 1
        self._registered = True
        return count

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            if not self.enabled:
                return output
            x = inputs[0]
            lora_a, lora_b = self.lora_dict[name]
            lora_a = lora_a.to(device=x.device, dtype=x.dtype)
            lora_b = lora_b.to(device=x.device, dtype=x.dtype)
            if not self.trainable:
                lora_a = lora_a.detach()
                lora_b = lora_b.detach()
            return output + x @ lora_a.T @ lora_b.T

        return hook

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    @contextmanager
    def toggled(self, enabled: bool):
        """Temporarily enable/disable LoRA; the previous state is restored on exit.

        Used by the DMD real-score forward (which must run the base weights) and by
        consistency distillation, which disables the generator LoRA to act as a frozen teacher.
        """
        prev = self.enabled
        self.enabled = bool(enabled)
        try:
            yield
        finally:
            self.enabled = prev

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        if not self.trainable:
            return []
        params = []
        for lora_a, lora_b in self.lora_dict.values():
            params.extend([lora_a, lora_b])
        return params

    def state_dict(self) -> dict[str, torch.Tensor]:
        state = {}
        for name, (lora_a, lora_b) in self.lora_dict.items():
            key = name.replace("blocks.", "transformer_blocks.")
            state[f"diffusion_model.{key}.lora_A.weight"] = lora_a.detach().cpu()
            state[f"diffusion_model.{key}.lora_B.weight"] = lora_b.detach().cpu()
        return state

    def save(self, path: str) -> None:
        safetensors.torch.save_file(self.state_dict(), path)

    def load(self, path: str) -> int:
        state = safetensors.torch.load_file(path, device="cpu")
        loaded = 0
        for key, tensor in state.items():
            if not key.startswith("diffusion_model."):
                continue
            if key.endswith(".lora_A.weight"):
                name = key.removeprefix("diffusion_model.").removesuffix(".lora_A.weight")
                slot = 0
            elif key.endswith(".lora_B.weight"):
                name = key.removeprefix("diffusion_model.").removesuffix(".lora_B.weight")
                slot = 1
            else:
                continue

            name = name.replace("transformer_blocks.", "blocks.")
            if name not in self.lora_dict:
                raise KeyError(f"LoRA checkpoint contains unknown target: {name}")

            target = self.lora_dict[name][slot]
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"LoRA shape mismatch for {name}: checkpoint={tuple(tensor.shape)} "
                    f"model={tuple(target.shape)}"
                )
            target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded += 1
        return loaded
