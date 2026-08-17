from __future__ import annotations

from pathlib import Path
import re
import shutil

import safetensors.torch
import torch

from alaya.config.schema import TrainConfig
from alaya.utils.distributed import DistributedState, barrier


def save_checkpoint(
    *,
    cfg: TrainConfig,
    dist_state: DistributedState,
    step: int,
    transformer: torch.nn.Module,
    history_encoder: torch.nn.Module | None,
    lora_manager=None,
    critic_lora=None,
    gan_discriminator=None,
    next_forcing_head=None,
) -> None:
    transformer_state = None
    if cfg.training.mode == "sft":
        transformer_state = _collect_transformer_state_dict(transformer, dist_state)

    if not dist_state.is_main:
        return

    ckpt_dir = Path(cfg.run.output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if history_encoder is not None:
        torch.save(history_encoder.state_dict(), ckpt_dir / "history_encoder.pt")
    if transformer_state is not None:
        torch.save(transformer_state, ckpt_dir / "transformer.pt")
    torch.save({"step": step, "training_mode": cfg.training.mode}, ckpt_dir / "trainer_state.pt")
    if lora_manager is not None:
        lora_manager.save(str(ckpt_dir / "lora.safetensors"))
    if critic_lora is not None:
        critic_lora.save(str(ckpt_dir / "critic_lora.safetensors"))
    if gan_discriminator is not None:
        torch.save(
            {k: v.detach().cpu() for k, v in gan_discriminator.state_dict().items()},
            ckpt_dir / "gan_discriminator.pt",
        )
    if next_forcing_head is not None:
        torch.save(
            {k: v.detach().cpu() for k, v in next_forcing_head.state_dict().items()},
            ckpt_dir / "next_forcing.pt",
        )
    action_state = {}
    if transformer_state is None:
        for name, param in transformer.named_parameters():
            if "action_adaln_embedder" in name or "action_adaln_projection" in name:
                action_state[_clean_state_name(name)] = param.detach().cpu()
    if action_state:
        torch.save(action_state, ckpt_dir / "action_adaln.pt")

    marker = ckpt_dir / "README.txt"
    marker.write_text(
        "Clean Alaya checkpoint.\n"
        "Saved: trainer_state.pt, optional history_encoder.pt, optional lora.safetensors, optional next_forcing.pt.\n"
        "DMD runs also save critic_lora.safetensors and optional gan_discriminator.pt.\n"
        "SFT mode saves transformer.pt, including action AdaLN weights.\n",
        encoding="utf-8",
    )
    _prune_old_checkpoints(Path(cfg.run.output_dir), keep=cfg.optimizer.max_checkpoints)
    print(f"[Checkpoint] saved {ckpt_dir}", flush=True)


def load_checkpoint_weights(
    *,
    cfg: TrainConfig,
    dist_state: DistributedState,
    transformer: torch.nn.Module,
    history_encoder: torch.nn.Module | None,
    lora_manager=None,
    critic_lora=None,
    score_model: torch.nn.Module | None = None,
    gan_discriminator: torch.nn.Module | None = None,
    next_forcing_head: torch.nn.Module | None = None,
) -> int:
    if not cfg.paths.resume_checkpoint:
        return 0

    ckpt_dir = Path(cfg.paths.resume_checkpoint)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {ckpt_dir}")
    if not ckpt_dir.is_dir():
        raise ValueError(f"paths.resume_checkpoint must point to a checkpoint directory: {ckpt_dir}")

    if history_encoder is not None:
        if dist_state.is_main:
            print(
                "[Resume] keep history encoder initialized from paths.history_encoder; "
                "resume checkpoint history_encoder.pt is ignored",
                flush=True,
            )

    transformer_path = ckpt_dir / "transformer.pt"
    loaded_transformer = False
    if transformer_path.exists():
        state = _torch_load(transformer_path, map_location="cpu")
        missing, unexpected = transformer.load_state_dict(state, strict=False)
        loaded_transformer = True
        if dist_state.is_main:
            print(
                f"[Resume] transformer loaded={len(state)} missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
        # When paths.real_score_model is set the teacher uses its own weights and must not be overwritten by the SFT base。
        if score_model is not None and not cfg.paths.real_score_model:
            s_missing, s_unexpected = score_model.load_state_dict(state, strict=False)
            if dist_state.is_main:
                print(
                    f"[Resume] DMD score model loaded from transformer.pt: "
                    f"loaded={len(state)} missing={len(s_missing)} unexpected={len(s_unexpected)}",
                    flush=True,
                )
        elif score_model is not None and dist_state.is_main:
            print(
                f"[Resume] DMD score model (teacher) keeps paths.real_score_model weights "
                f"({cfg.paths.real_score_model}); transformer.pt NOT loaded into teacher",
                flush=True,
            )
    else:
        diffusion_path = ckpt_dir / "diffusion_pytorch_model.safetensors"
        if diffusion_path.exists():
            from alaya.model.loader import convert_transformer_state_dict

            state = safetensors.torch.load_file(str(diffusion_path), device="cpu")
            model_keys = {name for name, _ in transformer.named_parameters()}
            model_keys.update(name for name, _ in transformer.named_buffers())
            converted = convert_transformer_state_dict(state, model_keys)
            missing, unexpected = transformer.load_state_dict(converted, strict=False)
            loaded_transformer = True
            if dist_state.is_main:
                print(
                    f"[Resume] diffusion transformer loaded={len(converted)} "
                    f"missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
            # Same as above: with paths.real_score_model the diffusion base must not overwrite the teacher either.
            if score_model is not None and not cfg.paths.real_score_model:
                s_missing, s_unexpected = score_model.load_state_dict(converted, strict=False)
                if dist_state.is_main:
                    print(
                        f"[Resume] DMD score model loaded from diffusion transformer: "
                        f"loaded={len(converted)} missing={len(s_missing)} unexpected={len(s_unexpected)}",
                        flush=True,
                    )
            elif score_model is not None and dist_state.is_main:
                print(
                    f"[Resume] DMD score model (teacher) keeps paths.real_score_model weights "
                    f"({cfg.paths.real_score_model}); diffusion base NOT loaded into teacher",
                    flush=True,
                )

    adapter_dir = Path(cfg.paths.dmd_resume) if cfg.paths.dmd_resume else ckpt_dir
    if cfg.paths.dmd_resume:
        if not adapter_dir.is_dir():
            raise ValueError(f"paths.dmd_resume must point to a checkpoint directory: {adapter_dir}")
        if dist_state.is_main:
            print(f"[Resume] DMD adapters/step from dmd_resume={adapter_dir} (base from {ckpt_dir})", flush=True)

    lora_path = adapter_dir / "lora.safetensors"
    if lora_path.exists():
        if lora_manager is None:
            raise RuntimeError(f"checkpoint has {lora_path.name}, but LoRA is disabled in this config")
        loaded = lora_manager.load(str(lora_path))
        if dist_state.is_main:
            print(f"[Resume] lora tensors loaded={loaded}", flush=True)

    critic_lora_path = adapter_dir / "critic_lora.safetensors"
    if critic_lora_path.exists():
        if critic_lora is None:
            raise RuntimeError(
                f"checkpoint has {critic_lora_path.name}, but DMD critic LoRA is disabled in this config"
            )
        loaded = critic_lora.load(str(critic_lora_path))
        if dist_state.is_main:
            print(f"[Resume] critic lora tensors loaded={loaded}", flush=True)

    gan_path = adapter_dir / "gan_discriminator.pt"
    if gan_path.exists() and gan_discriminator is not None:
        state = _torch_load(gan_path, map_location="cpu")
        missing, unexpected = gan_discriminator.load_state_dict(state, strict=False)
        if dist_state.is_main:
            print(
                f"[Resume] gan discriminator loaded={len(state)} "
                f"missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )

    nf_path = adapter_dir / "next_forcing.pt"
    if nf_path.exists() and next_forcing_head is not None:
        state = _torch_load(nf_path, map_location="cpu")
        missing, unexpected = next_forcing_head.load_state_dict(state, strict=False)
        if dist_state.is_main:
            print(
                f"[Resume] next_forcing head loaded={len(state)} missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )

    action_path = adapter_dir / "action_adaln.pt"
    if action_path.exists() and (adapter_dir != ckpt_dir or not loaded_transformer):
        state = _torch_load(action_path, map_location="cpu")
        state = {key: value for key, value in state.items() if value.numel() > 0}
        _, unexpected = transformer.load_state_dict(state, strict=False) if state else ([], [])
        if dist_state.is_main:
            print(
                f"[Resume] action_adaln tensors loaded={len(state)} unexpected={len(unexpected)}",
                flush=True,
            )

    trainer_state_path = adapter_dir / "trainer_state.pt"
    step = 0
    if cfg.paths.resume_reset_step:
        step = 0
    elif trainer_state_path.exists():
        trainer_state = _torch_load(trainer_state_path, map_location="cpu")
        step = int(trainer_state.get("step", 0))
    else:
        step = _infer_step_from_checkpoint_dir(adapter_dir)
        if dist_state.is_main and step > 0:
            print(
                f"[Resume] trainer_state.pt missing; inferred step={step} from {adapter_dir.name}",
                flush=True,
            )

    if dist_state.is_main:
        if cfg.paths.resume_reset_step:
            print(f"[Resume] loaded weights (base={ckpt_dir}); step reset to 0", flush=True)
        else:
            print(
                f"[Resume] loaded base={ckpt_dir} adapters={adapter_dir} at step={step}; "
                "optimizer state is intentionally not restored",
                flush=True,
            )
    barrier()
    return step


def _infer_step_from_checkpoint_dir(ckpt_dir: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)(?:-merged)?", ckpt_dir.name)
    return int(match.group(1)) if match else 0


def _collect_transformer_state_dict(
    transformer: torch.nn.Module,
    dist_state: DistributedState,
) -> dict[str, torch.Tensor] | None:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    except Exception:
        FSDP = None

    if FSDP is not None and isinstance(transformer, FSDP):
        save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(transformer, StateDictType.FULL_STATE_DICT, save_cfg):
            state = transformer.state_dict()
        return state if dist_state.is_main else None

    if not dist_state.is_main:
        return None
    return {name: tensor.detach().cpu() for name, tensor in transformer.state_dict().items()}


def _prune_old_checkpoints(output_dir: Path, *, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        checkpoints.append((step, path))
    checkpoints.sort(key=lambda item: item[0])
    for _, path in checkpoints[:-keep]:
        shutil.rmtree(path)
        print(f"[Checkpoint] pruned {path}", flush=True)


def _torch_load(path: Path, *, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _clean_state_name(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
