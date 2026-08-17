from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedState:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistributedState:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        timeout_seconds = int(os.environ.get("ALAYA_DISTRIBUTED_TIMEOUT_SECONDS", "3600"))
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    return DistributedState(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def broadcast_tensor(value: torch.Tensor, src: int = 0) -> torch.Tensor:
    if dist.is_initialized():
        dist.broadcast(value, src=src)
    return value


def rank0_print(state: DistributedState, *values, **kwargs) -> None:
    if state.is_main:
        print(*values, **kwargs, flush=True)
