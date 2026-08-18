from __future__ import annotations

import math
from typing import Sequence


def shard_indices_with_padding(
    indices: Sequence[int],
    *,
    rank: int,
    world_size: int,
) -> list[int | None]:
    """Give every rank the same number of slots; None marks sync-only padding."""
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if not indices:
        return []

    slots_per_rank = int(math.ceil(len(indices) / world_size))
    start = rank * slots_per_rank
    assigned = [int(x) for x in indices[start : start + slots_per_rank]]
    assigned.extend([None] * (slots_per_rank - len(assigned)))
    return assigned
