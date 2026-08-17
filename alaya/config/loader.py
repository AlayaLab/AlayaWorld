from __future__ import annotations

from pathlib import Path

from .schema import TrainConfig


def load_config(path: str | Path) -> TrainConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return TrainConfig.from_mapping(raw)

