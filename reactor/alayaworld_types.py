"""Define AlayaWorld configuration and the public Reactor schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)


@dataclass(frozen=True)
class Asset:
    """Describe one model asset pinned to a public repository revision."""

    path: Path
    repo_id: str
    revision: str


@dataclass(frozen=True)
class AlayaWorldConfig:
    """Hold validated source, asset, inference, and interaction settings."""

    repo_root: Path
    inference_config: Path
    upload_template: Path
    random_inputs: tuple[Path, ...]
    model: Asset
    gemma: Asset
    da3_source_path: Path
    da3_source_url: str
    da3_source_revision: str
    da3_model: Asset
    da3_cache: Path
    seed: int
    compile_mode: str
    warmup_chunks: int
    attention_backend: str
    flex_attention: bool
    ttc: bool
    bank_taehv: bool
    taehv_path: Path | None
    taehv_source_path: Path | None
    taehv_source_url: str | None
    taehv_source_revision: str | None
    decode_overlap_latents: int
    max_spatial_frames: int
    recent_spatial_frames: int
    max_chunks_per_rollout: int
    strafe_units_per_second: float
    vertical_units_per_second: float
    forward_units_per_second: float
    pitch_degrees_per_second: float
    yaw_degrees_per_second: float
    roll_degrees_per_second: float


class AlayaWorldOutput(Output):
    """Stream one generated RGB frame on `main_video`."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Emitted when observable session state changes or a viewer connects."""

    image_source: Literal["uploaded", "built_in"] | None = MessageField(
        description=(
            'Source of the rollout image: "uploaded", "built_in", or null before '
            "`set_image` or `random_image` succeeds."
        )
    )
    image_name: str | None = MessageField(
        description="Selected image filename, or null before the session has an image."
    )
    prompt: str | None = MessageField(
        description=(
            "Scene prompt queued for the next 32-frame chunk, or null before an image is selected."
        )
    )
    active_prompt: str | None = MessageField(
        description=(
            "Scene prompt used by the most recently completed chunk, or null before the "
            "rollout starts."
        )
    )
    seed: int = MessageField(
        description="Random seed used when the current rollout is initialized or reset."
    )
    paused: bool = MessageField(
        description="Whether automatic chunk generation is paused before the next chunk."
    )
    step_queued: bool = MessageField(
        description="Whether `step` has queued one 32-frame chunk while paused."
    )
    reset_queued: bool = MessageField(
        description="Whether the selected image, prompt, and seed will start a fresh rollout."
    )
    generating: bool = MessageField(
        description="Whether the model is currently resetting or generating a chunk."
    )
    completed_chunks: int = MessageField(
        description="Number of 32-frame chunks completed since the latest rollout reset."
    )
    next_chunk: int | None = MessageField(
        description=(
            "One-based chunk that will sample the queued prompt and camera motion, or null "
            "before an image is selected."
        )
    )
    max_chunks: int = MessageField(
        description="Chunk count that triggers an automatic fresh rollout from the same image."
    )
    strafe: float = MessageField(
        description="Left (-1) to right (1) velocity queued for `next_chunk`; zero is neutral."
    )
    vertical: float = MessageField(
        description="Down (-1) to up (1) velocity queued for `next_chunk`; zero is neutral."
    )
    forward: float = MessageField(
        description=(
            "Backward (-1) to forward (1) velocity queued for `next_chunk`; zero is neutral."
        )
    )
    pitch: float = MessageField(
        description="Look down (-1) to up (1) velocity queued for `next_chunk`; zero is neutral."
    )
    yaw: float = MessageField(
        description="Turn left (-1) to right (1) velocity queued for `next_chunk`; zero is neutral."
    )
    roll: float = MessageField(
        description=(
            "Counterclockwise (-1) to clockwise (1) velocity queued for `next_chunk`; "
            "zero is neutral."
        )
    )

    @classmethod
    def from_state(
        cls,
        state: AlayaWorldState,
        *,
        image_source: Literal["uploaded", "built_in"] | None,
        image_name: str | None,
        active_prompt: str | None,
        seed: int,
        generating: bool,
        completed_chunks: int,
        next_chunk: int | None,
        max_chunks: int,
    ) -> StateUpdate:
        """Build a client snapshot from public state and rollout position."""
        return cls(
            image_source=image_source,
            image_name=image_name,
            prompt=state.prompt.strip() or None,
            active_prompt=active_prompt,
            seed=seed,
            paused=state.paused,
            step_queued=state._step_requested,
            reset_queued=state._reset_requested,
            generating=generating,
            completed_chunks=completed_chunks,
            next_chunk=next_chunk,
            max_chunks=max_chunks,
            strafe=state.strafe,
            vertical=state.vertical,
            forward=state.forward,
            pitch=state.pitch,
            yaw=state.yaw,
            roll=state.roll,
        )


class ImageSelected(ModelMessage):
    """Emitted when `set_image` or `random_image` queues a fresh rollout."""

    source: Literal["uploaded", "built_in"] = MessageField(
        description=(
            'Selected image source: "uploaded" for `set_image` or "built_in" for `random_image`.'
        )
    )
    filename: str = MessageField(
        description="Selected image filename displayed by the client for the fresh rollout."
    )
    prompt: str = MessageField(
        description="Effective non-empty scene prompt that the fresh rollout will use."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk affected by the image selection; always 1 for a fresh rollout."
    )


class PromptQueued(ModelMessage):
    """Emitted when `set_prompt` queues text for a forthcoming chunk."""

    prompt: str = MessageField(
        description="Trimmed, non-empty scene prompt queued by `set_prompt`."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk in the active rollout that will first use `prompt`."
    )


class CameraMotionChanged(ModelMessage):
    """Emitted when a camera command changes motion queued for a chunk."""

    strafe: float = MessageField(
        description="Left (-1) to right (1) velocity held until another camera command changes it."
    )
    vertical: float = MessageField(
        description="Down (-1) to up (1) velocity held until another camera command changes it."
    )
    forward: float = MessageField(
        description=(
            "Backward (-1) to forward (1) velocity held until another camera command changes it."
        )
    )
    pitch: float = MessageField(
        description=(
            "Look down (-1) to up (1) velocity held until another camera command changes it."
        )
    )
    yaw: float = MessageField(
        description=(
            "Turn left (-1) to right (1) velocity held until another camera command changes it."
        )
    )
    roll: float = MessageField(
        description=(
            "Counterclockwise (-1) to clockwise (1) velocity held until another camera "
            "command changes it."
        )
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk that will first sample the complete six-axis motion."
    )


class PauseChanged(ModelMessage):
    """Emitted when `set_paused` changes playback and releases camera motion."""

    paused: bool = MessageField(
        description="Whether automatic generation will stop before the next chunk."
    )
    camera_motion_released: bool = MessageField(
        description="Whether all six persistent camera velocities were reset to zero; always true."
    )


class StepQueued(ModelMessage):
    """Emitted when `step` queues one chunk while generation is paused."""

    applies_to_chunk: int = MessageField(
        description="One-based chunk in the active rollout that the queued step will generate."
    )


class RolloutResetQueued(ModelMessage):
    """Emitted when a manual or automatic reset queues a fresh rollout."""

    trigger: Literal["manual", "automatic_chunk_limit"] = MessageField(
        description=(
            'Reset source: "manual" for `reset` or "automatic_chunk_limit" after '
            "`max_chunks` completes."
        )
    )
    seed: int = MessageField(
        description="Random seed that will initialize the queued fresh rollout."
    )
    completed_chunks: int = MessageField(
        description="Number of completed chunks in the rollout being replaced."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk affected by the reset; always 1 for a fresh rollout."
    )


class AlayaWorldState(InputState):
    """Expose controls shared by one playable AlayaWorld session."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        description=(
            "Scene prompt queued for the next 32-frame chunk. Requires a selected image; "
            "whitespace-only values are rejected by `set_prompt`."
        ),
    )
    forward: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Backward (-1) to forward (1) velocity sampled at each chunk boundary and held "
            "until changed; zero is neutral."
        ),
    )
    strafe: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Left (-1) to right (1) velocity sampled at each chunk boundary and held until "
            "changed; zero is neutral."
        ),
    )
    vertical: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Down (-1) to up (1) velocity sampled at each chunk boundary and held until "
            "changed; zero is neutral."
        ),
    )
    pitch: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Look down (-1) to up (1) velocity sampled at each chunk boundary and held until "
            "changed; zero is neutral."
        ),
    )
    yaw: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Turn left (-1) to right (1) velocity sampled at each chunk boundary and held "
            "until changed; zero is neutral."
        ),
    )
    roll: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Counterclockwise (-1) to clockwise (1) velocity sampled at each chunk boundary "
            "and held until changed; zero is neutral."
        ),
    )
    paused: bool = InputField(
        default=True,
        description=(
            "Pause before the next chunk while preserving the current world. `step` can queue "
            "one chunk while paused; `set_paused` releases all camera motion."
        ),
    )
    _step_requested: bool = False
    _reset_requested: bool = False
