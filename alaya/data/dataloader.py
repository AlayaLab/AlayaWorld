from __future__ import annotations

import os
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from fastvideo.dataset.t2v_datasets import MultiSourceVideoDataset, WeightedConcatDataset

from alaya.config.schema import TrainConfig, ValidationModeConfig
from alaya.utils.distributed import DistributedState


SOURCE_ALIASES = {
    "sekai_real": ["sekai_real_walking"],
    "sekai_real_mini": ["sekai_real_mini"],
    "sekai_game": ["sekai_game_walking"],
    "mugen_v2": ["mugen_v2"],
    "spatial_estate": ["spatialvid", "RealEstate10K"],
    "mem_event_phy": ["veo3"],
    "others": ["OpenVid"],
}


def build_train_dataloader(cfg: TrainConfig, dist_state: DistributedState) -> DataLoader:
    _configure_dataset_env(cfg)
    k_values = [int(k) for k in cfg.layout.output.latent_frames]
    k_probs = [float(p) for p in cfg.layout.output.probs]
    if len(k_values) != len(k_probs):
        raise ValueError("layout.output.latent_frames and layout.output.probs must have the same length")
    active = [(k, p) for k, p in zip(k_values, k_probs) if p > 0.0]
    if not active:
        raise ValueError("layout.output.probs must contain at least one positive value")
    k_values = [k for k, _ in active]
    k_probs = [p for _, p in active]

    if len(k_values) > 1:
        return SyncedKTrainLoader(cfg, dist_state, k_values, k_probs)

    train_frames = _required_train_frames_for_k(cfg, k_values[0])
    pose_extra_frames = _pose_extra_frames_for_self_rollout(cfg, k_values[0])
    train_frames += _pixel_extra_frames_for_self_rollout(cfg, k_values[0])

    _enable_valid_starts = (
        (int(k_values[0]) == 8 and bool(cfg.layout.k8_use_valid_starts))
        or (int(k_values[0]) == 4 and bool(cfg.layout.k4_use_valid_starts))
    )
    _roll_layout = _enable_valid_starts
    _max_gap_latents = int(cfg.layout.max_gap_sec * cfg.sample.fps / cfg.sample.temporal_stride)
    _min_gap_latents_hc = _min_gap_steps_for_target_prefix_context(cfg, cond_end=0)
    _min_gap_latents_i2v = _min_gap_steps_for_target_prefix_context(cfg, cond_end=1)
    _valid_starts_anchor_offset = 0

    datasets = []
    for source_name, weight in cfg.data.sources.items():
        if weight <= 0:
            continue
        sources = SOURCE_ALIASES.get(source_name, [source_name])
        dataset = MultiSourceVideoDataset(
            video_base_dir=cfg.paths.video_base_dir,
            annotation_base_dir=cfg.paths.annotation_base_dir,
            sources=sources,
            width=cfg.sample.width,
            height=cfg.sample.height,
            target_fps=cfg.sample.fps,
            min_frames=train_frames,
            max_frames=train_frames,
            allow_short_samples=cfg.layout.variable_length,
            vae_grid_align=bool(cfg.runtime.vae_latent_cache_dir),
            random_frames=True,
            use_cache=cfg.data.use_cache,
            skip_file_check=cfg.data.skip_file_check,
            abstract_caption_prob=cfg.data.abstract_caption_prob,
            return_raw_pose=False,
            require_camera=cfg.data.require_camera,
            camera_norm_mode=cfg.data.camera_norm_mode,
            camera_post_relic_scale=cfg.data.camera_post_relic_scale,
            vae_temporal_factor=cfg.sample.temporal_stride,
            cp_size=1,
            output_latent_frames=int(k_values[0]),
            use_valid_starts=_enable_valid_starts,
            valid_starts_anchor_offset=_valid_starts_anchor_offset,
            roll_layout=_roll_layout,
            max_gap_latents=_max_gap_latents,
            min_gap_latents_hc=_min_gap_latents_hc,
            min_gap_latents_i2v=_min_gap_latents_i2v,
            i2v_prob=float(cfg.layout.condition.i2v_prob),
            sink_remote=bool(cfg.layout.sink_remote),
            sink_remote_min_distance=int(cfg.layout.sink_remote_min_distance),
            sink_latent_frames=int(cfg.layout.sink_latent_frames),
            event_target_anchor_frame=_training_event_target_anchor_frame(cfg, int(k_values[0])),
            pose_extra_frames=pose_extra_frames,
        )
        _apply_min_frame_filter(dataset, train_frames + pose_extra_frames, label=f"train:{source_name}:K{k_values[0]}")
        if cfg.data.max_samples_per_source is not None:
            dataset.samples = dataset.samples[: int(cfg.data.max_samples_per_source)]
        datasets.append((dataset, float(weight), source_name))
        if dist_state.is_main:
            print(
                f"[Data] {source_name}: real={len(dataset)} weight={weight} "
                f"valid_starts={_enable_valid_starts} roll_layout={_roll_layout} "
                f"sink_remote={cfg.layout.sink_remote} "
                f"min_gap_hc={_min_gap_latents_hc} min_gap_i2v={_min_gap_latents_i2v}", flush=True,
            )

    if not datasets:
        raise ValueError("no enabled data sources")

    merged = WeightedConcatDataset(datasets) if len(datasets) > 1 else datasets[0][0]
    sampler = DistributedSampler(
        merged,
        num_replicas=dist_state.world_size,
        rank=dist_state.rank,
        shuffle=True,
        drop_last=True,
    )
    return DataLoader(
        merged,
        batch_size=cfg.optimizer.batch_size,
        sampler=sampler,
        num_workers=cfg.runtime.dataloader_workers,
        pin_memory=cfg.runtime.dataloader_pin_memory,
        drop_last=True,
        timeout=600 if cfg.runtime.dataloader_workers > 0 else 0,
        persistent_workers=cfg.runtime.dataloader_workers > 0,
        prefetch_factor=cfg.runtime.dataloader_prefetch_factor if cfg.runtime.dataloader_workers > 0 else None,
    )


class SyncedKTrainLoader:
    """Dataloader wrapper that samples output K before loading frames.

    A plain Dataset cannot independently choose K on each rank because the
    trainer broadcasts tensors whose shape depends on K.  This wrapper samples
    K on rank 0, broadcasts that choice, then consumes the matching per-K
    dataloader on every rank.  Each per-K dataset is built with the frame
    window required for that K, so short-K crops are not forced to use the
    longest configured window.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        dist_state: DistributedState,
        k_values: list[int],
        k_probs: list[float],
    ) -> None:
        self.cfg = cfg
        self.dist_state = dist_state
        total = sum(k_probs)
        if total <= 0:
            raise ValueError("layout.output.probs must sum to a positive value")
        self.k_values = k_values
        self.k_probs = [p / total for p in k_probs]
        self.loaders = {
            k: _build_fixed_k_train_dataloader(cfg, dist_state, k)
            for k in self.k_values
        }
        self.samplers = {
            k: loader.sampler
            for k, loader in self.loaders.items()
            if hasattr(loader, "sampler")
        }
        self._epoch = 0
        self._steps_per_epoch = max(len(loader) for loader in self.loaders.values())

    def __len__(self) -> int:
        return self._steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        for sampler in self.samplers.values():
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        for loader in self.loaders.values():
            dataset = getattr(loader, "dataset", None)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

    def __iter__(self):
        iters = {k: iter(loader) for k, loader in self.loaders.items()}
        for _ in range(self._steps_per_epoch):
            k = self._sample_synced_k()
            try:
                batch = next(iters[k])
            except StopIteration:
                iters[k] = iter(self.loaders[k])
                batch = next(iters[k])
            yield _append_layout_k(batch, k)

    def _sample_synced_k(self) -> int:
        if self.dist_state.is_main:
            idx = random.choices(range(len(self.k_values)), weights=self.k_probs, k=1)[0]
            tensor = torch.tensor([idx], dtype=torch.long, device=self.dist_state.device)
        else:
            tensor = torch.zeros(1, dtype=torch.long, device=self.dist_state.device)
        if dist.is_initialized():
            dist.broadcast(tensor, src=0)
        return self.k_values[int(tensor.item())]


def _append_layout_k(batch, k: int):
    if isinstance(batch, tuple):
        return batch + (torch.tensor([k], dtype=torch.long),)
    if isinstance(batch, list):
        return tuple(batch) + (torch.tensor([k], dtype=torch.long),)
    raise TypeError(f"expected dataloader batch tuple/list, got {type(batch)!r}")


def _build_fixed_k_train_dataloader(
    cfg: TrainConfig,
    dist_state: DistributedState,
    output_latent_frames: int,
) -> DataLoader:
    train_frames = _required_train_frames_for_k(cfg, output_latent_frames)
    pose_extra_frames = _pose_extra_frames_for_self_rollout(cfg, output_latent_frames)
    train_frames += _pixel_extra_frames_for_self_rollout(cfg, output_latent_frames)

    _enable_valid_starts = (
        (int(output_latent_frames) == 8 and bool(cfg.layout.k8_use_valid_starts))
        or (int(output_latent_frames) == 4 and bool(cfg.layout.k4_use_valid_starts))
    )
    _roll_layout = _enable_valid_starts
    _max_gap_latents = int(cfg.layout.max_gap_sec * cfg.sample.fps / cfg.sample.temporal_stride)
    _min_gap_latents_hc = _min_gap_steps_for_target_prefix_context(cfg, cond_end=0)
    _min_gap_latents_i2v = _min_gap_steps_for_target_prefix_context(cfg, cond_end=1)
    _valid_starts_anchor_offset = 0  # roll_layout computes the offset automatically; this is the fallback

    datasets = []
    for source_name, weight in cfg.data.sources.items():
        if weight <= 0:
            continue
        sources = SOURCE_ALIASES.get(source_name, [source_name])
        dataset = MultiSourceVideoDataset(
            video_base_dir=cfg.paths.video_base_dir,
            annotation_base_dir=cfg.paths.annotation_base_dir,
            sources=sources,
            width=cfg.sample.width,
            height=cfg.sample.height,
            target_fps=cfg.sample.fps,
            min_frames=train_frames,
            max_frames=train_frames,
            allow_short_samples=cfg.layout.variable_length,
            vae_grid_align=bool(cfg.runtime.vae_latent_cache_dir),
            random_frames=True,
            use_cache=cfg.data.use_cache,
            skip_file_check=cfg.data.skip_file_check,
            abstract_caption_prob=cfg.data.abstract_caption_prob,
            return_raw_pose=False,
            require_camera=cfg.data.require_camera,
            camera_norm_mode=cfg.data.camera_norm_mode,
            camera_post_relic_scale=cfg.data.camera_post_relic_scale,
            vae_temporal_factor=cfg.sample.temporal_stride,
            cp_size=1,
            output_latent_frames=int(output_latent_frames),
            use_valid_starts=_enable_valid_starts,
            valid_starts_anchor_offset=_valid_starts_anchor_offset,
            roll_layout=_roll_layout,
            max_gap_latents=_max_gap_latents,
            min_gap_latents_hc=_min_gap_latents_hc,
            min_gap_latents_i2v=_min_gap_latents_i2v,
            i2v_prob=float(cfg.layout.condition.i2v_prob),
            sink_remote=bool(cfg.layout.sink_remote),
            sink_remote_min_distance=int(cfg.layout.sink_remote_min_distance),
            sink_latent_frames=int(cfg.layout.sink_latent_frames),
            event_target_anchor_frame=_training_event_target_anchor_frame(cfg, int(output_latent_frames)),
            pose_extra_frames=pose_extra_frames,
        )
        _apply_min_frame_filter(dataset, train_frames + pose_extra_frames, label=f"train:{source_name}:K{output_latent_frames}")
        if cfg.data.max_samples_per_source is not None:
            dataset.samples = dataset.samples[: int(cfg.data.max_samples_per_source)]
        datasets.append((dataset, float(weight), source_name))
        if dist_state.is_main:
            print(
                f"[Data] K={output_latent_frames} {source_name}: "
                f"frames={train_frames} real={len(dataset)} weight={weight} "
                f"valid_starts={_enable_valid_starts} roll_layout={_roll_layout} "
                f"sink_remote={cfg.layout.sink_remote} max_gap_latents={_max_gap_latents} "
                f"min_gap_hc={_min_gap_latents_hc} min_gap_i2v={_min_gap_latents_i2v}",
                flush=True,
            )

    if not datasets:
        raise ValueError("no enabled data sources")

    merged = WeightedConcatDataset(datasets) if len(datasets) > 1 else datasets[0][0]
    sampler = DistributedSampler(
        merged,
        num_replicas=dist_state.world_size,
        rank=dist_state.rank,
        shuffle=True,
        drop_last=True,
    )
    return DataLoader(
        merged,
        batch_size=cfg.optimizer.batch_size,
        sampler=sampler,
        num_workers=cfg.runtime.dataloader_workers,
        pin_memory=cfg.runtime.dataloader_pin_memory,
        drop_last=True,
        timeout=600 if cfg.runtime.dataloader_workers > 0 else 0,
        persistent_workers=cfg.runtime.dataloader_workers > 0,
        prefetch_factor=cfg.runtime.dataloader_prefetch_factor if cfg.runtime.dataloader_workers > 0 else None,
    )
def build_validation_dataset(
    cfg: TrainConfig,
    mode_cfg: ValidationModeConfig,
    *,
    frames: int | None = None,
    min_frames: int | None = None,
    max_frames: int | None = None,
):
    if frames is not None:
        if min_frames is not None or max_frames is not None:
            raise ValueError("build_validation_dataset accepts either frames or min_frames/max_frames, not both")
        min_frames = frames
        max_frames = frames
    if min_frames is None and max_frames is None:
        raise ValueError("build_validation_dataset requires frames or min_frames/max_frames")
    if min_frames is None:
        min_frames = max_frames
    if max_frames is None:
        max_frames = min_frames
    _configure_dataset_env(cfg)
    source_name = mode_cfg.dataset.source
    if source_name == "custom_i2v":
        from alaya.data.custom_i2v import CustomI2VDataset

        return CustomI2VDataset(
            image_dir=str(mode_cfg.dataset.image_dir),
            pose_jsonl=str(mode_cfg.dataset.pose_jsonl),
            annotation_base_dir=mode_cfg.dataset.annotation_base_dir or cfg.paths.annotation_base_dir,
            width=mode_cfg.layout.width or cfg.sample.width,
            height=mode_cfg.layout.height or cfg.sample.height,
            frames=int(max_frames),
            pose_offset=int(getattr(mode_cfg.dataset, "pose_offset", 0) or 0),
            poses_per_image=int(getattr(mode_cfg.dataset, "poses_per_image", 1) or 1),
            pose_stride=int(getattr(mode_cfg.dataset, "pose_stride", 40) or 40),
            captions_json=getattr(mode_cfg.dataset, "captions_json", None),
        )
    if source_name == "wbench_navi":
        from alaya.data.wbench import WBenchNaviDataset

        return WBenchNaviDataset(
            root=str(mode_cfg.dataset.root),
            width=mode_cfg.layout.width or cfg.sample.width,
            height=mode_cfg.layout.height or cfg.sample.height,
            frames=int(max_frames),
            case_ids=list(mode_cfg.dataset.case_ids),
            image_dir=mode_cfg.dataset.image_dir,
            pose_case_id=mode_cfg.dataset.pose_case_id,
            pose_actions=list(mode_cfg.dataset.pose_actions),
            sekai_jsonl=mode_cfg.dataset.sekai_jsonl,
            sekai_video_base=mode_cfg.dataset.sekai_video_base,
            sekai_caption_base=(mode_cfg.dataset.annotation_base_dir or cfg.paths.annotation_base_dir),
            sekai_random_n=int(mode_cfg.dataset.sekai_random_n),
            sekai_seed=int(mode_cfg.dataset.sekai_seed),
        )
    sources = SOURCE_ALIASES.get(source_name, [source_name])
    caption_anchor_frame = _validation_caption_anchor_frame(cfg, mode_cfg)
    dataset = MultiSourceVideoDataset(
        video_base_dir=mode_cfg.dataset.video_base_dir or cfg.paths.video_base_dir,
        annotation_base_dir=mode_cfg.dataset.annotation_base_dir or cfg.paths.annotation_base_dir,
        sources=sources,
        width=mode_cfg.layout.width or cfg.sample.width,
        height=mode_cfg.layout.height or cfg.sample.height,
        target_fps=cfg.sample.fps,
        min_frames=int(min_frames),
        max_frames=int(max_frames),
        allow_short_samples=cfg.layout.variable_length,
        random_frames=False,
        prefer_max_frames=True,
        use_cache=False,
        skip_file_check=cfg.data.skip_file_check,
        abstract_caption_prob=0.0,
        return_raw_pose=True,
        require_camera=True,
        camera_norm_mode=cfg.data.camera_norm_mode,
        camera_post_relic_scale=cfg.data.camera_post_relic_scale,
        vae_temporal_factor=cfg.sample.temporal_stride,
        cp_size=1,
        caption_anchor_frame=caption_anchor_frame,
        event_target_anchor_frame=caption_anchor_frame,
    )
    _apply_validation_dataset_filter(dataset, mode_cfg.dataset.filter)
    if not cfg.layout.variable_length:  # variable_length allows short clips, so no pre-filtering
        _apply_min_frame_filter(dataset, int(min_frames), label=f"validation:{source_name}:{mode_cfg.dataset.filter or 'all'}")
    return dataset


def _configure_dataset_env(cfg: TrainConfig) -> None:
    sekai_config = MultiSourceVideoDataset.SOURCE_CONFIGS.get("sekai_game_walking")
    if cfg.data.sekai_game_jsonl:
        os.environ["LTX_SEKAI_GAME_JSONL"] = cfg.data.sekai_game_jsonl
        if sekai_config is not None:
            sekai_config["jsonl"] = cfg.data.sekai_game_jsonl
    if cfg.data.sekai_game_pose_subdir:
        os.environ["LTX_SEKAI_GAME_POSE_SUBDIR"] = cfg.data.sekai_game_pose_subdir
        if sekai_config is not None:
            sekai_config["pose_subdir"] = cfg.data.sekai_game_pose_subdir
    if cfg.data.overall_caption_prob is not None:
        os.environ["LTX_OVERALL_CAPTION_PROB"] = str(float(cfg.data.overall_caption_prob))
    if cfg.data.camera_drop_content_prob is not None:
        os.environ["LTX_CAMERA_DROP_CONTENT_PROB"] = str(float(cfg.data.camera_drop_content_prob))


def _validation_caption_anchor_frame(cfg: TrainConfig, mode_cfg: ValidationModeConfig) -> int:
    stride = int(cfg.sample.temporal_stride)
    sink_count = int(cfg.layout.sink_latent_frames)
    history_latents = (
        int(mode_cfg.layout.history_latent_frames)
        if mode_cfg.layout.history_latent_frames is not None
        else int(cfg.layout.history_latent_frames)
    )
    if _uses_vigeo_mode(cfg):
        return _vigeo_target_prefix_pixel_frames(cfg, history_latent_frames=history_latents)
    max_gap_sec = float(mode_cfg.layout.max_gap_sec or 0.0)
    gap_steps = int(max_gap_sec * cfg.sample.fps / stride)
    condition = str(mode_cfg.layout.condition)
    cond_end = 0 if condition == "hc" else int(mode_cfg.layout.condition_latent_frames)
    explicit_condition = cond_end if history_latents == 0 else 0
    target_start = sink_count + gap_steps + history_latents + explicit_condition
    return int(target_start * stride)


def _training_event_target_anchor_frame(cfg: TrainConfig, output_latent_frames: int) -> int:
    if _uses_vigeo_mode(cfg):
        return _vigeo_target_prefix_pixel_frames(cfg)
    stride = int(cfg.sample.temporal_stride)
    sink_count = int(cfg.layout.sink_latent_frames)
    history_latents = int(cfg.layout.history_latent_frames)
    explicit_condition = 0
    if history_latents == 0:
        if cfg.layout.condition.i2v_prob > 0:
            explicit_condition = max(explicit_condition, 1)
        if cfg.layout.condition.v2v_prob > 0:
            explicit_condition = max(
                explicit_condition,
                max(1, int(int(output_latent_frames) * cfg.layout.condition.v2v_ratio_max)),
            )
    # Use the earliest possible target position. Random training gap, when enabled,
    # shifts the actual target later into the event instead of before it.
    target_start = sink_count + history_latents + explicit_condition
    return int(target_start * stride)


def _apply_validation_dataset_filter(dataset: MultiSourceVideoDataset, dataset_filter: str | None) -> None:
    if not dataset_filter:
        return
    wanted = str(dataset_filter).strip()
    if not wanted:
        return
    wanted_lower = wanted.lower()
    kept = []
    for sample in dataset.samples:
        video_path, _caption_path, _pose_path, source_name, video_id = sample
        scene_name = ""
        candidates = {
            str(source_name),
            str(video_id),
            scene_name,
        }
        if any(c.lower() == wanted_lower for c in candidates if c):
            kept.append(sample)
    before = len(dataset.samples)
    dataset.samples = kept
    print(
        f"[ValidationData] filter={wanted} kept={len(dataset.samples)}/{before}",
        flush=True,
    )


def _apply_min_frame_filter(dataset: MultiSourceVideoDataset, min_frames: int | None, *, label: str) -> None:
    if min_frames is None or int(min_frames) <= 0:
        return
    frame_counts = getattr(dataset, "_sample_target_frame_counts", {}) or {}
    if not frame_counts:
        return
    before = len(dataset.samples)
    kept = []
    for sample in dataset.samples:
        frame_count = frame_counts.get(sample[0])
        if frame_count is None or int(frame_count) >= int(min_frames):
            kept.append(sample)
    dataset.samples = kept
    dropped = before - len(kept)
    if dropped > 0:
        print(
            f"[Data] min_frame_filter {label}: kept={len(kept)}/{before} "
            f"dropped={dropped} min_frames={int(min_frames)}",
            flush=True,
        )


def _pose_extra_frames_for_self_rollout(cfg: TrainConfig, output_latent_frames: int) -> int:
    """Extra pose frames (at target fps) needed by a long training rollout.

    The rollout reaches chunk r=max_chunks, so action indices extend to
    target_start + (max_chunks+1)*K - 1 latents while the existing pixel window covers target_start+K;
    the extra latents are max_chunks*K, i.e. max_chunks*K*temporal_stride pixel frames.
    Returns 0 when self_rollout is disabled.
    Returns 0 when score_gt_context is enabled, because the pixel window is already extended by the
    same amount and poses are loaded together with it.
    """
    dmd = getattr(cfg, "dmd", None)
    if dmd is None or not dmd.enabled or not dmd.self_rollout.enabled:
        return 0
    if bool(getattr(dmd.self_rollout, "score_gt_context", False)):
        return 0
    return int(dmd.self_rollout.max_chunks) * int(output_latent_frames) * int(cfg.sample.temporal_stride)


def _pixel_extra_frames_for_self_rollout(cfg: TrainConfig, output_latent_frames: int) -> int:
    """Extra pixel frames needed to cover the rollout horizon when score_gt_context is enabled.

    Scoring conditions are sliced from the ground-truth history/nearby by window position, so
    latent_full must reach target_start + max_chunks*K + K, i.e. max_chunks*K*temporal_stride more frames.
    Poses then cover the same horizon automatically (the pose extra returns 0, see above).
    Cost: video decode and VAE encode per step grow accordingly.
    Returns 0 when score_gt_context is disabled. The value is in pixel frames and adds to the
    required-frames computation; the matching latent coverage lives in the trainer.
    """
    dmd = getattr(cfg, "dmd", None)
    if dmd is None or not dmd.enabled or not dmd.self_rollout.enabled:
        return 0
    if not bool(getattr(dmd.self_rollout, "score_gt_context", False)):
        return 0
    return int(dmd.self_rollout.max_chunks) * int(output_latent_frames) * int(cfg.sample.temporal_stride)


def _required_train_frames_for_k(cfg: TrainConfig, output_latent_frames: int) -> int:
    if _uses_vigeo_mode(cfg):
        target_latents = int(output_latent_frames) * (2 if cfg.next_forcing.enabled else 1)
        target_pixels = target_latents * int(cfg.sample.temporal_stride)
        required_pixels = _vigeo_target_prefix_pixel_frames(cfg) + target_pixels
        stride = int(cfg.sample.temporal_stride)
        return 1 + ((required_pixels - 1 + stride - 1) // stride) * stride
    max_gap = int(cfg.layout.max_gap_sec * cfg.sample.fps / cfg.sample.temporal_stride)
    next_extra = int(output_latent_frames) if cfg.next_forcing.enabled else 0
    max_condition = 0
    if cfg.layout.history_latent_frames == 0 and cfg.layout.condition.type != "inline":
        max_condition = 1
        if cfg.layout.condition.v2v_prob > 0:
            max_condition = max(max_condition, max(1, int(output_latent_frames * cfg.layout.condition.v2v_ratio_max)))
    required_latents = (
        cfg.layout.sink_latent_frames
        + max_gap
        + cfg.layout.history_latent_frames
        + max_condition
        + int(output_latent_frames)
        + next_extra
    )
    return (required_latents - 1) * cfg.sample.temporal_stride + 1


def _vigeo_prefix_pixel_frames(cfg: TrainConfig) -> int:
    configured = cfg.spatial_memory.vigeo_prefix_frames
    return int(cfg.spatial_memory.num_context_frames if configured is None else configured)


def _vigeo_target_prefix_pixel_frames(
    cfg: TrainConfig,
    *,
    history_latent_frames: int | None = None,
) -> int:
    history_latents = (
        int(cfg.layout.history_latent_frames)
        if history_latent_frames is None
        else int(history_latent_frames)
    )
    if history_latents <= 0:
        return _vigeo_prefix_pixel_frames(cfg)
    return 1 + (history_latents - 1) * int(cfg.sample.temporal_stride)


def _uses_vigeo_mode(cfg: TrainConfig) -> bool:
    if bool(cfg.dmd.enabled):
        return False
    return (
        bool(cfg.spatial_memory.enabled)
        and str(cfg.spatial_memory.context_mode) == "vigeo_prefix_last_frame"
        and str(cfg.spatial_memory.depth_backend) == "vigeo"
    )


def _min_gap_steps_for_target_prefix_context(cfg: TrainConfig, *, cond_end: int) -> int:
    spatial = cfg.spatial_memory
    if (
        not bool(spatial.enabled)
        or str(getattr(spatial, "context_mode", "retrieval")) != "target_prefix_pixels"
        or not bool(getattr(spatial, "require_full_context", True))
    ):
        return 0

    stride = max(1, int(cfg.sample.temporal_stride))
    history_pixels = max(1, int(spatial.num_context_frames))
    sink_count = max(0, int(cfg.layout.sink_latent_frames))
    history_latents = max(0, int(cfg.layout.history_latent_frames))
    explicit_condition = max(0, int(cond_end)) if history_latents == 0 else 0

    source_floor = 0
    if not bool(spatial.include_sink):
        source_floor = sink_count * stride
    min_target_pixel_start = source_floor + history_pixels
    min_target_start = (min_target_pixel_start + stride - 1) // stride
    target_start_without_gap = sink_count + history_latents + explicit_condition
    return max(0, int(min_target_start) - int(target_start_without_gap))
