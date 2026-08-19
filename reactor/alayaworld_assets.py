"""Prepare model assets and scene metadata for the Reactor adapter.

The model code is this repository, so nothing is cloned for it. What is
prepared here are the weights and the two pinned code dependencies that live
outside it, all under the directory the CLI mounts.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Never, cast

import yaml

from reactor_runtime import get_weights_path
from reactor_runtime.log import get_logger

if TYPE_CHECKING:
    from examples.alayaworld.alayaworld_types import AlayaWorldConfig, Asset
else:
    module_prefix = f"{__package__}." if __package__ else ""
    types_module = importlib.import_module(f"{module_prefix}alayaworld_types")
    AlayaWorldConfig = types_module.AlayaWorldConfig
    Asset = types_module.Asset

logger = get_logger(__name__)

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_COMPILE_MODES = ["none", "default", "reduce-overhead", "max-autotune"]
_ATTENTION_BACKENDS = ["flash_attention_4", "pytorch", "upstream"]
_SCENE_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_SCENE_MEMBER_SUFFIXES = (
    "_video.mp4",
    "_camera.pt",
    "_prompt.txt",
    *tuple(f"_image{extension}" for extension in _SCENE_IMAGE_EXTENSIONS),
)
_SNAPSHOT_REVISION_MARKER = ".reactor-revision"


def read_config(config_path: Path | None) -> AlayaWorldConfig:
    """Read and validate the AlayaWorld adapter YAML."""
    if config_path is None:
        raise ValueError("AlayaWorld requires runtime.config in reactor.yaml")
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    assets = _mapping(document.get("assets"), "assets")
    inference = _mapping(document.get("inference"), "inference")
    inputs = _mapping(document.get("inputs"), "inputs")
    motion = _mapping(document.get("motion"), "motion")
    decode = _mapping(document.get("decode"), "decode")
    memory = _mapping(document.get("memory"), "memory")
    stream = _mapping(document.get("stream", {}), "stream")
    da3_source = _mapping(assets.get("da3_source"), "assets.da3_source")

    compile_mode = str(inference.get("compile", "reduce-overhead"))
    if compile_mode not in _COMPILE_MODES:
        raise ValueError(f"inference.compile must be one of {', '.join(_COMPILE_MODES)}")
    attention_backend = str(inference.get("attention_backend", "flash_attention_4"))
    if attention_backend not in _ATTENTION_BACKENDS:
        raise ValueError(
            f"inference.attention_backend must be one of {', '.join(_ATTENTION_BACKENDS)}"
        )
    warmup_chunks = int(inference.get("warmup_chunks", 1))
    if warmup_chunks < 0:
        raise ValueError("inference.warmup_chunks must be zero or more")

    overlap = int(decode.get("overlap_latents", 6))
    if overlap <= 0:
        raise ValueError("decode.overlap_latents must be positive")
    max_spatial_frames = int(memory.get("max_spatial_frames", 320))
    recent_spatial_frames = int(memory.get("recent_spatial_frames", 160))
    if max_spatial_frames < 10:
        raise ValueError("memory.max_spatial_frames must be at least 10")
    if not 1 <= recent_spatial_frames <= max_spatial_frames:
        raise ValueError("memory.recent_spatial_frames must be between 1 and max_spatial_frames")
    max_chunks_per_rollout = int(stream.get("max_chunks_per_rollout", 512))
    if max_chunks_per_rollout < 1:
        raise ValueError("stream.max_chunks_per_rollout must be positive")

    motion_rates = {
        "strafe_units_per_second": float(motion.get("strafe_units_per_second", 0.126)),
        "vertical_units_per_second": float(motion.get("vertical_units_per_second", 0.261)),
        "forward_units_per_second": float(motion.get("forward_units_per_second", 1.905)),
        "pitch_degrees_per_second": float(motion.get("pitch_degrees_per_second", 4.039)),
        "yaw_degrees_per_second": float(motion.get("yaw_degrees_per_second", 9.375)),
        "roll_degrees_per_second": float(motion.get("roll_degrees_per_second", 4.094)),
    }
    for name, value in motion_rates.items():
        if value <= 0:
            raise ValueError(f"motion.{name} must be positive")

    # Weights live in the mounted cache; code, configs, and playground cases
    # live in the repository the adapter ships inside.
    weights_root = get_weights_path()
    repo_root = Path(__file__).resolve().parent.parent
    taehv_raw = assets.get("taehv")
    taehv_path = _path(weights_root, taehv_raw) if taehv_raw else None
    taehv_source_raw = assets.get("taehv_source")
    taehv_source = (
        _mapping(taehv_source_raw, "assets.taehv_source") if taehv_source_raw else None
    )
    random_inputs_raw = inputs.get("random_images")
    if not isinstance(random_inputs_raw, list) or not random_inputs_raw:
        raise ValueError("inputs.random_images must be a non-empty YAML list")
    return AlayaWorldConfig(
        repo_root=repo_root,
        inference_config=_path(repo_root, inference["config"]),
        upload_template=_path(repo_root, inputs["upload_template"]),
        random_inputs=tuple(_path(repo_root, value) for value in random_inputs_raw),
        model=_asset(weights_root, assets.get("model"), "assets.model"),
        gemma=_asset(weights_root, assets.get("gemma"), "assets.gemma"),
        da3_source_path=_path(weights_root, da3_source["path"]),
        da3_source_url=_repository_url(da3_source.get("url"), "assets.da3_source.url"),
        da3_source_revision=_revision(da3_source.get("revision"), "assets.da3_source.revision"),
        da3_model=_asset(weights_root, assets.get("da3_model"), "assets.da3_model"),
        da3_cache=_path(weights_root, assets["da3_cache"]),
        seed=int(inference.get("seed", 1234)),
        compile_mode=compile_mode,
        warmup_chunks=warmup_chunks,
        attention_backend=attention_backend,
        flex_attention=bool(inference.get("flex_attention", True)),
        ttc=bool(inference.get("ttc", False)),
        bank_taehv=bool(inference.get("bank_taehv", False)),
        taehv_path=taehv_path,
        taehv_source_path=(
            None if taehv_source is None else _path(weights_root, taehv_source["path"])
        ),
        taehv_source_url=(
            None
            if taehv_source is None
            else _repository_url(taehv_source.get("url"), "assets.taehv_source.url")
        ),
        taehv_source_revision=(
            None
            if taehv_source is None
            else _revision(taehv_source.get("revision"), "assets.taehv_source.revision")
        ),
        decode_overlap_latents=overlap,
        max_spatial_frames=max_spatial_frames,
        recent_spatial_frames=recent_spatial_frames,
        max_chunks_per_rollout=max_chunks_per_rollout,
        strafe_units_per_second=motion_rates["strafe_units_per_second"],
        vertical_units_per_second=motion_rates["vertical_units_per_second"],
        forward_units_per_second=motion_rates["forward_units_per_second"],
        pitch_degrees_per_second=motion_rates["pitch_degrees_per_second"],
        yaw_degrees_per_second=motion_rates["yaw_degrees_per_second"],
        roll_degrees_per_second=motion_rates["roll_degrees_per_second"],
    )


def prepare_runtime_assets(config: AlayaWorldConfig) -> None:
    """Download each missing model asset and pinned code dependency."""
    _ensure_hf_file(config.model, name="AlayaWorld merged checkpoint")
    _ensure_hf_snapshot(config.gemma, name="Gemma text encoder")
    _ensure_git_checkout(
        config.da3_source_path,
        url=config.da3_source_url,
        revision=config.da3_source_revision,
        name="Depth-Anything-3 source",
    )
    _ensure_hf_snapshot(
        config.da3_model,
        name="Depth-Anything-3 checkpoint",
        cache_dir=config.da3_cache / "hub",
    )
    if (
        config.taehv_source_path is not None
        and config.taehv_source_url is not None
        and config.taehv_source_revision is not None
    ):
        _ensure_git_checkout(
            config.taehv_source_path,
            url=config.taehv_source_url,
            revision=config.taehv_source_revision,
            name="TAEHV tiny decoder source",
        )


def validate_runtime_paths(config: AlayaWorldConfig) -> None:
    """Require every prepared source, model asset, and scene input."""
    required = {
        "AlayaWorld inference config": config.inference_config,
        "AlayaWorld merged checkpoint": config.model.path,
        "Gemma text encoder": config.gemma.path,
        "Depth-Anything-3 source": config.da3_source_path,
        "Depth-Anything-3 checkpoint": config.da3_model.path,
        "DA3 checkpoint cache": config.da3_cache,
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    _validate_scene_triplet(config.upload_template, "upload template")
    for index, path in enumerate(config.random_inputs):
        _validate_scene_triplet(path, f"random image {index}")
    if config.bank_taehv and (config.taehv_path is None or not config.taehv_path.is_file()):
        raise FileNotFoundError("inference.bank_taehv requires assets.taehv")


def scene_prompt_path(path: Path) -> Path:
    """Return the prompt member for a configured scene."""
    return Path(f"{_scene_prefix(path)}_prompt.txt")


def scene_image_path(path: Path) -> Path:
    """Return the first supported still-image member for a configured scene."""
    prefix = _scene_prefix(path)
    for extension in _SCENE_IMAGE_EXTENSIONS:
        candidate = Path(f"{prefix}_image{extension}")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"scene image not found for prefix: {prefix}")


def load_scene_metadata(path: Path, torch_module: Any) -> dict[str, Any]:
    """Load a fresh metadata mapping from the upload camera template."""
    value = torch_module.load(
        _scene_camera_path(path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(value, dict) or "cam_c2w" not in value:
        raise ValueError("upload camera template must contain cam_c2w metadata")
    return cast(dict[str, Any], value)


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Return a YAML mapping or raise a precise configuration error."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return cast(dict[str, Any], value)


def _asset(base_path: Path, value: object, name: str) -> Asset:
    """Read one local path and immutable public repository identity."""
    document = _mapping(value, name)
    repo_id = str(document.get("repo_id", ""))
    if "/" not in repo_id:
        raise ValueError(f"{name}.repo_id must identify a public repository")
    return Asset(
        path=_path(base_path, document["path"]),
        repo_id=repo_id,
        revision=_revision(document.get("revision"), f"{name}.revision"),
    )


def _path(base_path: Path, value: object) -> Path:
    """Resolve a configured path relative to its owning directory."""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_path / path
    return path.resolve()


def _revision(value: object, name: str) -> str:
    """Return an immutable 40-character revision."""
    revision = str(value or "")
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"{name} must be a 40-character commit revision")
    return revision


def _repository_url(value: object, name: str) -> str:
    """Return the HTTPS URL for a public source repository."""
    url = str(value or "")
    if not url.startswith("https://"):
        raise ValueError(f"{name} must be a public HTTPS URL")
    return url


def _verify_repository_revision(path: Path, expected: str, name: str) -> None:
    """Require a local Git checkout to match its configured revision."""
    actual = _run_git(
        ["-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        name,
    ).stdout.strip()
    if actual != expected:
        raise RuntimeError(f"{name} revision is {actual}; expected {expected}")


def _ensure_git_checkout(path: Path, *, url: str, revision: str, name: str) -> None:
    """Clone a missing public repository and require its pinned revision."""
    if path.exists():
        _verify_repository_revision(path, revision, name)
        return
    logger.info("downloading source checkout", asset=name, url=url, revision=revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reactor-download-", dir=path.parent) as temporary:
        checkout = Path(temporary) / "checkout"
        _run_git(["clone", "--filter=blob:none", "--no-checkout", url, str(checkout)], name)
        _run_git(["-C", str(checkout), "checkout", "--detach", revision], name)
        with suppress(FileExistsError):
            checkout.rename(path)
    _verify_repository_revision(path, revision, name)


def _run_git(arguments: list[str], name: str) -> subprocess.CompletedProcess[str]:
    """Run Git and report an actionable public-resource error."""
    try:
        return subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Git is required to download {name}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "Git command failed"
        raise RuntimeError(f"Unable to prepare {name}: {detail}") from error


def _ensure_hf_file(asset: Asset, *, name: str) -> None:
    """Download one file at its pinned revision and record completed preparation."""
    marker = _file_revision_marker(asset.path)
    if _is_nonempty_file(asset.path) and _marker_matches(marker, asset.revision):
        return
    logger.info(
        "downloading model file",
        asset=name,
        repo_id=asset.repo_id,
        revision=asset.revision,
    )
    asset.path.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = _hf_hub_download(
            repo_id=asset.repo_id,
            filename=asset.path.name,
            revision=asset.revision,
            local_dir=asset.path.parent,
        )
    except Exception as error:
        _raise_hf_download_error(name, asset.repo_id, error)
    if downloaded.resolve() != asset.path.resolve() or not _is_nonempty_file(asset.path):
        raise RuntimeError(f"{name} download did not create {asset.path}")
    _write_marker(marker, asset.revision)


def _ensure_hf_snapshot(
    asset: Asset,
    *,
    name: str,
    cache_dir: Path | None = None,
) -> None:
    """Download one repository snapshot and resume incomplete preparation."""
    marker = asset.path / _SNAPSHOT_REVISION_MARKER
    if _snapshot_has_content(asset.path) and _marker_matches(marker, asset.revision):
        return
    logger.info(
        "downloading model snapshot",
        asset=name,
        repo_id=asset.repo_id,
        revision=asset.revision,
    )
    try:
        if cache_dir is None:
            asset.path.mkdir(parents=True, exist_ok=True)
            downloaded = _hf_snapshot_download(
                repo_id=asset.repo_id,
                revision=asset.revision,
                local_dir=asset.path,
            )
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            downloaded = _hf_snapshot_download(
                repo_id=asset.repo_id,
                revision=asset.revision,
                cache_dir=cache_dir,
            )
    except Exception as error:
        _raise_hf_download_error(name, asset.repo_id, error)
    if not downloaded.exists() or not _snapshot_has_content(asset.path):
        raise RuntimeError(f"{name} download did not create {asset.path}")
    _write_marker(marker, asset.revision)


def _file_revision_marker(path: Path) -> Path:
    """Return the revision marker dedicated to one downloaded file."""
    return path.with_name(f".{path.name}.reactor-revision")


def _is_nonempty_file(path: Path) -> bool:
    """Return whether a path names a file containing data."""
    return path.is_file() and path.stat().st_size > 0


def _snapshot_has_content(path: Path) -> bool:
    """Return whether a snapshot directory contains data beyond its marker."""
    return path.is_dir() and any(
        child.name != _SNAPSHOT_REVISION_MARKER for child in path.iterdir()
    )


def _marker_matches(path: Path, revision: str) -> bool:
    """Return whether a completed asset marker matches the pinned revision."""
    try:
        return path.read_text(encoding="utf-8").strip() == revision
    except OSError:
        return False


def _write_marker(path: Path, revision: str) -> None:
    """Record the revision of a fully prepared asset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{revision}\n", encoding="utf-8")


def _hf_hub_download(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    local_dir: Path,
) -> Path:
    """Download one Hugging Face file without importing model dependencies eagerly."""
    hugging_face = importlib.import_module("huggingface_hub")
    return Path(
        str(
            hugging_face.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=local_dir,
            )
        )
    )


def _hf_snapshot_download(
    *,
    repo_id: str,
    revision: str,
    local_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download one pinned Hugging Face snapshot lazily."""
    hugging_face = importlib.import_module("huggingface_hub")
    return Path(
        str(
            hugging_face.snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=local_dir,
                cache_dir=cache_dir,
            )
        )
    )


def _raise_hf_download_error(name: str, repo_id: str, error: Exception) -> Never:
    """Raise an authentication-aware Hugging Face download error."""
    raise RuntimeError(
        f"Unable to download {name} from {repo_id}. Check network access and run "
        "`hf auth login` if the repository is gated."
    ) from error


def _scene_prefix(path: Path) -> Path:
    """Return a scene prefix from a prefix or one of its triplet members."""
    value = str(path)
    for suffix in _SCENE_MEMBER_SUFFIXES:
        if value.endswith(suffix):
            return Path(value[: -len(suffix)])
    return path


def _scene_camera_path(path: Path) -> Path:
    """Return the camera metadata member for a configured scene."""
    return Path(f"{_scene_prefix(path)}_camera.pt")


def _validate_scene_triplet(path: Path, name: str) -> None:
    """Require the image, camera, and prompt files used by one scene."""
    members = {
        "image": scene_image_path(path),
        "camera": _scene_camera_path(path),
        "prompt": scene_prompt_path(path),
    }
    for member_name, member_path in members.items():
        if not member_path.is_file():
            raise FileNotFoundError(f"{name} {member_name} not found: {member_path}")
