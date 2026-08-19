"""Build AlayaWorld camera trajectories from frontend motion values."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    """Define the camera rates represented by full-scale frontend input."""

    fps: float
    strafe_units_per_second: float
    vertical_units_per_second: float
    forward_units_per_second: float
    pitch_degrees_per_second: float
    yaw_degrees_per_second: float
    roll_degrees_per_second: float


class CameraMotionPlanner:
    """Integrate six-axis controls into camera-to-world poses."""

    def __init__(self, initial_c2w: np.ndarray, config: MotionConfig) -> None:
        self._initial_c2w = _validate_pose(initial_c2w).copy()
        self._current_c2w = self._initial_c2w.copy()
        self._config = config

    @property
    def current_c2w(self) -> np.ndarray:
        """Return the current camera-to-world pose."""
        return self._current_c2w.copy()

    def reset(self) -> None:
        """Restore the initial camera pose."""
        self._current_c2w = self._initial_c2w.copy()

    def plan(
        self,
        *,
        strafe: float,
        vertical: float,
        forward: float,
        pitch: float,
        yaw: float,
        roll: float,
        frame_count: int,
    ) -> np.ndarray:
        """Return the next pixel-rate camera poses and advance the camera."""
        poses = plan_camera_motion(
            self._current_c2w,
            strafe=strafe,
            vertical=vertical,
            forward=forward,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            frame_count=frame_count,
            config=self._config,
        )
        self._current_c2w = poses[-1].copy()
        return poses


def plan_camera_motion(
    start_c2w: np.ndarray,
    *,
    strafe: float,
    vertical: float,
    forward: float,
    pitch: float,
    yaw: float,
    roll: float,
    frame_count: int,
    config: MotionConfig,
) -> np.ndarray:
    """Integrate normalized six-axis motion at the model's pixel rate.

    AlayaWorld consumes camera-to-world matrices in a local camera convention
    where positive X moves right, positive Y moves up, and positive Z moves
    forward. Euler rotations around X, Y, and Z represent pitch, yaw, and roll.
    The frontend owns keyboard, pointer, touch, or joystick mapping.

    Args:
        start_c2w: Camera-to-world transform immediately before the new poses.
        strafe: Normalized left-to-right velocity in ``[-1, 1]``.
        vertical: Normalized down-to-up velocity in ``[-1, 1]``.
        forward: Normalized forward velocity in ``[-1, 1]``.
        pitch: Normalized down-to-up pitch velocity in ``[-1, 1]``.
        yaw: Normalized yaw velocity in ``[-1, 1]``.
        roll: Normalized counterclockwise-to-clockwise roll velocity in ``[-1, 1]``.
        frame_count: Number of pixel-rate poses to generate.
        config: Camera cadence and full-scale motion rates.

    Returns:
        Contiguous float32 poses with shape ``(frame_count, 4, 4)``.

    Raises:
        ValueError: If the pose, controls, count, or rates are invalid.
    """
    pose = _validate_pose(start_c2w).copy()
    controls = {
        "strafe": strafe,
        "vertical": vertical,
        "forward": forward,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }
    for name, value in controls.items():
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between -1 and 1")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if config.fps <= 0:
        raise ValueError("motion fps must be positive")
    rates = {
        "strafe_units_per_second": config.strafe_units_per_second,
        "vertical_units_per_second": config.vertical_units_per_second,
        "forward_units_per_second": config.forward_units_per_second,
        "pitch_degrees_per_second": config.pitch_degrees_per_second,
        "yaw_degrees_per_second": config.yaw_degrees_per_second,
        "roll_degrees_per_second": config.roll_degrees_per_second,
    }
    for name, value in rates.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    translation = _normalized_vector(strafe, vertical, forward)
    translation *= (
        np.asarray(
            [
                config.strafe_units_per_second,
                config.vertical_units_per_second,
                config.forward_units_per_second,
            ],
            dtype=np.float64,
        )
        / config.fps
    )
    rotation = _normalized_vector(pitch, yaw, roll)
    pitch_radians, yaw_radians, roll_radians = np.radians(
        rotation
        * np.asarray(
            [
                config.pitch_degrees_per_second,
                config.yaw_degrees_per_second,
                config.roll_degrees_per_second,
            ],
            dtype=np.float64,
        )
        / config.fps
    )
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = (
        _rotation_z(float(roll_radians))
        @ _rotation_y(float(yaw_radians))
        @ _rotation_x(float(pitch_radians))
    )
    delta[:3, 3] = translation

    poses = np.empty((frame_count, 4, 4), dtype=np.float32)
    for index in range(frame_count):
        pose = pose @ delta
        poses[index] = pose
    return np.ascontiguousarray(poses)


def _normalized_vector(x: float, y: float, z: float) -> np.ndarray:
    """Return an axis vector whose magnitude does not exceed one."""
    vector = np.asarray([x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 1.0:
        vector /= norm
    return vector


def _rotation_x(angle: float) -> np.ndarray:
    """Return a right-handed rotation around the local camera X axis."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    """Return a right-handed rotation around the local camera Y axis."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    """Return a right-handed rotation around the local camera Z axis."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _validate_pose(value: np.ndarray) -> np.ndarray:
    """Return a validated camera-to-world transform as float64."""
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"camera pose must have shape (4, 4), got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError("camera pose must contain only finite values")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("camera pose must have a homogeneous final row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
        raise ValueError("camera rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-4):
        raise ValueError("camera rotation determinant must be one")
    return pose
