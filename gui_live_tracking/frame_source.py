from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class TrackingSample:
    frame_bgr: np.ndarray
    raw_joint_angles: np.ndarray
    timestamp_ns: int
    source_index: int


def tracking_sample_invalid_reason(sample: TrackingSample) -> Optional[str]:
    frame = sample.frame_bgr
    if frame is None:
        return "image is None"
    if not isinstance(frame, np.ndarray):
        return f"image is {type(frame).__name__}, expected ndarray"
    if frame.size == 0:
        return "image is empty"
    if frame.ndim < 2:
        return f"image has invalid shape {frame.shape}"
    if np.issubdtype(frame.dtype, np.floating):
        try:
            if not np.all(np.isfinite(frame)):
                return "image contains NaN or Inf"
        except TypeError:
            return f"image dtype {frame.dtype} cannot be checked for finite values"

    raw_joint_angles = sample.raw_joint_angles
    if raw_joint_angles is None:
        return "joint angles are None"
    try:
        joint_angles = np.asarray(raw_joint_angles, dtype=np.float64)
    except (TypeError, ValueError):
        return "joint angles cannot be converted to floats"
    if joint_angles.ndim != 1 or joint_angles.size == 0:
        return f"joint angles have invalid shape {joint_angles.shape}"
    if not np.all(np.isfinite(joint_angles)):
        return "joint angles contain NaN or Inf"

    return None


class FrameSource(Protocol):
    def start(self) -> None:
        ...

    def get_sample(self, timeout_sec: float = 0.5) -> Optional[TrackingSample]:
        ...

    def stop(self) -> None:
        ...
