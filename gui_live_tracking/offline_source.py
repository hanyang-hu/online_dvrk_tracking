from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.frame_source import TrackingSample


def load_joint_yaml(joint_angles_path: Path) -> np.ndarray:
    try:
        with open(joint_angles_path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as exc:
        raise ValueError(f"Failed to load joint-angle YAML {joint_angles_path}: {exc}") from exc

    if not isinstance(data, dict) or not data:
        raise ValueError(f"Joint-angle YAML must be a nonempty mapping: {joint_angles_path}")

    entries: List[np.ndarray] = []
    for i in range(len(data)):
        key = str(i)
        if key not in data:
            raise ValueError(f"Joint-angle YAML missing frame key {key!r}: {joint_angles_path}")
        arr = np.asarray(data[key], dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"Joint-angle entry {key!r} must be a nonempty 1D array.")
        entries.append(arr)

    try:
        return np.stack(entries, axis=0)
    except ValueError as exc:
        raise ValueError("All joint-angle YAML entries must have the same length.") from exc


def valid_video_fps(fps: float) -> float:
    if not np.isfinite(fps) or fps <= 1e-6:
        return 30.0
    return float(fps)


class OfflineFrameSource:
    def __init__(self, config: LiveTrackingConfig):
        self.video_path = Path(config.video_path)
        self.joint_angles_path = Path(config.joint_angles_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._joint_angles: Optional[np.ndarray] = None
        self._fps = 30.0
        self._source_index = 0
        self._max_samples = 0

    def start(self) -> None:
        self.stop()
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise ValueError(f"Could not open video: {self.video_path}")

        self._joint_angles = load_joint_yaml(self.joint_angles_path)
        self._fps = valid_video_fps(float(self._cap.get(cv2.CAP_PROP_FPS)))
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            frame_count = len(self._joint_angles)
        self._max_samples = min(frame_count, len(self._joint_angles))
        if self._max_samples <= 0:
            raise ValueError("Video and joint-angle YAML contain no paired samples.")
        self._source_index = 0

    def get_sample(self, timeout_sec: float = 0.5) -> Optional[TrackingSample]:
        del timeout_sec
        if self._cap is None or self._joint_angles is None:
            raise RuntimeError("OfflineFrameSource.start() must be called before get_sample().")
        if self._source_index >= self._max_samples:
            return None

        ok, frame = self._cap.read()
        if not ok:
            return None

        idx = self._source_index
        self._source_index += 1
        timestamp_ns = int(round(idx * 1_000_000_000 / self._fps))
        return TrackingSample(
            frame_bgr=frame.copy(),
            raw_joint_angles=self._joint_angles[idx].copy(),
            timestamp_ns=timestamp_ns,
            source_index=idx,
        )

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
