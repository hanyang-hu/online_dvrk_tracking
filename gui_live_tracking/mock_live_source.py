from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.frame_source import TrackingSample
from gui_live_tracking.offline_source import load_joint_yaml, valid_video_fps


class MockLiveFrameSource:
    def __init__(self, config: LiveTrackingConfig):
        self.video_path = Path(config.video_path)
        self.joint_angles_path = Path(config.joint_angles_path)
        self.rate_hz = float(config.mock_rate_hz)
        self.loop = bool(config.mock_loop)
        self._frames: List[np.ndarray] = []
        self._joint_angles: Optional[np.ndarray] = None
        self._sample_count = 0
        self._start_monotonic: Optional[float] = None
        self._last_source_index = -1
        self._fps = 30.0

    def start(self) -> None:
        self.stop()
        if self.rate_hz <= 0:
            raise ValueError("mock_rate_hz must be positive.")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        self._fps = valid_video_fps(float(cap.get(cv2.CAP_PROP_FPS)))

        try:
            frames: List[np.ndarray] = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame.copy())
        finally:
            cap.release()

        self._joint_angles = load_joint_yaml(self.joint_angles_path)
        self._sample_count = min(len(frames), len(self._joint_angles))
        if self._sample_count <= 0:
            raise ValueError("Video and joint-angle YAML contain no paired samples.")
        self._frames = frames[: self._sample_count]
        self._start_monotonic = time.monotonic() + (1.0 / self.rate_hz)
        self._last_source_index = -1

    def get_sample(self, timeout_sec: float = 0.5) -> Optional[TrackingSample]:
        if self._start_monotonic is None or self._joint_angles is None:
            raise RuntimeError("MockLiveFrameSource.start() must be called before get_sample().")

        period = 1.0 / self.rate_hz
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while True:
            now = time.monotonic()
            elapsed = max(0.0, now - self._start_monotonic)
            target_index = int(elapsed / period)

            if not self.loop and target_index >= self._sample_count:
                if self._last_source_index < self._sample_count - 1:
                    target_index = self._sample_count - 1
                else:
                    return None

            if target_index > self._last_source_index:
                break

            wait = min(period / 4.0, max(0.0, deadline - now))
            if wait <= 0:
                return None
            time.sleep(wait)

        source_index = target_index
        frame_index = source_index % self._sample_count
        self._last_source_index = source_index
        timestamp_ns = int(round(source_index * 1_000_000_000 / self.rate_hz))
        return TrackingSample(
            frame_bgr=self._frames[frame_index].copy(),
            raw_joint_angles=self._joint_angles[frame_index].copy(),
            timestamp_ns=timestamp_ns,
            source_index=source_index,
        )

    def stop(self) -> None:
        self._frames = []
        self._joint_angles = None
        self._sample_count = 0
        self._start_monotonic = None
