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


class FrameSource(Protocol):
    def start(self) -> None:
        ...

    def get_sample(self, timeout_sec: float = 0.5) -> Optional[TrackingSample]:
        ...

    def stop(self) -> None:
        ...
