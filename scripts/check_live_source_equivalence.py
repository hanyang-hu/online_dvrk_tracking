#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui_live_tracking.config import LiveTrackingConfig  # noqa: E402
from gui_live_tracking.mock_live_source import MockLiveFrameSource  # noqa: E402
from gui_live_tracking.offline_source import OfflineFrameSource  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare offline and mock-live source samples by source index.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--joint-angles", required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--rate", type=float, default=1000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = LiveTrackingConfig(
        video_path=Path(args.video),
        joint_angles_path=Path(args.joint_angles),
        mock_rate_hz=args.rate,
    )
    offline = OfflineFrameSource(cfg)
    mock = MockLiveFrameSource(cfg)
    try:
        offline.start()
        mock.start()
        for _ in range(args.count):
            offline_sample = offline.get_sample()
            mock_sample = mock.get_sample(timeout_sec=1.0)
            if offline_sample is None or mock_sample is None:
                break
            if offline_sample.source_index != mock_sample.source_index:
                print(f"FAIL: index mismatch {offline_sample.source_index} != {mock_sample.source_index}")
                return 1
            if not np.array_equal(offline_sample.frame_bgr, mock_sample.frame_bgr):
                print(f"FAIL: frame mismatch at index {offline_sample.source_index}")
                return 1
            if not np.array_equal(offline_sample.raw_joint_angles, mock_sample.raw_joint_angles):
                print(f"FAIL: joint mismatch at index {offline_sample.source_index}")
                return 1
        print("PASS: offline and mock-live samples match.")
        return 0
    finally:
        offline.stop()
        mock.stop()


if __name__ == "__main__":
    raise SystemExit(main())
