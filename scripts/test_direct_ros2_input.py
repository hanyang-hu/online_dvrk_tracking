#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui_live_tracking.config import LiveTrackingConfig  # noqa: E402
from gui_live_tracking.ros2_source import Ros2FrameSource  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive one synchronized direct ROS 2 tracking sample.")
    parser.add_argument("--image-topic", default="/stereo/left/image")
    parser.add_argument("--joint-topic", default="/dvrk/PSM3/state_joint_current")
    parser.add_argument("--jaw-topic", default="/dvrk/PSM3/state_jaw_current")
    parser.add_argument("--sync-queue-size", type=int, default=5)
    parser.add_argument("--sync-slop", type=float, default=0.015)
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = LiveTrackingConfig(
        input_mode="ros2",
        ros_image_topic=args.image_topic,
        ros_joint_topic=args.joint_topic,
        ros_jaw_topic=args.jaw_topic,
        ros_sync_queue_size=args.sync_queue_size,
        ros_sync_slop_sec=args.sync_slop,
        sample_timeout_sec=args.timeout,
    )
    source = Ros2FrameSource(cfg)
    try:
        source.start()
        sample = source.get_sample(timeout_sec=args.timeout)
    finally:
        source.stop()

    if sample is None:
        print("No synchronized ROS 2 sample received before timeout.", file=sys.stderr)
        return 1

    print("image shape:", sample.frame_bgr.shape)
    print("joint count:", sample.raw_joint_angles.size)
    print("timestamp:", sample.timestamp_ns)
    print("source index:", sample.source_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
