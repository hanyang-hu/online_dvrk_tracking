import sys

import cv2
import numpy as np
import yaml

from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.offline_source import OfflineFrameSource


def write_video_and_joints(tmp_path, count=3):
    video_path = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (4, 3))
    frames = []
    for i in range(count):
        frame = np.full((3, 4, 3), i * 40, dtype=np.uint8)
        frames.append(frame)
        writer.write(frame)
    writer.release()

    joints = {str(i): [float(i + j) for j in range(7)] for i in range(count)}
    joint_path = tmp_path / "joints.yaml"
    with open(joint_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(joints, f)
    return video_path, joint_path


def test_offline_source_returns_copied_samples(tmp_path):
    video_path, joint_path = write_video_and_joints(tmp_path, count=2)
    source = OfflineFrameSource(LiveTrackingConfig(video_path=video_path, joint_angles_path=joint_path))
    source.start()
    try:
        sample = source.get_sample()
        assert sample is not None
        assert sample.source_index == 0
        assert sample.timestamp_ns == 0
        sample.frame_bgr[:] = 255
        sample.raw_joint_angles[:] = 99

        next_sample = source.get_sample()
        assert next_sample is not None
        assert next_sample.source_index == 1
        assert not np.all(next_sample.frame_bgr == 255)
        assert not np.all(next_sample.raw_joint_angles == 99)
    finally:
        source.stop()


def test_offline_code_path_does_not_import_rclpy(tmp_path):
    sys.modules.pop("rclpy", None)
    video_path, joint_path = write_video_and_joints(tmp_path, count=1)
    source = OfflineFrameSource(LiveTrackingConfig(video_path=video_path, joint_angles_path=joint_path))
    source.start()
    source.stop()
    assert "rclpy" not in sys.modules
