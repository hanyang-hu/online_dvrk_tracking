import sys
import time

import numpy as np

from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.mock_live_source import MockLiveFrameSource
from gui_live_tracking.offline_source import OfflineFrameSource
from tests.test_offline_source import write_video_and_joints


def test_offline_and_mock_live_sources_match_by_index(tmp_path):
    video_path, joint_path = write_video_and_joints(tmp_path, count=3)
    cfg = LiveTrackingConfig(video_path=video_path, joint_angles_path=joint_path, mock_rate_hz=1000.0)
    offline = OfflineFrameSource(cfg)
    mock = MockLiveFrameSource(cfg)
    offline.start()
    mock.start()
    try:
        for _ in range(3):
            a = offline.get_sample()
            b = mock.get_sample(timeout_sec=1.0)
            assert a.source_index == b.source_index
            assert np.array_equal(a.frame_bgr, b.frame_bgr)
            assert np.array_equal(a.raw_joint_angles, b.raw_joint_angles)
    finally:
        offline.stop()
        mock.stop()


def test_mock_live_skips_stale_samples_when_delayed(tmp_path):
    video_path, joint_path = write_video_and_joints(tmp_path, count=8)
    cfg = LiveTrackingConfig(video_path=video_path, joint_angles_path=joint_path, mock_rate_hz=20.0)
    source = MockLiveFrameSource(cfg)
    source.start()
    try:
        time.sleep(0.16)
        sample = source.get_sample(timeout_sec=0.1)
        assert sample is not None
        assert sample.source_index >= 2
    finally:
        source.stop()


def test_mock_live_code_path_does_not_import_rclpy(tmp_path):
    sys.modules.pop("rclpy", None)
    video_path, joint_path = write_video_and_joints(tmp_path, count=1)
    source = MockLiveFrameSource(LiveTrackingConfig(video_path=video_path, joint_angles_path=joint_path))
    source.start()
    source.stop()
    assert "rclpy" not in sys.modules
