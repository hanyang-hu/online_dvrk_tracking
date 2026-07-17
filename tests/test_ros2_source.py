import builtins
import sys
import threading
import time

import numpy as np
import pytest

from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.frame_source import TrackingSample
from gui_live_tracking.ros2_source import LatestSampleBuffer, Ros2FrameSource


def make_sample(index):
    return TrackingSample(
        frame_bgr=np.full((2, 2, 3), index, dtype=np.uint8),
        raw_joint_angles=np.array([index], dtype=np.float64),
        timestamp_ns=index,
        source_index=index,
    )


def test_latest_sample_buffer_returns_newest_once():
    buffer = LatestSampleBuffer()
    buffer.put(make_sample(1))
    buffer.put(make_sample(2))

    sample = buffer.get_latest(timeout_sec=0.01)

    assert sample is not None
    assert sample.source_index == 2
    assert buffer.get_latest(timeout_sec=0.01) is None


def test_latest_sample_buffer_timeout_returns_none():
    buffer = LatestSampleBuffer()

    assert buffer.get_latest(timeout_sec=0.01) is None


def test_latest_sample_buffer_close_wakes_blocked_consumer():
    buffer = LatestSampleBuffer()
    result = []

    def wait_for_sample():
        result.append(buffer.get_latest(timeout_sec=10.0))

    thread = threading.Thread(target=wait_for_sample)
    thread.start()
    time.sleep(0.05)
    buffer.close()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert result == [None]


def test_ros2_source_import_is_lazy():
    sys.modules.pop("rclpy", None)
    source = Ros2FrameSource(LiveTrackingConfig(input_mode="ros2"))

    assert source is not None
    assert "rclpy" not in sys.modules


def test_ros2_source_reports_clear_error_when_rclpy_unavailable(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rclpy":
            raise ImportError("no rclpy here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    source = Ros2FrameSource(LiveTrackingConfig(input_mode="ros2"))

    with pytest.raises(RuntimeError, match="Direct ROS 2 mode requires ROS 2 Humble"):
        source.start()
