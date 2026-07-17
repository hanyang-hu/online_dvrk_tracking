from __future__ import annotations

from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.frame_source import FrameSource
from gui_live_tracking.mock_live_source import MockLiveFrameSource
from gui_live_tracking.offline_source import OfflineFrameSource
from gui_live_tracking.result_sink import NullTrackingResultSink, Ros2ResultSink, TrackingResultSink


def create_frame_source(config: LiveTrackingConfig) -> FrameSource:
    mode = config.input_mode.lower()
    if mode == "offline":
        return OfflineFrameSource(config)
    if mode == "mock_live":
        return MockLiveFrameSource(config)
    if mode == "ros2":
        from gui_live_tracking.ros2_source import Ros2FrameSource

        return Ros2FrameSource(config)
    raise ValueError(f"Unsupported input mode: {config.input_mode!r}")


def create_result_sink(config: LiveTrackingConfig) -> TrackingResultSink:
    if config.input_mode.lower() == "ros2":
        return Ros2ResultSink(config)
    return NullTrackingResultSink()
