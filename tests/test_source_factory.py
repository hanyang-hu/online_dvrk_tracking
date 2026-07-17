import sys

from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.mock_live_source import MockLiveFrameSource
from gui_live_tracking.offline_source import OfflineFrameSource
from gui_live_tracking.result_sink import NullTrackingResultSink, Ros2ResultSink
from gui_live_tracking.source_factory import create_frame_source, create_result_sink


def test_source_factory_modes():
    assert isinstance(create_frame_source(LiveTrackingConfig(input_mode="offline")), OfflineFrameSource)
    assert isinstance(create_frame_source(LiveTrackingConfig(input_mode="mock_live")), MockLiveFrameSource)
    source = create_frame_source(LiveTrackingConfig(input_mode="ros2"))
    assert source.__class__.__name__ == "Ros2FrameSource"


def test_sink_factory_modes():
    assert isinstance(create_result_sink(LiveTrackingConfig(input_mode="offline")), NullTrackingResultSink)
    assert isinstance(create_result_sink(LiveTrackingConfig(input_mode="mock_live")), NullTrackingResultSink)
    assert isinstance(create_result_sink(LiveTrackingConfig(input_mode="ros2")), Ros2ResultSink)


def test_importing_source_factory_does_not_import_rclpy():
    sys.modules.pop("rclpy", None)

    create_frame_source(LiveTrackingConfig(input_mode="offline"))

    assert "rclpy" not in sys.modules
