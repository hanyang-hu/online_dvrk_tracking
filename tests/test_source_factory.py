from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.mock_live_source import MockLiveFrameSource
from gui_live_tracking.offline_source import OfflineFrameSource
from gui_live_tracking.result_sink import NullTrackingResultSink, Ros2BridgeResultSink
from gui_live_tracking.ros2_bridge_source import Ros2BridgeFrameSource
from gui_live_tracking.source_factory import create_frame_source, create_result_sink


def test_source_factory_modes():
    assert isinstance(create_frame_source(LiveTrackingConfig(input_mode="offline")), OfflineFrameSource)
    assert isinstance(create_frame_source(LiveTrackingConfig(input_mode="mock_live")), MockLiveFrameSource)
    assert isinstance(create_frame_source(LiveTrackingConfig(input_mode="ros2_bridge")), Ros2BridgeFrameSource)


def test_sink_factory_modes():
    assert isinstance(create_result_sink(LiveTrackingConfig(input_mode="offline")), NullTrackingResultSink)
    assert isinstance(create_result_sink(LiveTrackingConfig(input_mode="mock_live")), NullTrackingResultSink)
    assert isinstance(create_result_sink(LiveTrackingConfig(input_mode="ros2_bridge")), Ros2BridgeResultSink)
