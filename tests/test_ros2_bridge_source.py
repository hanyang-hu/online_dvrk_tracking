import time

import numpy as np
import zmq

from gui_live_tracking.bridge_transport import serialize_tracking_sample
from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.frame_source import TrackingSample
from gui_live_tracking.ros2_bridge_source import Ros2BridgeFrameSource


def make_sample(index):
    return TrackingSample(
        frame_bgr=np.full((2, 2, 3), index, dtype=np.uint8),
        raw_joint_angles=np.array([index], dtype=np.float64),
        timestamp_ns=index,
        source_index=index,
    )


def test_ros2_bridge_source_returns_newest_available_sample():
    context = zmq.Context()
    endpoint = "inproc://bridge-source-newest"
    pub = context.socket(zmq.PUB)
    pub.bind(endpoint)
    cfg = LiveTrackingConfig(input_mode="ros2_bridge", bridge_input_endpoint=endpoint)
    source = Ros2BridgeFrameSource(cfg, context=context)
    source.start()
    try:
        time.sleep(0.05)
        for i in range(3):
            pub.send_multipart(serialize_tracking_sample(make_sample(i)))
        sample = source.get_sample(timeout_sec=1.0)
        assert sample is not None
        assert sample.source_index == 2
    finally:
        source.stop()
        pub.close(linger=0)
        context.term()


def test_ros2_bridge_source_timeout_returns_none():
    context = zmq.Context()
    cfg = LiveTrackingConfig(input_mode="ros2_bridge", bridge_input_endpoint="inproc://bridge-source-timeout")
    source = Ros2BridgeFrameSource(cfg, context=context)
    source.start()
    try:
        assert source.get_sample(timeout_sec=0.01) is None
    finally:
        source.stop()
        context.term()


def test_missing_bridge_does_not_crash():
    cfg = LiveTrackingConfig(input_mode="ros2_bridge", bridge_input_endpoint="tcp://127.0.0.1:59997")
    source = Ros2BridgeFrameSource(cfg)
    source.start()
    try:
        assert source.get_sample(timeout_sec=0.01) is None
    finally:
        source.stop()
