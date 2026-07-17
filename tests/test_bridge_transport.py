import numpy as np
import pytest

from gui_live_tracking.bridge_transport import (
    TrackingResult,
    deserialize_tracking_result,
    deserialize_tracking_sample,
    serialize_tracking_result,
    serialize_tracking_sample,
)
from gui_live_tracking.frame_source import TrackingSample


def test_sample_transport_round_trip_copies_arrays():
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    joints = np.arange(7, dtype=np.float64)
    sample = TrackingSample(frame, joints, timestamp_ns=123, source_index=4)

    parts = serialize_tracking_sample(sample)
    decoded = deserialize_tracking_sample(parts)

    assert decoded.timestamp_ns == 123
    assert decoded.source_index == 4
    assert np.array_equal(decoded.frame_bgr, frame)
    assert np.array_equal(decoded.raw_joint_angles, joints)

    parts[1] = bytes(len(parts[1]))
    assert np.array_equal(decoded.frame_bgr, frame)


def test_result_transport_round_trip():
    overlay = np.ones((2, 2, 3), dtype=np.uint8)
    result = TrackingResult(
        timestamp_ns=456,
        source_index=9,
        frame_id="camera",
        child_frame_id="tool",
        translation=[1, 2, 3],
        quaternion_xyzw=[0, 0, 0, 1],
        optimized_joint_angles=[0.1, 0.2],
        loss=0.3,
        fps=12.5,
        overlay_bgr=overlay,
    )

    decoded = deserialize_tracking_result(serialize_tracking_result(result))

    assert decoded.timestamp_ns == result.timestamp_ns
    assert decoded.source_index == result.source_index
    assert decoded.frame_id == "camera"
    assert decoded.child_frame_id == "tool"
    assert decoded.translation == [1.0, 2.0, 3.0]
    assert decoded.quaternion_xyzw == [0.0, 0.0, 0.0, 1.0]
    assert decoded.optimized_joint_angles == [0.1, 0.2]
    assert decoded.loss == pytest.approx(0.3)
    assert decoded.fps == pytest.approx(12.5)
    assert np.array_equal(decoded.overlay_bgr, overlay)


def test_invalid_shapes_and_lengths_are_rejected():
    sample = TrackingSample(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros(7), 0, 0)
    parts = serialize_tracking_sample(sample)
    parts[1] = parts[1][:-1]

    with pytest.raises(ValueError, match="byte length mismatch"):
        deserialize_tracking_sample(parts)

    with pytest.raises(ValueError, match="HxWx3"):
        serialize_tracking_sample(TrackingSample(np.zeros((2, 2), dtype=np.uint8), np.zeros(7), 0, 0))
