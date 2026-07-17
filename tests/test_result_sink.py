import numpy as np

from gui_live_tracking.result_sink import LatestResultBuffer, TrackingResult


def make_result(index):
    return TrackingResult(
        timestamp_ns=index,
        source_index=index,
        frame_id="camera",
        child_frame_id="tool",
        translation=[0, 0, 0],
        quaternion_xyzw=[0, 0, 0, 1],
        optimized_joint_angles=[0.0],
        loss=0.0,
        fps=0.0,
        overlay_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def test_latest_result_buffer_replaces_old_result():
    buffer = LatestResultBuffer()
    buffer.put(make_result(1))
    buffer.put(make_result(2))

    result = buffer.pop_latest()

    assert result is not None
    assert result.source_index == 2
    assert buffer.pop_latest() is None


def test_latest_result_buffer_close_drops_results():
    buffer = LatestResultBuffer()
    buffer.put(make_result(1))
    buffer.close()

    assert buffer.pop_latest() is None
