from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.ros2_source import ROS_IMPORT_ERROR


@dataclass(frozen=True)
class TrackingResult:
    timestamp_ns: int
    source_index: int
    frame_id: str
    child_frame_id: str
    translation: Sequence[float]
    quaternion_xyzw: Sequence[float]
    optimized_joint_angles: Sequence[float]
    loss: float
    fps: float
    overlay_bgr: np.ndarray


class TrackingResultSink(Protocol):
    def start(self) -> None:
        ...

    def send_result(self, result: TrackingResult) -> None:
        ...

    def stop(self) -> None:
        ...


class LatestResultBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: Optional[TrackingResult] = None
        self._closed = False

    def put(self, result: TrackingResult) -> None:
        with self._lock:
            if self._closed:
                return
            self._result = result

    def pop_latest(self) -> Optional[TrackingResult]:
        with self._lock:
            result = self._result
            self._result = None
            return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._result = None


class NullTrackingResultSink:
    def start(self) -> None:
        pass

    def send_result(self, result: TrackingResult) -> None:
        del result

    def stop(self) -> None:
        pass


def _import_ros_result_dependencies():
    try:
        import rclpy
        from cv_bridge import CvBridge
        from geometry_msgs.msg import PoseStamped
        from rclpy.context import Context
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Float32
    except ImportError as exc:
        raise RuntimeError(ROS_IMPORT_ERROR) from exc

    return {
        "rclpy": rclpy,
        "CvBridge": CvBridge,
        "PoseStamped": PoseStamped,
        "Context": Context,
        "MultiThreadedExecutor": MultiThreadedExecutor,
        "Node": Node,
        "qos_profile_sensor_data": qos_profile_sensor_data,
        "Image": Image,
        "JointState": JointState,
        "Float32": Float32,
    }


def _set_header_stamp(msg: Any, timestamp_ns: int) -> None:
    msg.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)


class Ros2ResultSink:
    def __init__(self, config: LiveTrackingConfig):
        self.config = config
        self._deps: Optional[dict[str, Any]] = None
        self._context = None
        self._node = None
        self._executor = None
        self._executor_thread: Optional[threading.Thread] = None
        self._bridge = None
        self._buffer = LatestResultBuffer()
        self._timer = None
        self._overlay_pub = None
        self._pose_pub = None
        self._joints_pub = None
        self._loss_pub = None
        self._fps_pub = None

    def start(self) -> None:
        self.stop()
        self._buffer = LatestResultBuffer()
        self._deps = _import_ros_result_dependencies()
        deps = self._deps
        self._context = deps["Context"]()
        deps["rclpy"].init(context=self._context)
        self._node = deps["Node"]("dvrk_tracking_gui_output", context=self._context)
        self._bridge = deps["CvBridge"]()
        qos = deps["qos_profile_sensor_data"]

        self._overlay_pub = self._node.create_publisher(deps["Image"], self.config.ros_overlay_topic, qos)
        self._pose_pub = self._node.create_publisher(deps["PoseStamped"], self.config.ros_pose_topic, 10)
        self._joints_pub = self._node.create_publisher(deps["JointState"], self.config.ros_optimized_joints_topic, 10)
        self._loss_pub = self._node.create_publisher(deps["Float32"], self.config.ros_loss_topic, 10)
        self._fps_pub = self._node.create_publisher(deps["Float32"], self.config.ros_fps_topic, 10)
        self._timer = self._node.create_timer(0.01, self._publish_latest)

        self._executor = deps["MultiThreadedExecutor"](num_threads=1, context=self._context)
        self._executor.add_node(self._node)
        self._executor_thread = threading.Thread(
            target=self._executor.spin,
            name="dvrk-ros2-output-executor",
            daemon=True,
        )
        self._executor_thread.start()

    def send_result(self, result: TrackingResult) -> None:
        self._buffer.put(result)

    def _publish_latest(self) -> None:
        result = self._buffer.pop_latest()
        if result is None:
            return

        deps = self._deps
        overlay_msg = self._bridge.cv2_to_imgmsg(result.overlay_bgr, encoding="bgr8")
        overlay_msg.header.frame_id = result.frame_id
        _set_header_stamp(overlay_msg, result.timestamp_ns)
        self._overlay_pub.publish(overlay_msg)

        pose_msg = deps["PoseStamped"]()
        pose_msg.header.frame_id = result.frame_id
        _set_header_stamp(pose_msg, result.timestamp_ns)
        pose_msg.pose.position.x = float(result.translation[0])
        pose_msg.pose.position.y = float(result.translation[1])
        pose_msg.pose.position.z = float(result.translation[2])
        pose_msg.pose.orientation.x = float(result.quaternion_xyzw[0])
        pose_msg.pose.orientation.y = float(result.quaternion_xyzw[1])
        pose_msg.pose.orientation.z = float(result.quaternion_xyzw[2])
        pose_msg.pose.orientation.w = float(result.quaternion_xyzw[3])
        self._pose_pub.publish(pose_msg)

        joints_msg = deps["JointState"]()
        joints_msg.header.frame_id = result.child_frame_id
        _set_header_stamp(joints_msg, result.timestamp_ns)
        joints_msg.name = [f"optimized_joint_{i}" for i in range(len(result.optimized_joint_angles))]
        joints_msg.position = [float(v) for v in result.optimized_joint_angles]
        self._joints_pub.publish(joints_msg)

        self._loss_pub.publish(deps["Float32"](data=float(result.loss)))
        self._fps_pub.publish(deps["Float32"](data=float(result.fps)))

    def stop(self) -> None:
        self._buffer.close()
        if self._executor is not None:
            self._executor.shutdown()
        if self._executor_thread is not None:
            self._executor_thread.join(timeout=2.0)
        if self._executor is not None and self._node is not None:
            try:
                self._executor.remove_node(self._node)
            except Exception:
                pass
        if self._node is not None:
            self._node.destroy_node()
        if self._context is not None:
            try:
                self._context.shutdown()
            except Exception:
                pass

        self._deps = None
        self._context = None
        self._node = None
        self._executor = None
        self._executor_thread = None
        self._bridge = None
        self._timer = None
        self._overlay_pub = None
        self._pose_pub = None
        self._joints_pub = None
        self._loss_pub = None
        self._fps_pub = None
