from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.frame_source import TrackingSample


ROS_IMPORT_ERROR = (
    "Direct ROS 2 mode requires ROS 2 Humble Python packages inside the active "
    "Python 3.10 environment. Activate the online_dvrk Conda environment, source "
    "/opt/ros/humble/setup.bash, and run the README preflight test."
)


@dataclass(frozen=True)
class RosDependencies:
    rclpy: Any
    CvBridge: Any
    ApproximateTimeSynchronizer: Any
    Subscriber: Any
    MultiThreadedExecutor: Any
    Context: Any
    Node: Any
    DurabilityPolicy: Any
    HistoryPolicy: Any
    QoSProfile: Any
    ReliabilityPolicy: Any
    Image: Any
    JointState: Any


def _import_ros_dependencies() -> RosDependencies:
    try:
        import rclpy
        from cv_bridge import CvBridge
        from message_filters import ApproximateTimeSynchronizer, Subscriber
        from rclpy.context import Context
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image, JointState
    except ImportError as exc:
        raise RuntimeError(ROS_IMPORT_ERROR) from exc

    return RosDependencies(
        rclpy=rclpy,
        CvBridge=CvBridge,
        ApproximateTimeSynchronizer=ApproximateTimeSynchronizer,
        Subscriber=Subscriber,
        MultiThreadedExecutor=MultiThreadedExecutor,
        Context=Context,
        Node=Node,
        DurabilityPolicy=DurabilityPolicy,
        HistoryPolicy=HistoryPolicy,
        QoSProfile=QoSProfile,
        ReliabilityPolicy=ReliabilityPolicy,
        Image=Image,
        JointState=JointState,
    )


def _stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class LatestSampleBuffer:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sample: Optional[TrackingSample] = None
        self._closed = False

    def put(self, sample: TrackingSample) -> None:
        with self._condition:
            if self._closed:
                return
            self._sample = sample
            self._condition.notify_all()

    def get_latest(self, timeout_sec: float) -> Optional[TrackingSample]:
        timeout_sec = max(0.0, float(timeout_sec))
        with self._condition:
            if self._sample is None and not self._closed:
                self._condition.wait(timeout_sec)
            if self._sample is None:
                return None
            sample = self._sample
            self._sample = None
            return sample

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class Ros2FrameSource:
    def __init__(self, config: LiveTrackingConfig):
        self.config = config
        self._deps: Optional[RosDependencies] = None
        self._context = None
        self._node = None
        self._executor = None
        self._executor_thread: Optional[threading.Thread] = None
        self._bridge = None
        self._buffer = LatestSampleBuffer()
        self._source_index = 0
        self._sync = None
        self._subscribers: list[Any] = []

    def start(self) -> None:
        self.stop()
        self._buffer = LatestSampleBuffer()
        self._deps = _import_ros_dependencies()
        deps = self._deps

        self._context = deps.Context()
        deps.rclpy.init(context=self._context)
        self._node = deps.Node("dvrk_tracking_gui_input", context=self._context)
        self._bridge = deps.CvBridge()

        qos = deps.QoSProfile(
            history=deps.HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=deps.ReliabilityPolicy.BEST_EFFORT,
            durability=deps.DurabilityPolicy.VOLATILE,
        )
        self._subscribers = [
            deps.Subscriber(self._node, deps.Image, self.config.ros_image_topic, qos_profile=qos),
            deps.Subscriber(self._node, deps.JointState, self.config.ros_joint_topic, qos_profile=qos),
            deps.Subscriber(self._node, deps.JointState, self.config.ros_jaw_topic, qos_profile=qos),
        ]
        self._sync = deps.ApproximateTimeSynchronizer(
            self._subscribers,
            queue_size=self.config.ros_sync_queue_size,
            slop=self.config.ros_sync_slop_sec,
        )
        self._sync.registerCallback(self._on_synchronized_sample)

        self._executor = deps.MultiThreadedExecutor(num_threads=2, context=self._context)
        self._executor.add_node(self._node)
        self._executor_thread = threading.Thread(
            target=self._executor.spin,
            name="dvrk-ros2-input-executor",
            daemon=True,
        )
        self._executor_thread.start()

    def _on_synchronized_sample(self, image_msg: Any, joint_msg: Any, jaw_msg: Any) -> None:
        try:
            frame_bgr = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as exc:
            if self._node is not None:
                self._node.get_logger().warning(f"Failed to convert ROS image to bgr8: {exc}")
            return

        arm = np.asarray(joint_msg.position, dtype=np.float64)
        jaw = np.asarray(jaw_msg.position, dtype=np.float64)
        raw_joint_angles = np.concatenate([arm, jaw]).astype(np.float64, copy=True)
        sample = TrackingSample(
            frame_bgr=np.ascontiguousarray(frame_bgr).copy(),
            raw_joint_angles=raw_joint_angles,
            timestamp_ns=_stamp_to_ns(image_msg.header.stamp),
            source_index=self._source_index,
        )
        self._source_index += 1
        self._buffer.put(sample)

    def get_sample(self, timeout_sec: float = 0.5) -> Optional[TrackingSample]:
        return self._buffer.get_latest(timeout_sec)

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

        self._executor_thread = None
        self._executor = None
        self._node = None
        self._context = None
        self._bridge = None
        self._sync = None
        self._subscribers = []
