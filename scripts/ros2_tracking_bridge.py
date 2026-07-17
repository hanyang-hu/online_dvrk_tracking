#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rclpy
import zmq
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui_live_tracking.bridge_transport import (  # noqa: E402
    deserialize_tracking_result,
    serialize_tracking_sample,
)
from gui_live_tracking.frame_source import TrackingSample  # noqa: E402


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ns_to_stamp(msg, timestamp_ns: int) -> None:
    msg.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)


class Ros2TrackingBridge(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("dvrk_tracking_bridge")
        self.args = args
        self.cv_bridge = CvBridge()
        self.context_zmq = zmq.Context.instance()
        self.sample_socket = self.context_zmq.socket(zmq.PUB)
        self.sample_socket.setsockopt(zmq.SNDHWM, 1)
        self.sample_socket.setsockopt(zmq.LINGER, 0)
        self.sample_socket.bind(args.input_endpoint)

        self.result_socket = self.context_zmq.socket(zmq.PULL)
        self.result_socket.setsockopt(zmq.RCVHWM, 1)
        self.result_socket.setsockopt(zmq.LINGER, 0)
        self.result_socket.bind(args.result_endpoint)

        self.image_sub = Subscriber(self, Image, args.image_topic, qos_profile=qos_profile_sensor_data)
        self.joint_sub = Subscriber(self, JointState, args.joint_topic, qos_profile=qos_profile_sensor_data)
        self.jaw_sub = Subscriber(self, JointState, args.jaw_topic, qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.joint_sub, self.jaw_sub],
            queue_size=args.sync_queue_size,
            slop=args.sync_slop,
        )
        self.sync.registerCallback(self._sample_callback)

        self.overlay_pub = self.create_publisher(Image, args.overlay_topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, args.pose_topic, 10)
        self.joints_pub = self.create_publisher(JointState, args.optimized_joints_topic, 10)
        self.loss_pub = self.create_publisher(Float32, args.loss_topic, 10)
        self.fps_pub = self.create_publisher(Float32, args.fps_topic, 10)
        self.result_timer = self.create_timer(0.005, self._poll_results)
        self.source_index = 0

    def _sample_callback(self, image_msg: Image, joint_msg: JointState, jaw_msg: JointState) -> None:
        try:
            frame_bgr = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert image: {exc}")
            return

        arm = np.asarray(joint_msg.position, dtype=np.float64)
        jaw = np.asarray(jaw_msg.position, dtype=np.float64)
        sample = TrackingSample(
            frame_bgr=np.ascontiguousarray(frame_bgr).copy(),
            raw_joint_angles=np.concatenate([arm, jaw]).astype(np.float64, copy=True),
            timestamp_ns=stamp_to_ns(image_msg.header.stamp),
            source_index=self.source_index,
        )
        self.source_index += 1
        try:
            self.sample_socket.send_multipart(serialize_tracking_sample(sample), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def _poll_results(self) -> None:
        while True:
            try:
                parts = self.result_socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return

            try:
                result = deserialize_tracking_result(parts)
            except Exception as exc:
                self.get_logger().warning(f"Failed to decode tracking result: {exc}")
                continue

            overlay_msg = self.cv_bridge.cv2_to_imgmsg(result.overlay_bgr, encoding="bgr8")
            overlay_msg.header.frame_id = result.frame_id
            ns_to_stamp(overlay_msg, result.timestamp_ns)
            self.overlay_pub.publish(overlay_msg)

            pose_msg = PoseStamped()
            pose_msg.header.frame_id = result.frame_id
            ns_to_stamp(pose_msg, result.timestamp_ns)
            pose_msg.pose.position.x = float(result.translation[0])
            pose_msg.pose.position.y = float(result.translation[1])
            pose_msg.pose.position.z = float(result.translation[2])
            pose_msg.pose.orientation.x = float(result.quaternion_xyzw[0])
            pose_msg.pose.orientation.y = float(result.quaternion_xyzw[1])
            pose_msg.pose.orientation.z = float(result.quaternion_xyzw[2])
            pose_msg.pose.orientation.w = float(result.quaternion_xyzw[3])
            self.pose_pub.publish(pose_msg)

            joints_msg = JointState()
            ns_to_stamp(joints_msg, result.timestamp_ns)
            joints_msg.header.frame_id = result.child_frame_id
            joints_msg.name = [f"optimized_joint_{i}" for i in range(len(result.optimized_joint_angles))]
            joints_msg.position = [float(v) for v in result.optimized_joint_angles]
            self.joints_pub.publish(joints_msg)

            self.loss_pub.publish(Float32(data=float(result.loss)))
            self.fps_pub.publish(Float32(data=float(result.fps)))

    def destroy_node(self) -> bool:
        self.sample_socket.close(linger=0)
        self.result_socket.close(linger=0)
        return super().destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge synchronized dVRK ROS 2 samples to the tracking GUI.")
    parser.add_argument("--image-topic", default="/stereo/left/image")
    parser.add_argument("--joint-topic", default="/dvrk/PSM3/state_joint_current")
    parser.add_argument("--jaw-topic", default="/dvrk/PSM3/state_jaw_current")
    parser.add_argument("--input-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--result-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--sync-queue-size", type=int, default=5)
    parser.add_argument("--sync-slop", type=float, default=0.015)
    parser.add_argument("--overlay-topic", default="/dvrk_tracking/overlay")
    parser.add_argument("--pose-topic", default="/dvrk_tracking/pose")
    parser.add_argument("--optimized-joints-topic", default="/dvrk_tracking/joint_states")
    parser.add_argument("--loss-topic", default="/dvrk_tracking/loss")
    parser.add_argument("--fps-topic", default="/dvrk_tracking/fps")
    return parser.parse_args()


def main() -> int:
    rclpy.init()
    node = Ros2TrackingBridge(parse_args())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
