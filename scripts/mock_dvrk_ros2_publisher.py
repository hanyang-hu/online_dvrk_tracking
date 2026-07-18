#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState


def load_joint_yaml(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Joint-angle YAML must be a nonempty mapping: {path}")
    entries = []
    for i in range(len(data)):
        key = str(i)
        if key not in data:
            raise ValueError(f"Joint-angle YAML missing key {key!r}")
        entries.append(np.asarray(data[key], dtype=np.float64))
    return np.stack(entries, axis=0)


def load_video_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame.copy())
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"Video contains no readable frames: {path}")
    return frames


def add_jitter_ns(timestamp_ns: int, jitter_ms: float) -> int:
    if jitter_ms <= 0:
        return timestamp_ns
    jitter = np.random.uniform(-jitter_ms, jitter_ms) * 1_000_000
    return max(0, int(timestamp_ns + jitter))


def set_stamp(header, timestamp_ns: int) -> None:
    header.stamp.sec = int(timestamp_ns // 1_000_000_000)
    header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)


def frame_to_image_msg(frame: np.ndarray, frame_id: str) -> Image:
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected BGR image shape HxWx3, got {frame.shape}")

    msg = Image()
    msg.header.frame_id = frame_id
    msg.height = int(frame.shape[0])
    msg.width = int(frame.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(frame.shape[1] * 3)
    msg.data = frame.tobytes()
    return msg


def build_joint_msg(joints: np.ndarray, timestamp_ns: int) -> JointState:
    msg = JointState()
    set_stamp(msg.header, timestamp_ns)
    msg.name = [f"joint_{i}" for i in range(max(0, len(joints) - 1))]
    msg.position = [float(v) for v in joints[:-1]]
    return msg


def build_jaw_msg(joints: np.ndarray, timestamp_ns: int) -> JointState:
    msg = JointState()
    set_stamp(msg.header, timestamp_ns)
    msg.name = ["jaw"]
    msg.position = [float(joints[-1])]
    return msg


class MockDvrkPublisher(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("mock_dvrk_ros2_publisher")
        self.args = args
        self.image_pub = self.create_publisher(Image, args.image_topic, qos_profile_sensor_data)
        self.joint_pub = self.create_publisher(JointState, args.joint_topic, qos_profile_sensor_data)
        self.jaw_pub = self.create_publisher(JointState, args.jaw_topic, qos_profile_sensor_data)
        self.frames = load_video_frames(Path(args.video))
        self.joint_angles = load_joint_yaml(Path(args.joint_angles))
        self.sample_count = min(len(self.frames), len(self.joint_angles))
        if self.sample_count <= 0:
            raise ValueError("Video and joint-angle YAML contain no paired samples.")
        self.image_messages = [
            frame_to_image_msg(frame, "camera_left_optical_frame")
            for frame in self.frames[: self.sample_count]
        ]
        self.joint_messages = [
            build_joint_msg(joints, 0)
            for joints in self.joint_angles[: self.sample_count]
        ]
        self.jaw_messages = [
            build_jaw_msg(joints, 0)
            for joints in self.joint_angles[: self.sample_count]
        ]
        self.frames = []
        self.publish_index = 0
        self.period = 1.0 / float(args.rate)
        self.start_time_ns = self.get_clock().now().nanoseconds
        self._rate_window_start = time.perf_counter()
        self._rate_window_count = 0
        self.timer = self.create_timer(self.period, self._publish_next)
        self.get_logger().info(
            f"Loaded {self.sample_count} paired samples. Publishing at {args.rate:.3f} Hz."
        )

    def _publish_next(self) -> None:
        callback_start = time.perf_counter()

        if not self.args.loop and self.publish_index >= self.sample_count:
            self.get_logger().info("Reached end of mock data.")
            rclpy.shutdown()
            return

        sample_index = self.publish_index % self.sample_count
        timestamp_ns = self.start_time_ns + int(round(self.publish_index * self.period * 1_000_000_000))
        timestamp_ns = add_jitter_ns(timestamp_ns, self.args.timestamp_jitter_ms)

        stamp_start = time.perf_counter()
        image_msg = self.image_messages[sample_index]
        arm_msg = self.joint_messages[sample_index]
        jaw_msg = self.jaw_messages[sample_index]
        set_stamp(image_msg.header, timestamp_ns)
        set_stamp(arm_msg.header, timestamp_ns)
        set_stamp(jaw_msg.header, timestamp_ns)
        stamp_time = time.perf_counter() - stamp_start

        publish_start = time.perf_counter()
        self.image_pub.publish(image_msg)
        self.joint_pub.publish(arm_msg)
        self.jaw_pub.publish(jaw_msg)
        publish_time = time.perf_counter() - publish_start
        callback_time = time.perf_counter() - callback_start

        if callback_time > min(0.08, self.period * 0.8):
            self.get_logger().warning(
                f"Slow callback at frame {sample_index}: "
                f"total={callback_time:.3f}s, "
                "read=0.000s, "
                "rewind=0.000s, "
                "conversion=0.000s, "
                f"stamp={stamp_time:.3f}s, "
                f"publish={publish_time:.3f}s"
            )

        self.publish_index += 1
        self._rate_window_count += 1
        now = time.perf_counter()
        elapsed = now - self._rate_window_start
        if elapsed >= 2.0:
            actual_rate = self._rate_window_count / elapsed
            self.get_logger().info(
                f"Actual publish rate: {actual_rate:.2f} Hz "
                f"(target {1.0 / self.period:.2f} Hz)"
            )
            self._rate_window_start = now
            self._rate_window_count = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a video and joint YAML as fake dVRK ROS 2 topics.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--joint-angles", required=True)
    parser.add_argument("--image-topic", default="/stereo/left/rectified_downscaled_image")
    parser.add_argument("--joint-topic", default="/PSM1/measured_js")
    parser.add_argument("--jaw-topic", default="/PSM1/jaw/measured_js")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--timestamp-jitter-ms", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = MockDvrkPublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
