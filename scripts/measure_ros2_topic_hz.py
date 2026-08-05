#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from collections import deque
from typing import Deque, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class TopicHzNode(Node):
    def __init__(self, topic: str, window: int):
        super().__init__("measure_ros2_topic_hz")
        self.timestamps: Deque[float] = deque(maxlen=max(2, window))
        self.count = 0
        self.create_subscription(Image, topic, self._on_msg, qos_profile_sensor_data)

    def _on_msg(self, _msg: Image) -> None:
        self.timestamps.append(time.perf_counter())
        self.count += 1

    def hz(self) -> Optional[float]:
        if len(self.timestamps) < 2:
            return None
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return None
        return (len(self.timestamps) - 1) / elapsed

    def period_stats_ms(self) -> Optional[tuple[float, float, float]]:
        if len(self.timestamps) < 3:
            return None
        periods = [
            (self.timestamps[i] - self.timestamps[i - 1]) * 1000.0
            for i in range(1, len(self.timestamps))
        ]
        return min(periods), statistics.mean(periods), max(periods)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Image topic rate using ROS 2 sensor-data QoS.")
    parser.add_argument("topic", nargs="?", default="/stereo/left/rectified_downscaled_image")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--print-every", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = TopicHzNode(args.topic, args.window)
    start = time.perf_counter()
    last_print = start
    try:
        while rclpy.ok():
            now = time.perf_counter()
            if args.duration > 0 and now - start >= args.duration:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.perf_counter()
            if now - last_print >= args.print_every:
                rate = node.hz()
                stats = node.period_stats_ms()
                if rate is None:
                    print(f"received={node.count}, waiting for more samples...")
                elif stats is None:
                    print(f"received={node.count}, rate={rate:.2f} Hz")
                else:
                    min_ms, mean_ms, max_ms = stats
                    print(
                        f"received={node.count}, rate={rate:.2f} Hz, "
                        f"period_ms min/mean/max={min_ms:.1f}/{mean_ms:.1f}/{max_ms:.1f}"
                    )
                last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        final_rate = node.hz()
        if final_rate is not None:
            print(f"final rate={final_rate:.2f} Hz over last {len(node.timestamps)} samples")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
