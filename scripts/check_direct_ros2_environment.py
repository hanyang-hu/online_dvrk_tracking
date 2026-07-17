#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


def main() -> int:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("ROS distribution:", os.environ.get("ROS_DISTRO", "<not set>"))
    print("RMW implementation:", os.environ.get("RMW_IMPLEMENTATION", "<not set>"))

    try:
        import cv2
        import numpy as np
        import rclpy
        from cv_bridge import CvBridge
        from message_filters import ApproximateTimeSynchronizer, Subscriber
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
    except Exception as exc:
        print(f"ERROR: ROS/Conda import preflight failed: {exc}", file=sys.stderr)
        print(
            "Activate online_dvrk, source /opt/ros/humble/setup.bash, and check for "
            "Conda/ROS shared-library conflicts. This script does not modify "
            "LD_LIBRARY_PATH automatically.",
            file=sys.stderr,
        )
        return 1

    print("rclpy location:", rclpy.__file__)
    print("NumPy version:", np.__version__)
    print("OpenCV location:", cv2.__file__)
    print("OpenCV version:", cv2.__version__)
    print("cv_bridge availability:", CvBridge)
    print("message_filters availability:", ApproximateTimeSynchronizer, Subscriber)

    context = Context()
    executor = None
    node = None
    try:
        rclpy.init(context=context)
        node = Node("direct_ros2_environment_check", context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        executor.spin_once(timeout_sec=0.1)
        executor.remove_node(node)
        node.destroy_node()
        node = None
        context.shutdown()
    except Exception as exc:
        print(f"ERROR: ROS runtime smoke test failed: {exc}", file=sys.stderr)
        print(
            "This can happen when ROS Humble Python packages load incompatible "
            "Conda shared libraries. Check ROS_DISTRO, PYTHONPATH, LD_LIBRARY_PATH, "
            "OpenCV, and NumPy versions.",
            file=sys.stderr,
        )
        return 1
    finally:
        if executor is not None:
            try:
                executor.shutdown()
            except Exception:
                pass
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if context.ok():
            try:
                context.shutdown()
            except Exception:
                pass

    print("Direct ROS 2 runtime smoke test succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
