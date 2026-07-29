from pathlib import Path
from typing import List

import yaml

from gui_live_tracking.config import LiveTrackingConfig


def _validate_file(path: Path, label: str, errors: List[str]) -> None:
    if not path.exists():
        errors.append(f"{label} does not exist: {path}")
        return
    if not path.is_file():
        errors.append(f"{label} is not a file: {path}")


def validate_config(config: LiveTrackingConfig) -> List[str]:
    errors: List[str] = []
    mode = config.input_mode.lower()

    if mode not in {"offline", "mock_live", "ros2"}:
        errors.append(f"Input mode must be one of offline, mock_live, ros2: {config.input_mode}")

    if mode in {"offline", "mock_live"}:
        _validate_file(config.video_path, "Video", errors)
        _validate_file(config.joint_angles_path, "Joint angles yaml", errors)
    elif mode == "ros2":
        for label, topic in [
            ("ROS image topic", config.ros_image_topic),
            ("ROS arm joint topic", config.ros_joint_topic),
            ("ROS jaw topic", config.ros_jaw_topic),
            ("ROS overlay topic", config.ros_overlay_topic),
            ("ROS pose topic", config.ros_pose_topic),
            ("ROS optimized joints topic", config.ros_optimized_joints_topic),
            ("ROS loss topic", config.ros_loss_topic),
            ("ROS fps topic", config.ros_fps_topic),
        ]:
            if not topic.strip():
                errors.append(f"{label} must not be empty.")
        if config.ros_sync_queue_size <= 0:
            errors.append("ROS synchronization queue size must be positive.")
        if config.ros_sync_slop_sec < 0:
            errors.append("ROS synchronization slop must be nonnegative.")

    if config.mock_rate_hz <= 0:
        errors.append("Mock replay rate must be positive.")
    if config.sample_timeout_sec <= 0:
        errors.append("Sample timeout must be positive.")
    if config.use_turbo_handeye_init:
        if config.batch_size <= 0:
            errors.append("TuRBO batch size must be positive.")
        if config.sample_number <= 0:
            errors.append("TuRBO sample number must be positive.")
        if config.sample_number <= config.batch_size:
            errors.append("TuRBO sample number must be greater than batch size.")
        if config.batch_size > 0 and config.sample_number % config.batch_size != 0:
            errors.append("TuRBO sample number must be divisible by batch size.")
        if config.batch_iters <= 0:
            errors.append("TuRBO batch iterations must be positive.")
        if config.virtual_handeye_save_path is not None:
            save_parent = config.virtual_handeye_save_path.parent
            if not save_parent.exists() or not save_parent.is_dir():
                errors.append(f"Virtual hand-eye save directory does not exist: {save_parent}")

    _validate_file(config.camera_calibration_path, "Camera calibration", errors)
    _validate_file(config.handeye_path, "Hand-eye calibration", errors)
    _validate_file(config.lnd_json_path, "LND json", errors)

    if config.use_pts_loss:
        _validate_file(config.contour_tip_net_path, "ContourTipNet weights", errors)

    if not config.mesh_dir.exists() or not config.mesh_dir.is_dir():
        errors.append(f"Mesh directory does not exist: {config.mesh_dir}")

    # Basic parse checks catch malformed config files early.
    if not errors:
        try:
            with open(config.handeye_path, "r", encoding="utf-8") as f:
                handeye = yaml.load(f, Loader=yaml.FullLoader)
            tvec_key = f"{config.machine_label}_tvec"
            rvec_key = f"{config.machine_label}_rvec"
            if tvec_key not in handeye or rvec_key not in handeye:
                if "PSM1_tvec" not in handeye or "PSM1_rvec" not in handeye:
                    errors.append(
                        "Hand-eye yaml missing both machine-specific and PSM1 fallback keys."
                    )
        except Exception as exc:
            errors.append(f"Failed to parse handeye yaml: {exc}")

        try:
            with open(config.camera_calibration_path, "r", encoding="utf-8") as f:
                yaml.load(f, Loader=yaml.FullLoader)
        except Exception as exc:
            errors.append(f"Failed to parse camera calibration yaml: {exc}")

    return errors
