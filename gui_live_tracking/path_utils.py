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

    _validate_file(config.video_path, "Video", errors)
    _validate_file(config.camera_calibration_path, "Camera calibration", errors)
    _validate_file(config.handeye_path, "Hand-eye calibration", errors)
    _validate_file(config.lnd_json_path, "LND json", errors)
    _validate_file(config.joint_angles_path, "Joint angles yaml", errors)

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
