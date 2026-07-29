from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LiveTrackingConfig:
    video_path: Path = Path("data/custom/bag1/left.mp4")
    joint_angles_path: Path = Path("data/custom/bag1/joint_angles.yaml")
    camera_calibration_path: Path = Path("data/custom/camera_calibration.yaml")
    handeye_path: Path = Path("data/custom/handeye.yaml")
    lnd_json_path: Path = Path("data/custom/LND.json")
    machine_label: str = "PSM1"
    input_mode: str = "offline"
    mock_rate_hz: float = 30.0
    mock_loop: bool = False
    sample_timeout_sec: float = 0.5
    ros_image_topic: str = "/stereo/left/rectified_downscaled_image"
    ros_joint_topic: str = "/PSM1/measured_js"
    ros_jaw_topic: str = "/PSM1/jaw/measured_js"
    ros_sync_queue_size: int = 5
    ros_sync_slop_sec: float = 0.015
    ros_overlay_topic: str = "/dvrk_tracking/overlay"
    ros_pose_topic: str = "/dvrk_tracking/pose"
    ros_optimized_joints_topic: str = "/dvrk_tracking/joint_states"
    ros_loss_topic: str = "/dvrk_tracking/loss"
    ros_fps_topic: str = "/dvrk_tracking/fps"
    ros_frame_id: str = "camera_left_optical_frame"
    ros_child_frame_id: str = "PSM1_joint4_tracked"
    mesh_dir: Path = Path("urdfs/dVRK/meshes")
    contour_tip_net_path: Path = Path("ContourTipNet/models/cnn_model.pth")
    use_low_res_mesh: bool = True
    downscale_factor: int = 2
    dark_factor: float = 0.7
    renderer: str = "nvdiffrast"  # nvdiffrast | pytorch3d
    searcher: str = "CMA-ES"
    use_lumped_error_init: bool = False
    online_iters: int = 3
    use_prev_joint_angles: bool = False
    use_pts_loss: bool = True
    use_contour_tip_net: bool = True
    rotation_parameterization: str = "MixAngle"
    use_filter: bool = True
    filter_option: str = "Kalman"
    cos_reparams: bool = True
    joint_angle_free_mode: bool = False
    use_turbo_handeye_init: bool = False
    virtual_handeye_save_path: Optional[Path] = None
    use_bo_initializer: bool = True
    sample_number: int = 2000
    batch_size: int = 50
    batch_iters: int = 100
    final_iters: int = 100
    popsize: int = 70

    @property
    def keypoints_path(self) -> Path:
        return self.video_path.parent / f"{self.machine_label}_keypoints.txt"

    @property
    def use_nvdiffrast(self) -> bool:
        return self.renderer.lower() == "nvdiffrast"
