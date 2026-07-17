from dataclasses import dataclass
from pathlib import Path


@dataclass
class LiveTrackingConfig:
    video_path: Path
    joint_angles_path: Path
    camera_calibration_path: Path
    handeye_path: Path
    lnd_json_path: Path
    machine_label: str = "PSM3"
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
