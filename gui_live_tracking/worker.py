from __future__ import annotations

import os
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
import yaml
from PySide6.QtCore import QMutex, QMutexLocker, QObject, Signal

# Local path bootstrap for script-style usage.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "SurgicalSAM2") not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT / "SurgicalSAM2"))
if str(REPO_ROOT / "ParticleFilter") not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT / "ParticleFilter"))
if str(REPO_ROOT / "TuRBO") not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT / "TuRBO"))

from core.RobotLink import RobotLink
from diffcali.eval_dvrk.trackers import Tracker
from diffcali.models.CtRNet import CtRNet
from diffcali.utils.angle_transform_utils import axis_angle_to_mix_angle
from diffcali.utils.link_utils import axisAngleToRotationMatrix
from diffcali.utils.skeleton_visualizer import SkeletonVisualizer
from sam2.build_sam import build_sam2_camera_predictor

from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.result_sink import TrackingResult
from gui_live_tracking.source_factory import create_frame_source, create_result_sink


class TrackingWorker(QObject):
    frame_ready = Signal(object)
    paused = Signal(object, int)
    status = Signal(str)
    metrics = Signal(float, float, int)
    virtual_handeye_saved = Signal(object)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, config: LiveTrackingConfig, prompt_points: List[Tuple[int, int]], prompt_labels: List[int]):
        super().__init__()
        self.config = config
        self.prompt_points = prompt_points
        self.prompt_labels = prompt_labels
        self._stop = False
        self._pause_requested = False
        self._mutex = QMutex()
        self._pending_reinit_prompts: Optional[Tuple[List[Tuple[int, int]], List[int]]] = None
        self._waiting_for_ros = False
        self._logged_ros_sample_details = False

        self._runtime_online_iters = config.online_iters
        self._runtime_use_lumped = config.use_lumped_error_init
        self._runtime_joint_angle_free = config.joint_angle_free_mode
        self._turbo_handeye_applied = False
        self._joint_free_lumped_warned = False

    def request_stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._stop = True

    def request_pause(self) -> None:
        with QMutexLocker(self._mutex):
            self._pause_requested = True

    def resume(self) -> None:
        with QMutexLocker(self._mutex):
            self._pause_requested = False

    def resume_with_reinit(self, prompt_points: List[Tuple[int, int]], prompt_labels: List[int]) -> None:
        with QMutexLocker(self._mutex):
            self._pending_reinit_prompts = (prompt_points, prompt_labels)
            self._pause_requested = False

    def update_runtime(self, online_iters: int, use_lumped_error: bool, joint_angle_free: bool) -> None:
        with QMutexLocker(self._mutex):
            self._runtime_online_iters = max(1, int(online_iters))
            self._runtime_use_lumped = bool(use_lumped_error)
            self._runtime_joint_angle_free = bool(joint_angle_free)

    def _snapshot_runtime(self) -> Tuple[bool, bool, int, bool, bool]:
        with QMutexLocker(self._mutex):
            return (
                self._stop,
                self._pause_requested,
                self._runtime_online_iters,
                self._runtime_use_lumped,
                self._runtime_joint_angle_free,
            )

    def _consume_pending_reinit_prompts(self) -> Optional[Tuple[List[Tuple[int, int]], List[int]]]:
        with QMutexLocker(self._mutex):
            prompts = self._pending_reinit_prompts
            self._pending_reinit_prompts = None
            return prompts

    def _wait_while_paused(self, frame_rgb: np.ndarray, frame_idx: int) -> Optional[Tuple[List[Tuple[int, int]], List[int]]]:
        self.status.emit(f"Paused at frame {frame_idx}. You can continue or re-initialize from this frame.")
        self.paused.emit(frame_rgb.copy(), frame_idx)

        while True:
            should_stop, pause_requested, _, _, _ = self._snapshot_runtime()
            if should_stop:
                return None
            if not pause_requested:
                return self._consume_pending_reinit_prompts()
            time.sleep(0.05)

    def run(self) -> None:
        try:
            self._run_impl()
            self.finished.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _make_stdev_init(self, use_prev_joint_angles: bool) -> torch.Tensor:
        stdev_init = torch.tensor([1.0] * 10, dtype=torch.float32).cuda()
        stdev_init[:3] *= torch.tensor([1e-2, 1e-1, 1e-2], dtype=torch.float32).cuda()
        stdev_init[3:6] *= 1e-3
        stdev_init[6:] *= 5e-2
        stdev_init = stdev_init.detach()
        stdev_init[6] *= 2
        stdev_init[7] *= 2
        stdev_init[8:] *= 2
        if not use_prev_joint_angles:
            stdev_init[6:] /= 10.0
        return stdev_init

    def _apply_joint_mode_to_tracker(
        self,
        args: SimpleNamespace,
        tracker: Tracker,
        joint_angle_free: bool,
    ) -> None:
        args.use_prev_joint_angles = bool(joint_angle_free)
        args.stdev_init = self._make_stdev_init(args.use_prev_joint_angles)
        tracker.stdev_init = args.stdev_init[:9] if args.symmetric_jaw else args.stdev_init

    def _build_args(self) -> SimpleNamespace:
        cfg = self.config
        args = SimpleNamespace()
        args.mesh_dir = str(cfg.mesh_dir)
        args.batch_opt_lr = 3e-3
        args.single_opt_lr = 5e-4
        args.batch_size = cfg.batch_size
        args.dark_factor = cfg.dark_factor
        args.batch_iters = cfg.batch_iters
        args.final_iters = cfg.final_iters
        args.arm = "psm2"
        args.sample_number = cfg.sample_number
        args.use_bo_initializer = cfg.use_turbo_handeye_init
        args.use_nvdiffrast = cfg.use_nvdiffrast
        args.searcher = cfg.searcher
        args.online_iters = cfg.online_iters
        args.no_cache = True
        args.use_lumped_error_init = cfg.use_lumped_error_init
        args.interactive_prompts = True
        args.use_full_joint_angles = True
        args.downscale_factor = cfg.downscale_factor
        args.use_low_res_mesh = cfg.use_low_res_mesh
        args.symmetric_jaw = True
        args.use_render_loss = True
        args.use_pts_loss = cfg.use_pts_loss
        args.use_prev_joint_angles = cfg.use_prev_joint_angles or cfg.joint_angle_free_mode
        args.rotation_parameterization = cfg.rotation_parameterization
        args.mse_weight = 6.0
        args.dist_weight = 0.0
        args.app_weight = 6e-6
        args.pts_weight = 3e-3
        args.use_contour_tip_net = cfg.use_contour_tip_net
        args.contour_tip_net_path = str(cfg.contour_tip_net_path)
        args.popsize = cfg.popsize
        args.filter_option = cfg.filter_option if cfg.use_filter else "None"
        args.cos_reparams = cfg.cos_reparams
        args.video_label = "gui"
        args.machine_label = cfg.machine_label
        args.use_filter = args.filter_option != "None"
        args.use_mix_angle = args.rotation_parameterization == "MixAngle"

        # Tracker expects the concrete tensor value here; RealOrVector is only a typing alias.
        args.stdev_init = self._make_stdev_init(args.use_prev_joint_angles)

        return args

    def _load_intrinsics(self, camera_yaml: Path) -> Tuple[float, float, float, float]:
        with open(camera_yaml, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        if isinstance(data, dict):
            if "camera_matrix" in data and isinstance(data["camera_matrix"], dict):
                vals = data["camera_matrix"].get("data", None)
                if vals and len(vals) >= 9:
                    fx = float(vals[0])
                    fy = float(vals[4])
                    px = float(vals[2])
                    py = float(vals[5])
                    return fx, fy, px, py
            if all(k in data for k in ["fx", "fy", "px", "py"]):
                return float(data["fx"]), float(data["fy"]), float(data["px"]), float(data["py"])

        # Fallback to existing custom defaults.
        return 1025.88223, 1025.88223, 167.919017, 234.152707

    def _initialization(self, cam_T_b: np.ndarray, joint_angles: np.ndarray, psm_arm: RobotLink) -> Tuple[torch.Tensor, torch.Tensor]:
        psm_arm.updateJointAngles(joint_angles)

        T_4 = np.dot(cam_T_b, psm_arm.baseToJointT[3])
        R, t_vec = T_4[:3, :3], T_4[:3, 3]
        R_ = torch.from_numpy(R).float().cuda()
        T_ = torch.from_numpy(t_vec).float().cuda()
        axis_angle = kornia.geometry.conversions.rotation_matrix_to_axis_angle(R_.unsqueeze(0)).squeeze(0)
        pose_vec = torch.cat([axis_angle, T_], dim=0)

        visible_joint_angles = torch.from_numpy(joint_angles).float().cuda()[-3:]
        visible_joint_angles[-1] /= 2.0
        visible_joint_angles = torch.cat([visible_joint_angles, visible_joint_angles[-1].unsqueeze(0)], dim=0)
        return pose_vec, visible_joint_angles

    def _visible_joint_angles_from_raw(self, raw_joint_angles: np.ndarray) -> torch.Tensor:
        visible_joint_angles = torch.from_numpy(np.asarray(raw_joint_angles)).float().cuda()[-3:]
        visible_joint_angles[-1] /= 2.0
        return torch.cat([visible_joint_angles, visible_joint_angles[-1].unsqueeze(0)], dim=0)

    def _rotation_matrix_to_quaternion_xyzw(self, rot: np.ndarray) -> List[float]:
        trace = float(np.trace(rot))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rot[2, 1] - rot[1, 2]) / s
            qy = (rot[0, 2] - rot[2, 0]) / s
            qz = (rot[1, 0] - rot[0, 1]) / s
        else:
            diag = np.diag(rot)
            if diag[0] > diag[1] and diag[0] > diag[2]:
                s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
                qw = (rot[2, 1] - rot[1, 2]) / s
                qx = 0.25 * s
                qy = (rot[0, 1] + rot[1, 0]) / s
                qz = (rot[0, 2] + rot[2, 0]) / s
            elif diag[1] > diag[2]:
                s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
                qw = (rot[0, 2] - rot[2, 0]) / s
                qx = (rot[0, 1] + rot[1, 0]) / s
                qy = 0.25 * s
                qz = (rot[1, 2] + rot[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
                qw = (rot[1, 0] - rot[0, 1]) / s
                qx = (rot[0, 2] + rot[2, 0]) / s
                qy = (rot[1, 2] + rot[2, 1]) / s
                qz = 0.25 * s
        quat = np.array([qx, qy, qz, qw], dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm > 0:
            quat /= norm
        return quat.tolist()

    def _cTr_to_matrix(self, model: CtRNet, cTr: torch.Tensor) -> torch.Tensor:
        return model.cTr_to_pose_matrix(cTr.unsqueeze(0))[0]

    def _matrix_to_cTr(self, model: CtRNet, pose_matrix: torch.Tensor) -> torch.Tensor:
        return model.pose_matrix_to_cTr(pose_matrix.unsqueeze(0))[0]

    def _run_turbo_pose_initializer(
        self,
        mask: torch.Tensor,
        model: CtRNet,
        robot_renderer,
        ctrnet_args: SimpleNamespace,
        args: SimpleNamespace,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from TuRBO.turbo.turbo_1 import Turbo1
        from diffcali.eval_dvrk.black_box_optimize import BayesOptBatchProblem

        mask_path: Optional[str] = None
        try:
            mask_np = (mask.detach().squeeze() > 0).cpu().numpy().astype(np.uint8) * 255
            with tempfile.NamedTemporaryFile(prefix="dvrk_turbo_mask_", suffix=".png", delete=False) as f:
                mask_path = f.name
            if not cv2.imwrite(mask_path, mask_np):
                raise RuntimeError(f"Failed to write temporary TuRBO mask: {mask_path}")

            bo_batch_problem = BayesOptBatchProblem(
                model=model,
                robot_renderer=robot_renderer,
                ref_mask_file=mask_path,
                ref_keypoints=None,
                fx=ctrnet_args.fx,
                fy=ctrnet_args.fy,
                px=ctrnet_args.px,
                py=ctrnet_args.py,
                batch_size=args.batch_size,
                ld1=3,
                ld2=3,
                ld3=3,
                batch_iters=args.batch_iters,
                lr=args.batch_opt_lr,
            )

            turbo = Turbo1(
                f=bo_batch_problem,
                lb=np.array([0.10, 90.0 - 60.0, 0.0, 0.0, -1.5707, -1.3963, 0.0]),
                ub=np.array([0.17, 90.0 - 30.0, 360.0, 360.0, 0.0, 1.3963, 1.5707]),
                n_init=args.batch_size,
                max_evals=args.sample_number,
                batch_size=args.batch_size,
                max_cholesky_size=1000,
                n_training_steps=50,
                verbose=True,
                min_cuda=1000,
                device="cuda",
                batch_eval=True,
            )
            turbo.optimize()

            optimized_cTr_batch = bo_batch_problem.final_cTr_batch
            optimized_joint_angles_batch = bo_batch_problem.joint_angles_batch
            optimized_loss_batch = bo_batch_problem.final_loss_batch
            if optimized_cTr_batch is None or optimized_joint_angles_batch is None or optimized_loss_batch is None:
                raise RuntimeError("TuRBO did not produce any initialization candidates.")

            valid_mask = torch.isfinite(optimized_loss_batch).to(device=optimized_loss_batch.device)
            if not torch.any(valid_mask):
                raise RuntimeError("TuRBO produced no finite initialization losses.")

            valid_losses = optimized_loss_batch[valid_mask]
            valid_cTrs = optimized_cTr_batch[valid_mask]
            valid_joint_angles = optimized_joint_angles_batch[valid_mask]
            best_idx = torch.argmin(valid_losses)
            return (
                valid_cTrs[best_idx].detach().clone(),
                valid_joint_angles[best_idx].detach().clone(),
                valid_losses[best_idx].detach().clone(),
            )
        finally:
            if mask_path is not None:
                try:
                    os.unlink(mask_path)
                except OSError:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _virtual_handeye_from_first_frame(
        self,
        model: CtRNet,
        estimated_cTr_joint4: torch.Tensor,
        raw_joint_angles: np.ndarray,
        psm_arm: RobotLink,
    ) -> np.ndarray:
        """Solve cam_T_base from the TuRBO-estimated joint-4 pose and FK."""
        psm_arm.updateJointAngles(raw_joint_angles)
        fk_base_T_joint4 = np.asarray(psm_arm.baseToJointT[3], dtype=np.float64)
        estimated_cam_T_joint4 = (
            self._cTr_to_matrix(model, estimated_cTr_joint4)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        return estimated_cam_T_joint4 @ np.linalg.inv(fk_base_T_joint4)

    def _save_virtual_handeye_yaml(
        self,
        hand_eye_data: dict,
        cam_T_b: np.ndarray,
        loss: torch.Tensor,
        estimated_visible_joints: torch.Tensor,
    ) -> None:
        if self.config.virtual_handeye_save_path is None:
            return

        save_path = Path(self.config.virtual_handeye_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        out_data = dict(hand_eye_data) if isinstance(hand_eye_data, dict) else {}
        tvec_key = f"{self.config.machine_label}_tvec"
        rvec_key = f"{self.config.machine_label}_rvec"
        rvec, _ = cv2.Rodrigues(cam_T_b[:3, :3])
        out_data[tvec_key] = (cam_T_b[:3, 3] * 1000.0).astype(float).tolist()
        out_data[rvec_key] = rvec.reshape(3).astype(float).tolist()
        out_data["virtual_handeye_metadata"] = {
            "machine_label": self.config.machine_label,
            "source_handeye_path": str(self.config.handeye_path),
            "turbo_loss": float(loss.item()) if isinstance(loss, torch.Tensor) else float(loss),
            "estimated_visible_joints": estimated_visible_joints.detach().cpu().numpy().astype(float).tolist(),
            "units": {
                tvec_key: "millimeters",
                rvec_key: "axis-angle radians",
            },
        }

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(out_data, f, sort_keys=False)

        self.status.emit(f"Saved virtual calibrated hand-eye YAML: {save_path}")
        self.virtual_handeye_saved.emit(save_path)

    def _axis_to_optimizer_rot(self, cTr_axis: torch.Tensor, use_mix_angle: bool) -> torch.Tensor:
        cTr_opt = cTr_axis.clone()
        if use_mix_angle:
            cTr_opt[:3] = axis_angle_to_mix_angle(cTr_axis[:3].unsqueeze(0)).squeeze(0)
        return cTr_opt

    def _extract_mask_logits(self, sam_out) -> torch.Tensor:
        """Extract mask logits from SAM2 outputs across possible tuple/list shapes."""
        if isinstance(sam_out, tuple):
            # Common forms: (obj_ids, mask_logits) or (_, _, mask_logits)
            if len(sam_out) >= 3:
                mask_logits = sam_out[2]
            elif len(sam_out) >= 2:
                mask_logits = sam_out[1]
            else:
                raise RuntimeError(f"Unexpected SAM output tuple length: {len(sam_out)}")
        else:
            mask_logits = sam_out

        if isinstance(mask_logits, (list, tuple)):
            if len(mask_logits) == 0:
                raise RuntimeError("SAM returned an empty mask logits list.")
            mask_logits = mask_logits[0]

        if not torch.is_tensor(mask_logits):
            raise RuntimeError(f"SAM mask logits must be a tensor, got: {type(mask_logits)}")

        return mask_logits

    def _run_impl(self) -> None:
        cfg = self.config
        args = self._build_args()

        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        ctrnet_args = SimpleNamespace()
        ctrnet_args.use_gpu = True
        ctrnet_args.trained_on_multi_gpus = False
        ctrnet_args.height = 480
        ctrnet_args.width = 640
        fx, fy, px, py = self._load_intrinsics(cfg.camera_calibration_path)
        ctrnet_args.fx, ctrnet_args.fy, ctrnet_args.px, ctrnet_args.py = fx, fy, px, py
        ctrnet_args.scale = 1.0
        ctrnet_args.use_nvdiffrast = args.use_nvdiffrast

        model = CtRNet(ctrnet_args)

        if args.use_low_res_mesh:
            mesh_files = [
                f"{args.mesh_dir}/low_res_shaft_multi_cylinder.ply",
                f"{args.mesh_dir}/low_res_logo_low_res_1.ply",
                f"{args.mesh_dir}/low_res_jawright_lowres.ply",
                f"{args.mesh_dir}/low_res_jawleft_lowres.ply",
            ]
        else:
            mesh_files = [
                f"{args.mesh_dir}/shaft_multi_cylinder.ply",
                f"{args.mesh_dir}/logo_low_res_1.ply",
                f"{args.mesh_dir}/jawright_lowres.ply",
                f"{args.mesh_dir}/jawleft_lowres.ply",
            ]

        robot_renderer = model.setup_robot_renderer(mesh_files, downscale_factor=args.downscale_factor)
        robot_renderer.set_mesh_visibility([True, True, True, True])

        intr = torch.tensor(
            [[ctrnet_args.fx, 0, ctrnet_args.px], [0, ctrnet_args.fy, ctrnet_args.py], [0, 0, 1]],
            device="cuda",
            dtype=torch.float32,
        )

        tip_length = 0.0096
        p_local1 = torch.tensor([0.0, 0.0004, tip_length], dtype=torch.float32, device=model.device)
        p_local2 = torch.tensor([0.0, -0.0004, tip_length], dtype=torch.float32, device=model.device)

        predictor = build_sam2_camera_predictor(
            "./configs/sam2.1/sam2.1_hiera_s.yaml",
            "./SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth",
            vos_optimized=True,
        )

        skeleton_visualizer = SkeletonVisualizer(model, ctrnet_args, args, intr, p_local1, p_local2, thickness=5)

        def build_tracker(init_cTr: torch.Tensor, init_joint_angles: torch.Tensor, num_iters: int) -> Tracker:
            return Tracker(
                model=model,
                robot_renderer=robot_renderer,
                init_cTr=init_cTr.detach().clone(),
                init_joint_angles=init_joint_angles.detach().clone(),
                num_iters=num_iters,
                stdev_init=args.stdev_init,
                intr=intr,
                p_local1=p_local1,
                p_local2=p_local2,
                searcher=args.searcher,
                args=args,
            )

        def get_init_mask(**kwargs):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                return predictor.add_new_points(**kwargs)

        def get_next_mask(frame_bgr):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                return predictor.track(frame_bgr)

        with open(cfg.handeye_path, "r", encoding="utf-8") as f:
            hand_eye_data = yaml.load(f, Loader=yaml.FullLoader)

        cam_T_b = np.eye(4)
        tvec_key = f"{cfg.machine_label}_tvec"
        rvec_key = f"{cfg.machine_label}_rvec"
        if tvec_key in hand_eye_data and rvec_key in hand_eye_data:
            cam_T_b[:-1, -1] = np.array(hand_eye_data[tvec_key]) / 1000.0
            cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(hand_eye_data[rvec_key])
            self.status.emit(f"Using hand-eye calibration for {cfg.machine_label}.")
        else:
            cam_T_b[:-1, -1] = np.array(hand_eye_data["PSM1_tvec"]) / 1000.0
            cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(hand_eye_data["PSM1_rvec"])
            self.status.emit(
                f"Missing {cfg.machine_label} hand-eye keys in yaml. Falling back to PSM1 hand-eye."
            )

        psm_arm = RobotLink(str(cfg.lnd_json_path))

        init_done = False
        w_lumped = torch.eye(4, dtype=torch.float32, device=model.device)
        seg_time_lst: List[float] = []
        track_time_lst: List[float] = []
        source = create_frame_source(cfg)
        sink = create_result_sink(cfg)

        try:
            source.start()
            sink.start()
            self.status.emit(f"Using input mode: {cfg.input_mode}")
            if cfg.joint_angle_free_mode:
                self.status.emit(
                    "Joint-angle free mode enabled: first-frame joints seed initialization; "
                    "subsequent frames and re-init use only the previous optimized pose and joints."
                )

            while True:
                should_stop, _, runtime_iters, runtime_use_lumped, runtime_joint_angle_free = self._snapshot_runtime()
                if should_stop:
                    break
                if runtime_joint_angle_free and runtime_use_lumped:
                    runtime_use_lumped = False
                    if not self._joint_free_lumped_warned:
                        self.status.emit("Ignoring lumped-error init while joint-angle free mode is enabled.")
                        self._joint_free_lumped_warned = True

                sample = source.get_sample(timeout_sec=cfg.sample_timeout_sec)
                if sample is None:
                    if cfg.input_mode == "ros2":
                        if not self._waiting_for_ros:
                            self.status.emit("Waiting for ROS 2 samples...")
                            self._waiting_for_ros = True
                        continue
                    break

                if self._waiting_for_ros:
                    self.status.emit("ROS 2 sample received.")
                    self._waiting_for_ros = False

                frame_idx = sample.source_index
                frame = sample.frame_bgr.copy()
                raw_joint_angles = sample.raw_joint_angles.copy()
                if cfg.input_mode == "ros2" and not self._logged_ros_sample_details:
                    self.status.emit(
                        "ROS 2 first sample: "
                        f"frame={frame_idx}, image_shape={frame.shape}, "
                        f"image_dtype={frame.dtype}, joint_count={raw_joint_angles.size}, "
                        f"timestamp_ns={sample.timestamp_ns}, "
                        f"joints={np.round(raw_joint_angles, 5).tolist()}"
                    )
                    self._logged_ros_sample_details = True
                frame_shape_orig = (frame.shape[1], frame.shape[0])
                frame = cv2.resize(frame, (ctrnet_args.width, ctrnet_args.height))
                frame = (frame * args.dark_factor).astype(np.uint8)

                if not init_done:
                    if len(self.prompt_points) == 0:
                        raise RuntimeError("No prompt points provided. Add at least one FG prompt before starting.")

                    predictor.load_first_frame(frame)
                    pts_np = np.array(self.prompt_points, dtype=np.float32)
                    lbs_np = np.array(self.prompt_labels, dtype=np.int64)
                    out_mask_logits = self._extract_mask_logits(
                        get_init_mask(frame_idx=0, obj_id=0, points=pts_np, labels=lbs_np)
                    )
                    mask = (out_mask_logits.squeeze() > 0).float()
                    if cfg.input_mode == "ros2":
                        self.status.emit(
                            "ROS 2 initialization mask: "
                            f"shape={tuple(mask.shape)}, foreground_pixels={int(mask.sum().item())}"
                        )

                    if cfg.use_turbo_handeye_init and not self._turbo_handeye_applied:
                        self.status.emit(
                            "Running TuRBO first-frame pose initialization for virtual hand-eye correction..."
                        )
                        turbo_start = time.time()
                        turbo_cTr, turbo_joint_angles, turbo_loss = self._run_turbo_pose_initializer(
                            mask=mask,
                            model=model,
                            robot_renderer=robot_renderer,
                            ctrnet_args=ctrnet_args,
                            args=args,
                        )
                        cam_T_b = self._virtual_handeye_from_first_frame(
                            model=model,
                            estimated_cTr_joint4=turbo_cTr,
                            raw_joint_angles=raw_joint_angles,
                            psm_arm=psm_arm,
                        )
                        self._turbo_handeye_applied = True
                        self._save_virtual_handeye_yaml(
                            hand_eye_data=hand_eye_data,
                            cam_T_b=cam_T_b,
                            loss=turbo_loss,
                            estimated_visible_joints=turbo_joint_angles,
                        )
                        elapsed = time.time() - turbo_start
                        self.status.emit(
                            "TuRBO virtual hand-eye correction applied "
                            f"in {elapsed:.1f}s. Best loss: {float(turbo_loss.item()):.6g}. "
                            "cam_T_base was solved from FK, not copied from TuRBO. "
                            f"Estimated visible joints: {turbo_joint_angles.detach().cpu().numpy().round(4).tolist()}"
                        )

                    cTr, joint_angles = self._initialization(cam_T_b=cam_T_b, joint_angles=raw_joint_angles, psm_arm=psm_arm)
                    joint_angles = self._visible_joint_angles_from_raw(raw_joint_angles)

                    tracker = build_tracker(
                        init_cTr=cTr,
                        init_joint_angles=joint_angles,
                        num_iters=runtime_iters,
                    )
                    self._apply_joint_mode_to_tracker(args, tracker, runtime_joint_angle_free)

                    cTr, joint_angles, loss = tracker.track_frame(
                        ref_mask=mask,
                        joint_angles=joint_angles,
                        is_init=True,
                        keypoints=None,
                    )

                    if runtime_use_lumped:
                        cTr_A, _ = self._initialization(cam_T_b=cam_T_b, joint_angles=raw_joint_angles, psm_arm=psm_arm)
                        T_A = self._cTr_to_matrix(model, cTr_A)
                        T_B = self._cTr_to_matrix(model, cTr)
                        w_lumped = T_B @ torch.linalg.inv(T_A)

                    init_done = True
                else:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    out_mask_logits = self._extract_mask_logits(get_next_mask(frame))
                    torch.cuda.synchronize()
                    t1 = time.time()
                    seg_time_lst.append(t1 - t0)

                    mask = (out_mask_logits.squeeze() > 0).float()

                    torch.cuda.synchronize()
                    t2 = time.time()

                    tracker.num_iters = runtime_iters
                    self._apply_joint_mode_to_tracker(args, tracker, runtime_joint_angle_free)
                    if runtime_joint_angle_free:
                        cTr, joint_angles, loss = tracker.track_frame(
                            ref_mask=mask,
                            joint_angles=joint_angles,
                            is_init=False,
                            keypoints=None,
                            cTr_init=None,
                        )
                    elif runtime_use_lumped:
                        cTr_fk, joint_angles_fk = self._initialization(
                            cam_T_b=cam_T_b,
                            joint_angles=raw_joint_angles,
                            psm_arm=psm_arm,
                        )
                        T_A = self._cTr_to_matrix(model, cTr_fk)
                        T_init = w_lumped @ T_A
                        cTr_init_axis = self._matrix_to_cTr(model, T_init)
                        cTr_init = self._axis_to_optimizer_rot(cTr_init_axis, args.use_mix_angle)
                        cTr, joint_angles, loss = tracker.track_frame(
                            ref_mask=mask,
                            joint_angles=joint_angles_fk,
                            is_init=False,
                            keypoints=None,
                            cTr_init=cTr_init,
                        )
                        T_B = self._cTr_to_matrix(model, cTr)
                        w_lumped = T_B @ torch.linalg.inv(T_A)
                    else:
                        joint_angles_reading = self._visible_joint_angles_from_raw(raw_joint_angles)
                        cTr, joint_angles, loss = tracker.track_frame(
                            ref_mask=mask,
                            joint_angles=joint_angles_reading,
                            is_init=False,
                            keypoints=None,
                            cTr_init=None,
                        )

                    torch.cuda.synchronize()
                    t3 = time.time()
                    track_time_lst.append(t3 - t2)

                mask_np = (out_mask_logits.squeeze() > 0).cpu().numpy().astype(np.uint8) * 255
                color = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)
                blended = cv2.addWeighted(frame, 0.7, color, 0.3, 0)
                blended = skeleton_visualizer.plot_skeleton_overlay(blended, cTr, joint_angles)
                blended = cv2.resize(blended, frame_shape_orig)

                fps = 0.0
                if len(seg_time_lst) > 5 and len(track_time_lst) > 5:
                    avg_time = sum(seg_time_lst[-5:]) / len(seg_time_lst[-5:]) + sum(track_time_lst[-5:]) / len(track_time_lst[-5:])
                    fps = 1.0 / avg_time if avg_time > 0 else 0.0

                loss_val = float(loss.item()) if isinstance(loss, torch.Tensor) else float(loss)
                cv2.putText(
                    blended,
                    f"Frame: {frame_idx} | Loss: {loss_val:.4f} | FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )

                pose_matrix = self._cTr_to_matrix(model, cTr).detach().cpu().numpy()
                result = TrackingResult(
                    timestamp_ns=sample.timestamp_ns,
                    source_index=sample.source_index,
                    frame_id=cfg.ros_frame_id,
                    child_frame_id=cfg.ros_child_frame_id,
                    translation=pose_matrix[:3, 3].tolist(),
                    quaternion_xyzw=self._rotation_matrix_to_quaternion_xyzw(pose_matrix[:3, :3]),
                    optimized_joint_angles=joint_angles.detach().cpu().numpy().astype(float).tolist(),
                    loss=loss_val,
                    fps=fps,
                    overlay_bgr=blended.copy(),
                )
                sink.send_result(result)

                rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(rgb)
                self.metrics.emit(fps, loss_val, frame_idx)

                # Pause support: allow user to continue directly or re-initialize from the current sample.
                (
                    should_stop,
                    pause_requested,
                    runtime_iters_after,
                    runtime_use_lumped_after,
                    runtime_joint_angle_free_after,
                ) = self._snapshot_runtime()
                if should_stop:
                    break
                if runtime_joint_angle_free_after:
                    runtime_use_lumped_after = False
                if pause_requested:
                    paused_sample = sample
                    paused_frame = frame.copy()
                    paused_raw_rgb = cv2.cvtColor(paused_frame, cv2.COLOR_BGR2RGB)
                    reinit_prompts = self._wait_while_paused(paused_raw_rgb, frame_idx)
                    if reinit_prompts is None:
                        break

                    if reinit_prompts[0] and np.sum(np.array(reinit_prompts[1]) == 1) > 0:
                        predictor.load_first_frame(paused_frame)
                        pts_np = np.array(reinit_prompts[0], dtype=np.float32)
                        lbs_np = np.array(reinit_prompts[1], dtype=np.int64)
                        out_mask_logits = self._extract_mask_logits(
                            get_init_mask(frame_idx=0, obj_id=0, points=pts_np, labels=lbs_np)
                        )
                        mask = (out_mask_logits.squeeze() > 0).float()

                        if runtime_joint_angle_free_after:
                            cTr_reinit = cTr.detach().clone()
                            joint_reinit = joint_angles.detach().clone()
                            cTr_reinit_input = None
                        elif runtime_use_lumped_after:
                            paused_raw_joints = paused_sample.raw_joint_angles.copy()
                            cTr_reinit, _ = self._initialization(
                                cam_T_b=cam_T_b,
                                joint_angles=paused_raw_joints,
                                psm_arm=psm_arm,
                            )
                            joint_reinit = self._visible_joint_angles_from_raw(paused_raw_joints)
                            cTr_reinit_input = self._axis_to_optimizer_rot(cTr_reinit, args.use_mix_angle)
                        else:
                            paused_raw_joints = paused_sample.raw_joint_angles.copy()
                            cTr_reinit = cTr.detach().clone()
                            joint_reinit = self._visible_joint_angles_from_raw(paused_raw_joints)
                            cTr_reinit_input = None

                        # Re-initialize at the paused sample with the selected initialization source.
                        tracker = build_tracker(
                            init_cTr=cTr_reinit,
                            init_joint_angles=joint_reinit,
                            num_iters=runtime_iters_after,
                        )
                        self._apply_joint_mode_to_tracker(args, tracker, runtime_joint_angle_free_after)

                        cTr, joint_angles, loss = tracker.track_frame(
                            ref_mask=mask,
                            joint_angles=joint_reinit,
                            is_init=True,
                            keypoints=None,
                            cTr_init=cTr_reinit_input,
                        )

                        if runtime_use_lumped_after:
                            T_A = self._cTr_to_matrix(model, cTr_reinit)
                            T_B = self._cTr_to_matrix(model, cTr)
                            w_lumped = T_B @ torch.linalg.inv(T_A)

                        mask_np = (out_mask_logits.squeeze() > 0).cpu().numpy().astype(np.uint8) * 255
                        color = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)
                        reinit_blended = cv2.addWeighted(paused_frame, 0.7, color, 0.3, 0)
                        reinit_blended = skeleton_visualizer.plot_skeleton_overlay(reinit_blended, cTr, joint_angles)
                        reinit_blended = cv2.resize(reinit_blended, frame_shape_orig)
                        reinit_loss_val = float(loss.item()) if isinstance(loss, torch.Tensor) else float(loss)
                        reinit_pose_matrix = self._cTr_to_matrix(model, cTr).detach().cpu().numpy()
                        sink.send_result(
                            TrackingResult(
                                timestamp_ns=paused_sample.timestamp_ns,
                                source_index=paused_sample.source_index,
                                frame_id=cfg.ros_frame_id,
                                child_frame_id=cfg.ros_child_frame_id,
                                translation=reinit_pose_matrix[:3, 3].tolist(),
                                quaternion_xyzw=self._rotation_matrix_to_quaternion_xyzw(reinit_pose_matrix[:3, :3]),
                                optimized_joint_angles=joint_angles.detach().cpu().numpy().astype(float).tolist(),
                                loss=reinit_loss_val,
                                fps=fps,
                                overlay_bgr=reinit_blended.copy(),
                            )
                        )
                        rgb = cv2.cvtColor(reinit_blended, cv2.COLOR_BGR2RGB)
                        self.frame_ready.emit(rgb)
                        self.metrics.emit(fps, reinit_loss_val, frame_idx)
                        self.status.emit(f"Re-initialized at frame {frame_idx} with new prompts.")
                    else:
                        self.status.emit(f"Continuing from frame {frame_idx} without re-initialization.")
        finally:
            source.stop()
            sink.stop()
