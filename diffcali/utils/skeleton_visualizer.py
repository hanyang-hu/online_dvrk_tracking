import numpy as np
from collections import defaultdict
import cv2
import torch
import time

from diffcali.eval_dvrk.LND_fk import lndFK
from diffcali.utils.projection_utils import get_img_coords

class OneEuroFilter:
    """
    One-Euro filter for 2D points (e.g., keypoints in image coordinates)
    """
    def __init__(self, min_cutoff=1.0, beta=0.0, alpha_d=0.3, dt=1/30.):
        self.min_cutoff = min_cutoff  # minimum cutoff frequency
        self.beta = beta              # speed coefficient
        self.alpha_d = alpha_d        # smoothing factor for derivative
        self.dt = dt                  # time step

        self.prev_x = None            # previous filtered value
        self.dx_hat = np.array([0.0, 0.0], dtype=np.float32)  # filtered derivative

    def _compute_alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / self.dt)

    def reset(self, x_init):
        """Reset filter state"""
        self.prev_x = np.array(x_init, dtype=np.float32)
        self.dx_hat = np.array([0.0, 0.0], dtype=np.float32)

    def filter(self, x):
        x = np.array(x, dtype=np.float32)
        if self.prev_x is None:
            self.reset(x)
            return x.astype(int)

        # 1) Compute derivative
        dx = (x - self.prev_x) / self.dt
        self.dx_hat = self.alpha_d * dx + (1 - self.alpha_d) * self.dx_hat

        # 2) Adaptive cutoff frequency
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx_hat)
        alpha = self._compute_alpha(cutoff)

        # 3) Filtered value
        x_hat = alpha * x + (1 - alpha) * self.prev_x

        # 4) Update state
        self.prev_x = x_hat

        return x_hat.astype(int)


class SkeletonVisualizer:
    def __init__(
        self,
        model,
        ctrnet_args,
        args,
        intr,
        p_local1,
        p_local2,
        thickness=5,
        use_filter=True,
        freq=30,
        min_cutoff=0.4,
        beta=0.01,
        alpha_d=0.3,
    ):
        self.model = model
        self.ctrnet_args = ctrnet_args
        self.args = args
        self.intr = intr
        self.p_local1 = p_local1
        self.p_local2 = p_local2
        self.thickness = thickness
        self.use_filter = use_filter

        # One-Euro filters per keypoint
        self.filters = defaultdict(lambda: OneEuroFilter(min_cutoff=min_cutoff, beta=beta, alpha_d=alpha_d, dt=1/freq))

    def project_cam(self, p_cam):
        x = self.ctrnet_args.fx * (p_cam[0] / p_cam[2]) + self.ctrnet_args.px
        y = self.ctrnet_args.fy * (p_cam[1] / p_cam[2]) + self.ctrnet_args.py
        return (x, y)

    def _filter_point(self, name, pt):
        if not self.use_filter:
            return tuple(map(int, pt))
        return tuple(self.filters[name].filter(pt))

    def plot_skeleton_overlay(
        self,
        blended,
        cTr,
        joint_angles,
    ):
        # -----------------------------
        # Camera pose
        # -----------------------------
        pose_matrix = self.model.cTr_to_pose_matrix(cTr.unsqueeze(0)).squeeze(0)

        # -----------------------------
        # Forward kinematics
        # -----------------------------
        R_list, t_list = lndFK(joint_angles)

        # -----------------------------
        # SHAFT: image border → base
        # -----------------------------
        base_cam = pose_matrix[:3, 3].cpu().numpy()
        shaft_axis = pose_matrix[:3, :3][:, 2].cpu().numpy()
        shaft_axis = shaft_axis / np.linalg.norm(shaft_axis)
        p_neg = base_cam - 0.03 * shaft_axis

        pt_base = self._filter_point("base", self.project_cam(base_cam))
        pt_neg  = self._filter_point("p_neg", self.project_cam(p_neg))

        h, w, _ = blended.shape
        
        p0 = np.array(pt_base, dtype=np.float32)
        p1 = np.array(pt_neg, dtype=np.float32)

        d = p1 - p0
        norm = np.linalg.norm(d)
        if norm > 1e-6:
            d /= norm

            far_pt = p0 + d * max(w, h) * 2

            ok, _, border_pt = cv2.clipLine(
                (0, 0, w, h),
                tuple(p0.astype(int)),
                tuple(far_pt.astype(int))
            )

            if ok:
                pt_neg_far = border_pt
                cv2.line(blended, pt_base, pt_neg_far, (255, 255, 0), self.thickness)

        # cv2.line(blended, pt_neg, pt_base, (255, 255, 0), self.thickness)

        # -----------------------------
        # BASE → TIP END
        # -----------------------------
        tip_end_cam = (pose_matrix @ torch.cat([t_list[2], t_list[2].new_ones(1)]))[:3].cpu().numpy()
        pt_tip_end = self._filter_point("tip_end", self.project_cam(tip_end_cam))
        cv2.line(blended, pt_base, pt_tip_end, (0, 255, 0), self.thickness)

        # -----------------------------
        # TIP END → TIP 1 / TIP 2
        # -----------------------------
        tip_1 = get_img_coords(self.p_local1, R_list[2], t_list[2], pose_matrix, self.intr, None).cpu().numpy()
        tip_2 = get_img_coords(self.p_local2, R_list[3], t_list[3], pose_matrix, self.intr, None).cpu().numpy()

        pt_tip_1 = self._filter_point("tip_1", tip_1)
        pt_tip_2 = self._filter_point("tip_2", tip_2)

        cv2.line(blended, pt_tip_end, pt_tip_1, (0, 0, 255), self.thickness)
        cv2.line(blended, pt_tip_end, pt_tip_2, (255, 0, 0), self.thickness)

        return blended


class RealTimeVideoWriter:
    """
    Writes constant-FPS video whose *duration matches real wall-clock time*.
    If your processing is slow, it duplicates the last frame to fill time.
    """
    def __init__(self, path, fourcc, fps, frame_size):
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.writer = cv2.VideoWriter(path, fourcc, self.fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter at: {path}")

        self.t0 = None
        self.next_t = None
        self.last_frame = None

    def start(self):
        t = time.perf_counter()
        self.t0 = t
        self.next_t = t

    def write_realtime(self, frame_bgr):
        if self.t0 is None:
            self.start()

        self.last_frame = frame_bgr
        now = time.perf_counter()

        # Fill the timeline up to 'now'
        while self.next_t <= now:
            self.writer.write(self.last_frame)
            self.next_t += self.dt

    def release(self):
        # Optionally flush a tiny bit (not required). Keep simple:
        self.writer.release()
