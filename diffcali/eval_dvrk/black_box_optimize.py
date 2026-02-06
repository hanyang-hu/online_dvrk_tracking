import torch
import torch.nn.functional as F
import numpy as np
import skfmm
import math
import cv2
import os
import tqdm

from evotorch import Problem, SolutionBatch
from evotorch.algorithms import CMAES, SNES, XNES, CEM
from evotorch.logging import Logger, StdOutLogger

from diffcali.eval_dvrk.LND_fk import lndFK, batch_lndFK
from diffcali.utils.projection_utils import *
from diffcali.utils.cylinder_projection_utils import (
    projectCylinderTorch,
    transform_points_b,
)
from diffcali.utils.detection_utils import detect_lines

from diffcali.eval_dvrk.batch_optimize import BatchOptimize, HeterogeneousBatchOptimize


torch.set_default_dtype(torch.float32)


class ImageLogger(Logger):
    def __init__(
        self, 
        searcher, 
        *, 
        num_iters: int,
        interval: int = 1, 
        after_first_step: bool = False, 
        save_fig_dir="./bbox_optimize/"):
        # Call the super constructor
        super().__init__(searcher, interval=interval, after_first_step=after_first_step)

        self.problem = searcher.problem

        # Set up the target status
        os.makedirs(save_fig_dir, exist_ok=True)
        self.save_fig_dir=save_fig_dir
        self.progress_bar = tqdm.tqdm(range(num_iters), desc="Black box optimization...")

        self.best_solution, self.best_eval = None, float('inf')

    def __call__(self, status: dict):
        if self._after_first_step:
            n = self._steps_count
            self._steps_count += 1
        else:
            self._steps_count += 1
            n = self._steps_count

        if (n % self._interval) == 0:
            self._log(self._filter(status))

        self.progress_bar.update()
        self.progress_bar.set_postfix({"pop_best_eval" : status["pop_best_eval"]})

        # Update best value and evaluation
        if status["pop_best_eval"] < self.best_eval:
            self.best_solution = status["pop_best"].values.clone()
            self.best_eval = status["pop_best_eval"]

    def save_image_cv(self, overlay, save_path):
        overlay_bgr = (overlay * 255).astype(
            np.uint8
        )  # Convert to uint8 and BGR format
        cv2.imwrite(save_path, overlay_bgr)

    def _overlay_masks(
        self,
        rendered_np,
        ref_np,
        ref_pts,
        proj_pts,
        detected_lines_params,
        projected_lines_params,
    ):
        """
        Overlay the rendered and reference masks along with keypoints.

        Args:
            rendered_np: Rendered mask in NumPy format.
            ref_np: Reference mask in NumPy format.
            ref_pts: Reference keypoints selected by the user (Nx2).
            proj_pts: Projected keypoints from the rendered image (Nx2).
        Returns:
            overlay: Overlayed image with masks and keypoints.
        """

        h, w = rendered_np.shape
        overlay = np.zeros((h, w, 3), dtype=np.float32)

        # # Red for rendered mask
        # rendered_color = np.stack([np.zeros_like(rendered_np), np.zeros_like(rendered_np), rendered_np], axis=-1)
        # # Green for reference mask
        # ref_color = np.stack([np.zeros_like(ref_np), ref_np, np.zeros_like(ref_np)], axis=-1)

        # Light blue for rendered mask
        rendered_color = np.stack(
            [0.5 * rendered_np, 0.8 * rendered_np, np.zeros_like(rendered_np)], axis=-1
        )

        # Orange for reference mask
        ref_color = np.stack([ref_np, 0.6 * ref_np, 0.1 * ref_np], axis=-1)  # RGB?

        alpha = 0.5
        overlay = rendered_color * alpha + ref_color * alpha
        overlay = np.clip(overlay, 0, 1)

        # Convert to BGR format for OpenCV
        overlay = (overlay * 255).astype(np.uint8)
        center_ref_pt = torch.mean(ref_pts, dim=0)
        center_proj_pt = torch.mean(proj_pts, dim=0)
        # Draw reference keypoints in green

        if ref_pts != None:
            for ref_pt in ref_pts:
                u_ref, v_ref = int(ref_pt[0]), int(ref_pt[1])
                cv2.circle(
                    overlay,
                    (u_ref, v_ref),
                    radius=5,
                    color=(255, 0.6 * 255, 0.1 * 255),
                    thickness=-1,
                )  # Green

        u_ref, v_ref = int(center_ref_pt[0]), int(center_ref_pt[1])
        cv2.circle(
            overlay,
            (u_ref, v_ref),
            radius=5,
            color=(255, 0.6 * 255, 0.1 * 255),
            thickness=-1,
        )  # Green  BGR
        # Draw projected keypoints in red

        for proj_pt in proj_pts.squeeze():
            # print(f"debugging the project pts {proj_pts}")
            u_proj, v_proj = int(proj_pt[0].item()), int(proj_pt[1].item())
            cv2.circle(
                overlay, (u_proj, v_proj), radius=5, color=(0.7, 0.8, 0), thickness=-1
            )  # Red

        u_ref, v_ref = int(center_proj_pt[0]), int(center_proj_pt[1])
        cv2.circle(
            overlay, (u_ref, v_ref), radius=5, color=(0.7, 0.8, 0), thickness=-1
        )  # Green

        # Draw detected lines in blue (convert from line parameters to endpoints)
        if detected_lines_params is not None:
            for line_params in detected_lines_params:
                a, b = line_params
                (x1, y1), (x2, y2) = convert_line_params_to_endpoints(a, b, w, h)
                cv2.line(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (255, 0.6 * 255, 0.1 * 255),
                    thickness=2,
                )  # Blue

        # Draw projected lines in cyan (convert from line parameters to endpoints)
        if projected_lines_params is not None:
            for line_params in projected_lines_params:
                a, b = line_params
                (x1, y1), (x2, y2) = convert_line_params_to_endpoints(a, b, w, h)
                cv2.line(
                    overlay, (x1, y1), (x2, y2), (0.7, 0.8, 0), thickness=2
                )  # green

        return overlay

    def _log(self, status: dict):
        values = status['pop_best'].values.clone()
        cTr = values[:6] # choose the best candidate in the population
        joint_angles = values[6:]

        # print(joint_angles)

        robot_mesh = self.problem.robot_renderer.get_robot_mesh(joint_angles)

        rendered_image = self.problem.model.render_single_robot_mask(
            cTr, robot_mesh, self.problem.robot_renderer
        )  

        rendered_np = rendered_image.squeeze().detach().cpu().numpy()
        ref_np = self.problem.ref_mask.squeeze().detach().cpu().numpy()
        ref_pts = self.problem.ref_keypoints

        intr = torch.tensor(
            [[self.problem.fx, 0, self.problem.px], [0, self.problem.fy, self.problem.py], [0, 0, 1]],
            device="cuda",
            dtype=self.problem.joint_angles.dtype,
        )

        pose_matrix = self.problem.model.cTr_to_pose_matrix(cTr.unsqueeze(0))
        R_list, t_list = lndFK(joint_angles)
        R_list = R_list.to(self.problem.model.device)
        t_list = t_list.to(self.problem.model.device)
        p_local1 = (
            torch.tensor([0.0, 0.0004, 0.009])
            .to(self.problem.joint_angles.dtype)
            .to(self.problem.model.device)
        )
        p_local2 = (
            torch.tensor([0.0, -0.0004, 0.009])
            .to(self.problem.joint_angles.dtype)
            .to(self.problem.model.device)
        )
        p_img1 = get_img_coords(
            p_local1,
            R_list[2],
            t_list[2],
            pose_matrix.squeeze().to(self.problem.joint_angles.dtype),
            intr,
        )
        p_img2 = get_img_coords(
            p_local2,
            R_list[3],
            t_list[3],
            pose_matrix.squeeze().to(self.problem.joint_angles.dtype),
            intr,
        )
        proj_pts = torch.stack([p_img1, p_img2], dim=0)

        # Overlay the rendered mask with the reference mask
        det_line_params = self.problem.det_line_params
        overlay = self._overlay_masks(
            rendered_np,
            ref_np,
            ref_pts,
            proj_pts,
            det_line_params,
            projected_lines_params=None,
        )

        # draw blue box around the rendered mask
        x, y, w, h = self.problem.single_mask_to_box(rendered_image.squeeze(0))
        # bbox_coord = self.problem.batch_mask_to_box(self.problem.ref_mask.unsqueeze(0))
        # print(bbox_coord)
        overlay = cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (1, 0, 0),
            2,
        )

        # draw yellow box around the reference mask
        x, y, w, h = self.problem.single_mask_to_box(self.problem.ref_mask)
        overlay = cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (0, 1, 1),
            2,
        )

        save_path = os.path.join(self.save_fig_dir, f"{self._steps_count}.png")
        self.save_image_cv(overlay, save_path)


def cylinder_loss_params(detected_lines, projected_lines):
    """Input:
    detected_lines [B, 2, 2]
    projected_lines [B, 2, 2]

    Output:
    loss [B]
    """

    def to_theta_rho(lines):
        # lines: [B, 2, 2], so:
        #   lines[:, 0, :] -> line0 for each batch -> shape [B, 2]
        #   lines[:, 1, :] -> line1 for each batch -> shape [B, 2]
        a = lines[..., 0]  # shape [B, 2]
        b = lines[..., 1]  # shape [B, 2]
        n = (a**2 + b**2).sqrt() + 1e-9  # shape [B, 2]
        rho = 1.0 / n  # shape [B, 2]
        theta = torch.atan2(b, a)  # shape [B, 2], range [-pi, +pi]
        return theta, rho

    # 2) Per-line difference:  (theta1, rho1) vs. (theta2, rho2)
    #    Each is shape [B], returning shape [B].
    def line_difference(theta1, rho1, theta2, rho2):
        # Each input is [B], because we’ll call this once per line pairing.
        delta_theta = torch.abs(theta1 - theta2)  # [B]
        delta_theta = torch.min(
            delta_theta, 2 * math.pi - delta_theta
        )  # handle 2π periodicity
        delta_theta = torch.min(
            delta_theta, math.pi - delta_theta
        )  # optional, if you want symmetrical ranges
        delta_rho = torch.abs(rho1 - rho2)  # [B]
        # Return elementwise line distance
        # (Originally you did a mean, but now we keep it per-batch-sample.)
        dist = delta_rho + 0.7 * delta_theta  # [B]
        return dist, delta_theta

    # 3) Extract batched theta, rho for detected and projected lines
    theta_det, rho_det = to_theta_rho(detected_lines)  # each => shape [B, 2]
    theta_proj, rho_proj = to_theta_rho(projected_lines)  # each => shape [B, 2]

    # 4) Pairing 1: Det[0] ↔ Proj[0], Det[1] ↔ Proj[1]
    #    We'll index line 0: (theta_det[:, 0], rho_det[:, 0]) vs. (theta_proj[:, 0], rho_proj[:, 0])
    loss_1_0, theta_1_0 = line_difference(
        theta_det[:, 0], rho_det[:, 0], theta_proj[:, 0], rho_proj[:, 0]
    )  # each => [B]
    loss_1_1, theta_1_1 = line_difference(
        theta_det[:, 1], rho_det[:, 1], theta_proj[:, 1], rho_proj[:, 1]
    )  # each => [B]
    total_loss_1 = loss_1_0 + loss_1_1  # shape [B]

    # 5) Pairing 2: Det[0] ↔ Proj[1], Det[1] ↔ Proj[0]
    loss_2_0, theta_2_0 = line_difference(
        theta_det[:, 0], rho_det[:, 0], theta_proj[:, 1], rho_proj[:, 1]
    )
    loss_2_1, theta_2_1 = line_difference(
        theta_det[:, 1], rho_det[:, 1], theta_proj[:, 0], rho_proj[:, 0]
    )
    total_loss_2 = loss_2_0 + loss_2_1  # shape [B]

    # 6) Centerline alignment
    #    We take the mean over the lines dimension = 1, so each batch item has a single (theta, rho) average.
    theta_det_mean = torch.mean(theta_det, dim=1)  # shape [B]
    rho_det_mean = torch.mean(rho_det, dim=1)  # shape [B]
    theta_proj_mean = torch.mean(theta_proj, dim=1)
    rho_proj_mean = torch.mean(rho_proj, dim=1)

    centerline_loss, _ = line_difference(
        theta_det_mean, rho_det_mean, theta_proj_mean, rho_proj_mean
    )  # shape [B]

    # 7) Choose minimal pairing for each batch element, then add centerline loss
    #    total_loss_1, total_loss_2, and centerline_loss are all shape [B].
    line_loss = torch.where(total_loss_1 < total_loss_2, total_loss_1, total_loss_2)
    line_loss = line_loss + centerline_loss

    # line_loss is now [B].
    return line_loss  # [B]


# torch.autograd.set_detect_anomaly(True)
def keypoint_loss_batch(keypoints_a, keypoints_b):
    """
    Computes the Chamfer distance between two sets of keypoints.

    Args:
        keypoints_a (torch.Tensor): Tensor of keypoints (shape: [B, 2, 2]).
        keypoints_b (torch.Tensor): Tensor of keypoints (shape: [B, 2, 2]).

    Returns:
        torch.Tensor: The computed Chamfer distance.
    """
    if keypoints_a.size(1) != 2 or keypoints_b.size(1) != 2:
        raise ValueError("This function assumes two keypoints per set in each batch.")

    # Permutation 1: A0->B0 and A1->B1
    dist_1 = torch.norm(keypoints_a[:, 0] - keypoints_b[:, 0], dim=1) + torch.norm(
        keypoints_a[:, 1] - keypoints_b[:, 1], dim=1
    )

    # Permutation 2: A0->B1 and A1->B0
    dist_2 = torch.norm(keypoints_a[:, 0] - keypoints_b[:, 1], dim=1) + torch.norm(
        keypoints_a[:, 1] - keypoints_b[:, 0], dim=1
    )

    # Choose the pairing that results in minimal distance for each batch
    min_dist = torch.min(dist_1, dist_2)  # [B]

    # Align the centerline for each batch
    centerline_loss = torch.norm(
        torch.mean(keypoints_a, dim=1) - torch.mean(keypoints_b, dim=1), dim=1
    )  # [B]

    return min_dist + centerline_loss


def compute_distance_map(ref_mask, gamma):
    ref_mask_np = ref_mask.detach().cpu().numpy().astype(np.float32)
    distance_map = skfmm.distance(ref_mask_np == 0) / gamma
    distance_map[ref_mask_np == 1] = 0
    return torch.from_numpy(distance_map).float().to(ref_mask.device)


def distance_loss(pred_mask, distance_map_ref):
    """input: both B, H, W"""
    # Compute distance map from reference to predicted
    loss_ref_to_pred = torch.sum(pred_mask * distance_map_ref, dim=(1, 2))
    total_loss = loss_ref_to_pred
    return total_loss


def appearance_loss(pred_mask, ref_mask):
    """input: both B, H, W"""

    sum_pred = torch.sum(pred_mask, dim=(1, 2))
    sum_ref = torch.sum(ref_mask, dim=(1, 2))
    return torch.abs(sum_pred - sum_ref)  # (B)


def convert_line_params_to_endpoints(a, b, image_width, image_height):
    """
    Convert line parameters (a, b) in the form au + bv = 1 to endpoints (x1, y1), (x2, y2).

    Args:
        a (float): Parameter 'a' from the line equation.
        b (float): Parameter 'b' from the line equation.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tuple: Endpoints (x1, y1), (x2, y2) for drawing the line.
    """
    # Calculate endpoints by choosing boundary values for 'u'
    if b != 0:
        # Set u = 0 to find y-intercept
        x1 = 0
        y1 = int((1 - a * x1) / b)

        # Set u = image_width to find corresponding 'v'
        x2 = image_width
        y2 = int((1 - a * x2) / b)
    else:
        # Vertical line: set v based on boundaries
        y1 = 0
        y2 = image_height
        x1 = x2 = int(1 / a)

    return (x1, y1), (x2, y2)


class PoseEstimationProblem(Problem):
    def __init__(
        self, model, robot_mesh, robot_renderer, ref_mask_file, ref_keypoints, 
        batch_size, joint_angles, fx, fy, px, py, ld1=3, ld2=3, ld3=3,
    ):
        super().__init__(
            objective_sense="min",
            solution_length=10, 
            device=model.device,
        )

        self.model = model
        self.robot_mesh = robot_mesh
        self.robot_renderer = robot_renderer
        self.ref_keypoints = ref_keypoints
        self.batch_size = batch_size
        self.joint_angles = joint_angles

        self.fx, self.fy, self.px, self.py = fx, fy, px, py

        # Camera intrinsic matrix
        self.intr = torch.tensor(
            [[self.fx, 0, self.px], [0, self.fy, self.py], [0, 0, 1]],
            device="cuda",
            dtype=self.joint_angles.dtype,
        )

        self.p_local1 = (
            torch.tensor([0.0, 0.0004, 0.009])
            .to(self.joint_angles.dtype)
            .to(self.model.device)
        )
        self.p_local2 = (
            torch.tensor([0.0, -0.0004, 0.009])
            .to(self.joint_angles.dtype)
            .to(self.model.device)
        )

        self.det_line_params = None

        # normalize
        ld1, ld2, ld3 = np.array([ld1, ld2, ld3]) / np.linalg.norm(np.array([ld1, ld2, ld3]))
        self.ld1, self.ld2, self.ld3 = ld1, ld2, ld3

        self.readRefImage(ref_mask_file)

    def compute_weighting_mask(self, shape, center_weight=1.0, edge_weight=0.5):
        """
        Copied from your single-sample code: creates a weighting mask for the MSE.
        shape: (H,W)
        """
        h, w = shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        max_distance = np.sqrt(center_y**2 + center_x**2)
        normalized_distance = distance / max_distance
        weights = edge_weight + (center_weight - edge_weight) * (
            1 - normalized_distance
        )
        self.weighting_mask = torch.from_numpy(weights).float().to(self.model.device)

    def single_mask_to_box(self, mask):
        y, x = torch.nonzero(mask, as_tuple=True)
        if len(x) == 0 or len(y) == 0:
            return 0, 0, 0, 0
        x_min, x_max = torch.min(x), torch.max(x)
        y_min, y_max = torch.min(y), torch.max(y)
        w = x_max - x_min
        h = y_max - y_min
        return x_min.item(), y_min.item(), w.item(), h.item()

    def readRefImage(self, ref_mask):
        cv_img = cv2.imread(ref_mask)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = cv2.inRange(cv_img, np.ones(3) * 128, np.ones(3) * 255) / 255.0
        self.ref_mask = torch.Tensor(img).to(self.model.device)  # [480, 640]
        self.ref_mask_b = self.ref_mask.unsqueeze(0).expand(
            self.batch_size, self.ref_mask.shape[0], self.ref_mask.shape[1]
        )  # [B, H, W]
        self.compute_weighting_mask(self.ref_mask.shape)
        self.distance_map = compute_distance_map(self.ref_mask, gamma=1)
        self.ref_bbox_x, self.ref_bbox_y, self.ref_bbox_w, self.ref_bbox_h = self.single_mask_to_box(self.ref_mask)
        # align with the result of batch_mask_to_box
        self.ref_bbox_coord = torch.tensor(
            [
                [self.ref_bbox_y, self.ref_bbox_x],
                [self.ref_bbox_y, self.ref_bbox_x + self.ref_bbox_w],
                [self.ref_bbox_y + self.ref_bbox_h, self.ref_bbox_x],
                [self.ref_bbox_y + self.ref_bbox_h, self.ref_bbox_x + self.ref_bbox_w],
            ]
        ).to(self.model.device).to(torch.float32)
        print(f"ref_bbox_coord: {self.ref_bbox_coord}")
    
    def _fill(self, values: torch.Tensor):
        raise NotImplementedError("Must initialize the problem with a solution.")
    
    def cylinder_loss_batch(
        self, position, direction, pose_matrix, radius
    ):
        # get the projected cylinder lines parameters
        ref_mask = self.ref_mask
        intr = torch.tensor(
            [[self.fx, 0, self.px], [0, self.fy, self.py], [0, 0, 1]],
            device=self.model.device,
            dtype=torch.float32,
        )

        _, cam_pts_3d_position = transform_points_b(position, pose_matrix, intr)
        _, cam_pts_3d_norm = transform_points_b(direction, pose_matrix, intr)
        cam_pts_3d_norm = torch.nn.functional.normalize(
            cam_pts_3d_norm
        )  # NORMALIZE !!!!!!

        # print(f"checking shape of cylinder input: {cam_pts_3d_position.shape, cam_pts_3d_norm.shape}")  both [B,3]
        e_1, e_2 = projectCylinderTorch(
            cam_pts_3d_position, cam_pts_3d_norm, radius, self.fx, self.fy, self.px, self.py
        )  # [B,2], [B,2]
        projected_lines = torch.stack((e_1, e_2), dim=1)  # [B, 2, 2]

        # get the detected reference lines parameters
        if self.det_line_params == None:
            ref_mask_np = ref_mask.detach().cpu().numpy()
            longest_lines = detect_lines(ref_mask_np, output=True)

            longest_lines = np.array(longest_lines, dtype=np.float64)

            if longest_lines.shape[0] < 2:
                # Force skip cylinder or fallback
                print(
                    "WARNING: Not enough lines found by Hough transform. Skipping cylinder loss."
                )
                # You can set self.det_line_params to None or some fallback
                self.det_line_params = None

            else:
                # print(f"debugging the longest lines {longest_lines}")
                x1 = longest_lines[:, 0]
                y1 = longest_lines[:, 1]
                x2 = longest_lines[:, 2]
                y2 = longest_lines[:, 3]
                # print(f"debugging the end points x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
                # Calculate line parameters (a, b, c) for detected lines
                a = y2 - y1
                b = x1 - x2
                c = x1 * y2 - x2 * y1  # Determinant for the line equation

                # Normalize to match the form au + bv = 1
                # norm = c + 1e-6  # Ensure no division by zero
                norm = np.abs(c)  # Compute the absolute value
                norm = np.maximum(norm, 1e-6)  # Clamp to a minimum value of 1e-6
                a /= norm
                b /= norm

                # Stack line parameters into a tensor and normalize to match au + bv = 1 form
                detected_lines = torch.from_numpy(np.stack((a, b), axis=-1)).to(
                    self.model.device
                )
                self.det_line_params = detected_lines

        # print(f"debugging the line params: {detected_lines}")
        self.proj_line_params = projected_lines  # [B, 2, 2]
        detected_lines = self.det_line_params
        B = self.batch_size
        detected_lines = detected_lines.unsqueeze(0).expand(
            B, detected_lines.shape[0], detected_lines.shape[1]
        )  # [B, 2, 2]
        cylinder_loss = cylinder_loss_params(detected_lines, projected_lines)  # [B]
        return cylinder_loss

    def batch_mask_to_box(self, masks):
        """
        Convert [B,H,W] binary masks to [B,4,2] bounding-box corners.
        Empty masks yield all-zero boxes.
        """
        B, H, W = masks.shape
        m = masks.bool()  # ensure boolean
        # Check which rows/cols have any True
        rows_any = m.any(dim=2)     # [B, H]
        cols_any = m.any(dim=1)     # [B, W]
        # First and last row indices with True
        first_row = rows_any.int().argmax(dim=1)   # [B]
        last_row  = (H - 1) - rows_any.flip(dims=[1]).int().argmax(dim=1)
        # First and last col indices with True
        first_col = cols_any.int().argmax(dim=1)   # [B]
        last_col  = (W - 1) - cols_any.flip(dims=[1]).int().argmax(dim=1)
        # Stack into corner coordinates
        top_left     = torch.stack([first_row, first_col], dim=1)  # [B,2]
        top_right    = torch.stack([first_row, last_col],  dim=1)  # [B,2]
        bottom_left  = torch.stack([last_row,  first_col], dim=1)  # [B,2]
        bottom_right = torch.stack([last_row,  last_col],  dim=1)  # [B,2]
        boxes = torch.stack([top_left, top_right, bottom_left, bottom_right], dim=1)  # [B,4,2]
        # Zero out empty-mask boxes
        nonempty = m.any(dim=(1,2))  # [B]
        boxes[~nonempty] = 0
        return boxes
    
    def batch_loss(
        self,
        joint_angles, # shape (B, 4)
        pred_mask_2d,  # shape (B, H, W)
        ref_mask_2d,  # shape (B, H, W)
        cTr_b,  # shape (B, 6)
        ref_kps_2d,  # shape (2, 2), but will be broadcasted
    ):
        B = pred_mask_2d.shape[0]
        # weighting mask
        weighting_mask = self.weighting_mask
        weighting_mask = weighting_mask.unsqueeze(0).expand(
            B, *[-1 for _ in weighting_mask.shape]
        )

        # MSE
        mse_val = F.mse_loss(
            pred_mask_2d * weighting_mask,
            ref_mask_2d * weighting_mask,
            reduction="none",
        )  # [B, H, W]
        mse_val = mse_val.mean(dim=(1, 2))  # should be [B]

        use_aux_render_loss = True
        use_geometry_loss = True
        use_cv_loss = False

        if use_aux_render_loss:
            # distance loss
            distance_map_ref = self.distance_map.unsqueeze(0).expand(
                B, *[-1 for _ in self.distance_map.shape]
            )  # [B, H, W]

            dist_val = distance_loss(
                pred_mask_2d * weighting_mask, distance_map_ref * weighting_mask
            )  # [B]

            # appearance
            app_val = appearance_loss(pred_mask_2d, ref_mask_2d)  # [B]
        else:
            dist_val = 0.0
            app_val = 0.0

        if use_geometry_loss:
            # keypoint
            if ref_kps_2d is not None:

                pose_matrix_b = self.model.cTr_to_pose_matrix(cTr_b)  # [B, 4, 4]

                R_list, t_list = batch_lndFK(joint_angles)
                R_list = R_list.to(self.model.device) # [B, 4, 3, 3]
                t_list = t_list.to(self.model.device) # [B, 4, 3]
                
                p_img1 = get_img_coords_batch(
                    self.p_local1,
                    R_list[:,2,...],
                    t_list[:,2,...],
                    pose_matrix_b.squeeze().to(self.joint_angles.dtype),
                    self.intr,
                )
                p_img2 = get_img_coords_batch(
                    self.p_local2,
                    R_list[:,3,...],
                    t_list[:,3,...],
                    pose_matrix_b.squeeze().to(self.joint_angles.dtype),
                    self.intr,
                )
                # They are both B, 2

                proj_pts = torch.stack((p_img1, p_img2), dim=1)  # [B, 2, 2]

                ref_kps_2d = ref_kps_2d.unsqueeze(0).expand(
                    B, ref_kps_2d.shape[0], ref_kps_2d.shape[1]
                )  # [B, 2, 2]
                pts_val = keypoint_loss_batch(proj_pts, ref_kps_2d)  # [B]
            else:
                pts_val = torch.tensor(0.0, device=self.model.device)
                print("WARNING: No keypoints provided. Skipping keypoint loss.")

            # TODO Batchfy cylinder projection and computation, projection is done.

            position = torch.zeros((B, 3), dtype=torch.float32, device=self.model.device)  # (B, 3)
            direction = torch.zeros((B, 3), dtype=torch.float32, device=self.model.device)  # (B, 3)
            direction[:, 2] = 1.0
            pose_matrix_b = self.model.cTr_to_pose_matrix(cTr_b).squeeze(0)  # shape(B, 4, 4)
            radius = 0.0085 / 2

            # We'll do a small custom function "cylinder_loss_single" that returns (loss, angle).
            cylinder_val = self.cylinder_loss_batch(
                position,
                direction,
                pose_matrix_b,
                radius,
            )  # [B]

        else:
            pts_val = 0.0
            cylinder_val = 0.0

        if use_cv_loss:
            # compute the bbox loss
            pred_box_coord = self.batch_mask_to_box(pred_mask_2d).to(torch.float32)  # [B, 4, 2]
            ref_box_coord = self.ref_bbox_coord.unsqueeze(0).expand(
                B, self.ref_bbox_coord.shape[0], self.ref_bbox_coord.shape[1]
            )

            bbox_loss = torch.sum(
                torch.norm(pred_box_coord - ref_box_coord, dim=2), dim=1
            ) # [B]

        else:
            bbox_loss = 0.0

        # print(10 * self.ld1, self.ld2 * 1e-7, self.ld3 * 1e-5)
        if use_aux_render_loss or use_geometry_loss or use_cv_loss:
            item_loss = (
                10 * self.ld1 * mse_val
                + self.ld2 * (1e-7) * dist_val
                + self.ld3 * (1e-5) * app_val
                + 5 * (1e-3) * pts_val
                + 10 * (1e-3) * cylinder_val
                + 1e-7 * bbox_loss
            )
        else:
            item_loss = 10 * self.ld1 * mse_val

        item_loss_mask = torch.isnan(item_loss)
        if item_loss_mask.any():
            print(
                f"Found NaN in per_sample_loss for samples: {[mse_val], [dist_val], [app_val], [pts_val], [cylinder_val]}"
            )

        return item_loss  # [B]

    def _evaluate_batch(self, solutions: SolutionBatch):
        # import time
        # time_start = time.time()
        values = solutions.values.clone()
        cTr_batch = values[:, :6]  # shape (B, 6)
        joint_angles = values[:, 6:]  # shape (B, 4)
        B = cTr_batch.shape[0]
        if B != self.batch_size:
            self.batch_size = B
            self.ref_mask_b = self.ref_mask.unsqueeze(0).expand(
                B, self.ref_mask.shape[0], self.ref_mask.shape[1]
            )

        verts, faces = self.robot_renderer.batch_get_robot_verts_and_faces(joint_angles)

        pred_masks_b = self.model.render_robot_mask_batch_nvdiffrast(
            cTr_batch, verts, faces, self.robot_renderer
        )  # shape (B,H,W)
        ref_masks_b = self.ref_mask_b  # shape (B,H,W)

        per_sample_loss = self.batch_loss(
            joint_angles,
            pred_masks_b,
            ref_masks_b,
            cTr_batch,
            self.ref_keypoints,
        )  # [B]

        solutions.set_evals(per_sample_loss)


class BlackBoxOptimize:
    def __init__(
        self, model, robot_mesh, robot_renderer, ref_mask_file, ref_keypoints, joint_angles, 
        fx, fy, px, py, ld1, ld2, ld3, stdev_init=None, popsize=None, center_init=None, log_interval=5
    ):

        if torch.is_tensor(ref_keypoints):
            ref_keypoints = ref_keypoints.clone().detach().squeeze().to(model.device).float()
        else:
            ref_keypoints = torch.tensor(ref_keypoints).squeeze().to(model.device).float() # cannot be a list

        if not model.args.use_nvdiffrast or model.use_antialiasing:
            print("[Use NvDiffRast without antialiasing for black box optimization.]")
            model.args.use_nvdiffrast = True  # use NvDiffRast renderer
            model.use_antialiasing = False # do not use antialiasing as gradients are not needed

        # Initialize the problem
        self.problem = PoseEstimationProblem(
            model, robot_mesh, robot_renderer, ref_mask_file, ref_keypoints=ref_keypoints, batch_size=popsize if popsize is not None else 10, 
            joint_angles=joint_angles, fx=fx, fy=fy, px=px, py=py, ld1=ld1, ld2=ld2, ld3=ld3,
        )

        if stdev_init is None:
            stdev_init = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=torch.float32).cuda() # Initial standard deviation for XNES/SNES
            stdev_init[:3] *= 1e-1 # angles (3D)
            stdev_init[3:6] *= 1e-2 # translations (3D)
            stdev_init[6:] *= 1e-1 # joint angles (4D)

        # Initialize the searcher (currently, CMA-ES and XNES works)
        self.searcher = XNES(
            problem=self.problem,
            stdev_init=stdev_init,
            center_init=center_init,
        )

        self.log_interval = log_interval
        self.center_init = center_init

    # @torch.no_grad()
    def optimize(self, num_iters, register_logger=True):
        """
        Return final_cTr_s, final_loss_s, final_angle_s 
        """
        if num_iters <= 0:
            cTr = self.center_init[:6]
            angles = self.center_init[6:]
            return cTr, float("inf"), angles

        # Register a logger
        if register_logger:
            print("Starting optimization...")
            logger = ImageLogger(self.searcher, num_iters=num_iters, interval=self.log_interval, after_first_step=True)

        self.searcher.run(num_iters)

        if register_logger:
            best_solution = logger.best_solution
            best_eval = logger.best_eval

            final_cTr_s = best_solution[:6]
            final_loss_s = best_eval
            final_angle_s = best_solution[6:]

            logger.progress_bar.close()

            return final_cTr_s, final_loss_s, final_angle_s
    

class BayesOptBatchProblem:
    def __init__(
        self, model, robot_renderer, ref_mask_file, 
        ref_keypoints, fx, fy, px, py, batch_size,
        ld1, ld2, ld3, batch_iters, lr
    ):
        self.model = model
        self.robot_renderer = robot_renderer
        self.ref_keypoints = ref_keypoints
        self.device = model.device

        self.fx, self.fy, self.px, self.py = fx, fy, px, py

        self.batch_size = batch_size
        self.ld1, self.ld2, self.ld3 = ld1, ld2, ld3
        self.batch_iters = batch_iters
        self.lr = lr

        # Get ref_mask, weighting_mask, distance_map
        self.readRefImage(ref_mask_file)

        # Compute det_line_params
        ref_mask_np = self.ref_mask.detach().cpu().numpy()
        longest_lines = detect_lines(ref_mask_np, output=True)

        longest_lines = np.array(longest_lines, dtype=np.float64)

        if longest_lines.shape[0] < 2:
            # Force skip cylinder or fallback
            print(
                "WARNING: Not enough lines found by Hough transform. Skipping cylinder loss."
            )
            # You can set self.det_line_params to None or some fallback
            self.det_line_params = None

        else:
            # print(f"debugging the longest lines {longest_lines}")
            x1 = longest_lines[:, 0]
            y1 = longest_lines[:, 1]
            x2 = longest_lines[:, 2]
            y2 = longest_lines[:, 3]
            # print(f"debugging the end points x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
            # Calculate line parameters (a, b, c) for detected lines
            a = y2 - y1
            b = x1 - x2
            c = x1 * y2 - x2 * y1  # Determinant for the line equation

            # Normalize to match the form au + bv = 1
            # norm = c + 1e-6  # Ensure no division by zero
            norm = np.abs(c)  # Compute the absolute value
            norm = np.maximum(norm, 1e-6)  # Clamp to a minimum value of 1e-6
            a /= norm
            b /= norm

            # Stack line parameters into a tensor and normalize to match au + bv = 1 form
            detected_lines = th.from_numpy(np.stack((a, b), axis=-1)).to(
                self.model.device
            )
            self.det_line_params = detected_lines

        self.final_cTr_batch = None
        self.joint_angles_batch = None
        self.final_loss_batch = None

    def readRefImage(self, ref_mask):
        cv_img = cv2.imread(ref_mask)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = cv2.inRange(cv_img, np.ones(3) * 128, np.ones(3) * 255) / 255.0
        self.ref_mask = th.Tensor(img).to(self.model.device)  # [480, 640]
        self.ref_mask_b = self.ref_mask.unsqueeze(0).expand(
            self.batch_size, self.ref_mask.shape[0], self.ref_mask.shape[1]
        )  # [B, H, W]
        self.compute_weighting_mask(self.ref_mask.shape)
        self.distance_map = compute_distance_map(self.ref_mask, gamma=1)

    def compute_weighting_mask(self, shape, center_weight=1.0, edge_weight=0.5):
        """
        Copied from your single-sample code: creates a weighting mask for the MSE.
        shape: (H,W)
        """
        h, w = shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        max_distance = np.sqrt(center_y**2 + center_x**2)
        normalized_distance = distance / max_distance
        weights = edge_weight + (center_weight - edge_weight) * (
            1 - normalized_distance
        )
        self.weighting_mask = th.from_numpy(weights).float().to(self.device)

    def __call__(self, input):
        assert input.shape == (self.batch_size, 7), f"Input shape must be (B, 7), got {input.shape}"   

        # Convert to cTr and joint angles
        cTr_batch = th.zeros((self.batch_size, 6), dtype=th.float32).cuda()
        joint_angles_batch = th.zeros((self.batch_size, 4), dtype=th.float32).cuda()

        for i in range(self.batch_size):
            distance, elevation, camera_roll_local, camera_roll = input[i,:4]
            joint_angles = input[i,4:]

            distance = th.tensor(distance, dtype=torch.float32).unsqueeze(0).cuda()
            elevation = th.tensor(elevation, dtype=torch.float32).unsqueeze(0).cuda()
            camera_roll_local = th.tensor(camera_roll_local, dtype=torch.float32).unsqueeze(0).cuda()
            camera_roll = th.tensor(camera_roll, dtype=torch.float32).unsqueeze(0).cuda()

            pose_matrix = self.model.from_lookat_to_pose_matrix(
                distance, elevation, camera_roll_local
            )
            roll_rad = th.deg2rad(camera_roll)  # Convert roll angle to radians
            roll_matrix = th.tensor(
                [
                    [th.cos(roll_rad), -th.sin(roll_rad), 0],
                    [th.sin(roll_rad), th.cos(roll_rad), 0],
                    [0, 0, 1],
                ]
            )
            pose_matrix[:, :3, :3] = th.matmul(roll_matrix, pose_matrix[:, :3, :3])
            cTr = self.model.pose_matrix_to_cTr(pose_matrix)

            cTr_batch[i] = cTr.clone().squeeze(0)
            joint_angles_batch[i,:3] = th.tensor(joint_angles, dtype=th.float32).cuda()
            joint_angles_batch[i,3] = joint_angles_batch[i,2] # 4th joint angle = 3rd joint angle for initialization

        # Batch optimization
        optimizer = HeterogeneousBatchOptimize(
            cTr_batch=cTr_batch,
            joint_angles=joint_angles_batch,
            model=self.model,
            robot_renderer=self.robot_renderer,
            ref_keypoints=self.ref_keypoints,
            fx=self.fx,
            fy=self.fy,
            px=self.px,
            py=self.py,
            lr=self.lr,
            batch_size=self.batch_size,
            ref_mask=self.ref_mask,
            weighting_mask=self.weighting_mask,
            distance_map=self.distance_map,
            det_line_params=self.det_line_params,
        )

        final_cTr_batch, final_loss_batch = optimizer.optimize_batch(
            iterations=self.batch_iters, ld1=self.ld1, ld2=self.ld2, ld3=self.ld3
        )

        # Return numpy array of losses and maintain best final cTr and joint angles
        per_sample_loss = final_loss_batch.unsqueeze(1).detach().cpu().numpy()  # [B, 1]
        if self.final_cTr_batch is None:
            self.final_cTr_batch = final_cTr_batch
            self.joint_angles_batch = joint_angles_batch
            self.final_loss_batch = final_loss_batch.detach().clone()
        else:
            self.final_cTr_batch = th.cat((self.final_cTr_batch, final_cTr_batch), dim=0)
            self.joint_angles_batch = th.cat((self.joint_angles_batch, joint_angles_batch), dim=0)
            self.final_loss_batch = th.cat((self.final_loss_batch, final_loss_batch.detach().clone()), dim=0)
        
        return per_sample_loss
        
        