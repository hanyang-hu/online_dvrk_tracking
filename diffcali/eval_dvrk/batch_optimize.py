import cv2
import numpy as np
import torch as th
from tqdm import tqdm
import skfmm
import torch.nn.functional as F
import math

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from diffcali.eval_dvrk.LND_fk import lndFK, batch_lndFK
from diffcali.utils.projection_utils import *
from diffcali.utils.cylinder_projection_utils import (
    projectCylinderTorch,
    transform_points_b,
)
from diffcali.utils.detection_utils import detect_lines

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

th.set_default_dtype(th.float32)


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
        theta = th.atan2(b, a)  # shape [B, 2], range [-pi, +pi]
        return theta, rho

    # 2) Per-line difference:  (theta1, rho1) vs. (theta2, rho2)
    #    Each is shape [B], returning shape [B].
    def line_difference(theta1, rho1, theta2, rho2):
        # Each input is [B], because we’ll call this once per line pairing.
        delta_theta = th.abs(theta1 - theta2)  # [B]
        delta_theta = th.min(
            delta_theta, 2 * math.pi - delta_theta
        )  # handle 2π periodicity
        delta_theta = th.min(
            delta_theta, math.pi - delta_theta
        )  # optional, if you want symmetrical ranges
        delta_rho = th.abs(rho1 - rho2)  # [B]
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
    theta_det_mean = th.mean(theta_det, dim=1)  # shape [B]
    rho_det_mean = th.mean(rho_det, dim=1)  # shape [B]
    theta_proj_mean = th.mean(theta_proj, dim=1)
    rho_proj_mean = th.mean(rho_proj, dim=1)

    centerline_loss, _ = line_difference(
        theta_det_mean, rho_det_mean, theta_proj_mean, rho_proj_mean
    )  # shape [B]

    # 7) Choose minimal pairing for each batch element, then add centerline loss
    #    total_loss_1, total_loss_2, and centerline_loss are all shape [B].
    line_loss = th.where(total_loss_1 < total_loss_2, total_loss_1, total_loss_2)
    line_loss = line_loss + centerline_loss

    # line_loss is now [B].
    return line_loss  # [B]


# th.autograd.set_detect_anomaly(True)
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
    dist_1 = th.norm(keypoints_a[:, 0] - keypoints_b[:, 0], dim=1) + th.norm(
        keypoints_a[:, 1] - keypoints_b[:, 1], dim=1
    )

    # Permutation 2: A0->B1 and A1->B0
    dist_2 = th.norm(keypoints_a[:, 0] - keypoints_b[:, 1], dim=1) + th.norm(
        keypoints_a[:, 1] - keypoints_b[:, 0], dim=1
    )

    # Choose the pairing that results in minimal distance for each batch
    min_dist = th.min(dist_1, dist_2)  # [B]

    # Align the centerline for each batch
    centerline_loss = th.norm(
        th.mean(keypoints_a, dim=1) - th.mean(keypoints_b, dim=1), dim=1
    )  # [B]

    return min_dist + centerline_loss


def compute_distance_map(ref_mask, gamma):
    ref_mask_np = ref_mask.detach().cpu().numpy().astype(np.float32)
    distance_map = skfmm.distance(ref_mask_np == 0) / gamma
    distance_map[ref_mask_np == 1] = 0
    return th.from_numpy(distance_map).float().to(ref_mask.device)


""" This should be bidirectional, since otherwise the loss would be zero if one fully convers the other one"""
# def distance_loss(pred_mask, ref_mask, gamma):
#     distance_map = compute_distance_map(ref_mask, gamma)
#     return th.sum(pred_mask * distance_map)


def distance_loss(pred_mask, distance_map_ref):
    """input: both B, H, W"""
    # Compute distance map from reference to predicted
    loss_ref_to_pred = th.sum(pred_mask * distance_map_ref, dim=(1, 2))
    total_loss = loss_ref_to_pred
    return total_loss


def appearance_loss(pred_mask, ref_mask):
    """input: both B, H, W"""

    sum_pred = th.sum(pred_mask, dim=(1, 2))
    sum_ref = th.sum(ref_mask, dim=(1, 2))
    return th.abs(sum_pred - sum_ref)  # (B)


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


class HeterogeneousBatchOptimize:
    def __init__(
        self,
        cTr_batch,  # (B,6) angle-axis+translation
        joint_angles,  # (B,4) 
        model,
        robot_renderer,
        ref_keypoints=None,  # shape np array (B,2,2) or (2,2)
        fx=None,
        fy=None,
        px=None,
        py=None,
        lr=1e-4,
        batch_size=None,
        ref_mask=None, 
        weighting_mask=None,
        distance_map=None,
        det_line_params=None,
    ):
        """
        If ref_keypoints is the same for all items, store shape (2,2). If each item has separate keypoints, store (B,2,2).
        """
        self.model = model
        self.robot_renderer = robot_renderer
        self.device = model.device

        # Store cTr_batch as a trainable parameter
        self.cTr_batch = th.nn.Parameter(cTr_batch.clone().detach()).to(
            self.device
        )  # shape (B,6)
        self.optimizer = th.optim.Adam([self.cTr_batch], lr=lr)

        self.joint_angles = joint_angles  # (B,4)

        self.distance_map = None  # Ref only
        self.weighting_mask = None
        self.batch_size = batch_size
        # Convert reference mask to shape (H,W) => store it in float
        # If you have a single reference for the entire batch, we just broadcast later
        # If you want separate references, store shape (B,H,W).
        self.ref_mask_b = None
        if th.is_tensor(ref_keypoints):
            self.ref_keypoints = ref_keypoints.clone().squeeze().to(self.device).float()
        else:
            self.ref_keypoints = th.tensor(ref_keypoints).squeeze().to(self.device).float()
        self.fx = fx
        self.fy = fy
        self.px = px
        self.py = py
        self.det_line_params = None
        self.line_image = None  # just for visualize
        self.longest_lines = None  # just for visualize
        # We'll define constants from your single-sample code
        # (like "10, 0.7" etc) inside a method or pass them as arguments
        self.gamma = 1

        self.ref_mask = ref_mask
        self.ref_mask_b = self.ref_mask.unsqueeze(0).expand(
            self.batch_size, self.ref_mask.shape[0], self.ref_mask.shape[1]
        )  # [B, H, W]
        self.weighting_mask = weighting_mask
        self.distance_map = distance_map

        # Pre-compute intrinsic matrix and forward kinematics
        self.intr = th.tensor(
            [[self.fx, 0, self.px], [0, self.fy, self.py], [0, 0, 1]],
            device="cuda",
            dtype=self.joint_angles.dtype,
        )

        self.verts, self.faces = self.robot_renderer.batch_get_robot_verts_and_faces(joint_angles)

        R_list, t_list = batch_lndFK(joint_angles)
        self.R_list = R_list.to(self.model.device) # [B, 4, 3, 3]
        self.t_list = t_list.to(self.model.device) # [B, 4, 3]
        self.p_local1 = (
            th.tensor([0.0, 0.0004, 0.009])
            .to(self.joint_angles.dtype)
            .to(self.model.device)
        )
        self.p_local2 = (
            th.tensor([0.0, -0.0004, 0.009])
            .to(self.joint_angles.dtype)
            .to(self.model.device)
        )

        self.det_line_params = det_line_params

    # @th.compile
    def batch_loss(
        self,
        pred_mask_2d,  # shape (B, H, W)
        ref_mask_2d,  # shape (B, H, W)
        cTr_b,  # shape (B, 6)
        ref_kps_2d,  # shape (2, 2), but will be broadcasted
        ld1=3,
        ld2=3,
        ld3=3,
    ):
        B = pred_mask_2d.shape[0]
        # weighting mask
        weighting_mask = self.weighting_mask
        weighting_mask = weighting_mask.unsqueeze(0).expand(
            B, *[-1 for _ in weighting_mask.shape]
        )
        distance_map_ref = self.distance_map.unsqueeze(0).expand(
            B, *[-1 for _ in self.distance_map.shape]
        )  # [B, H, W]

        # MSE
        mse_val = F.mse_loss(
            pred_mask_2d * weighting_mask,
            ref_mask_2d * weighting_mask,
            reduction="none",
        )  # [B, H, W]
        mse_val = mse_val.mean(dim=(1, 2))  # should be [B]
        # print(f"debugging mse scales: {mse_val}")
        # distance loss
        dist_val = distance_loss(
            pred_mask_2d * weighting_mask, distance_map_ref * weighting_mask
        )  # [B]

        # appearance
        app_val = appearance_loss(pred_mask_2d, ref_mask_2d)  # [B]

        # keypoint
        if ref_kps_2d is not None:
            intr = th.tensor(
                [[self.fx, 0, self.px], [0, self.fy, self.py], [0, 0, 1]],
                device="cuda",
                dtype=self.joint_angles.dtype,
            )

            pose_matrix_b = self.model.cTr_to_pose_matrix(cTr_b)  # [B, 4, 4]

            p_img1 = get_img_coords_batch(
                self.p_local1,
                self.R_list[:,2,...],
                self.t_list[:,2,...],
                pose_matrix_b.squeeze().to(self.joint_angles.dtype),
                self.intr,
            )
            p_img2 = get_img_coords_batch(
                self.p_local2,
                self.R_list[:,3,...],
                self.t_list[:,3,...],
                pose_matrix_b.squeeze().to(self.joint_angles.dtype),
                self.intr,
            )
            # They are both B, 2

            proj_pts = th.stack((p_img1, p_img2), dim=1)  # [B, 2, 2]

            ref_kps_2d = ref_kps_2d.unsqueeze(0).expand(
                B, ref_kps_2d.shape[0], ref_kps_2d.shape[1]
            )  # [B, 2, 2]
            pts_val = keypoint_loss_batch(proj_pts, ref_kps_2d)  # [B]
        else:
            pts_val = th.tensor(0.0, device=self.device)

        position = th.zeros((B, 3), dtype=th.float32, device=self.device)  # (B, 3)
        direction = th.zeros((B, 3), dtype=th.float32, device=self.device)  # (B, 3)
        direction[:, 2] = 1.0
        pose_matrix_b = self.model.cTr_to_pose_matrix(cTr_b).squeeze(
            0
        )  # shape(B, 4, 4)
        radius = 0.0085 / 2

        # We'll do a small custom function "cylinder_loss_single" that returns (loss, angle).
        cylinder_val = self.cylinder_loss_batch(
            position,
            direction,
            pose_matrix_b,
            radius,
            self.fx,
            self.fy,
            self.px,
            self.py,
        )  # [B]

        # normalize
        ld1, ld2, ld3 = np.array([ld1, ld2, ld3]) / np.linalg.norm(
            np.array([ld1, ld2, ld3])
        )

        item_loss = (
            100 * (
                10 * ld1 * mse_val
                + ld2 * (1e-7) * dist_val
                + ld3 * (1e-5) * app_val
            )
            + 5 * (1e-3) * pts_val
            + 10 * (1e-3) * cylinder_val
        )
        item_loss_mask = th.isnan(item_loss)
        if item_loss_mask.any():
            print(
                f"Found NaN in per_sample_loss for samples: {[mse_val], [dist_val], [app_val], [pts_val], [cylinder_val]}"
            )

        return item_loss  # [B]

    # @th.compile
    def cylinder_loss_batch(
        self, position, direction, pose_matrix, radius, fx, fy, px, py
    ):
        """
        Batchfied Input:
        position [B ,3]
        direction [B ,3]
        pose_matrix [B, 4, 4]
        radius  [1]

        Batchfied Output:
        Cylinder loss [B,]
        """

        # get the projected cylinder lines parameters
        ref_mask = self.ref_mask
        intr = self.intr

        _, cam_pts_3d_position = transform_points_b(position, pose_matrix, intr)
        _, cam_pts_3d_norm = transform_points_b(direction, pose_matrix, intr)
        cam_pts_3d_norm = th.nn.functional.normalize(
            cam_pts_3d_norm
        )  # NORMALIZE !!!!!!

        # print(f"checking shape of cylinder input: {cam_pts_3d_position.shape, cam_pts_3d_norm.shape}")  both [B,3]
        e_1, e_2 = projectCylinderTorch(
            cam_pts_3d_position, cam_pts_3d_norm, radius, fx, fy, px, py
        )  # [B,2], [B,2]
        projected_lines = th.stack((e_1, e_2), dim=1)  # [B, 2, 2]

        # print(f"debugging the line params: {detected_lines}")
        self.proj_line_params = projected_lines  # [B, 2, 2]
        detected_lines = self.det_line_params
        B = self.batch_size

        if detected_lines is None:
            # If no detected lines, return zero loss
            return 0.0

        detected_lines = detected_lines.unsqueeze(0).expand(
            B, detected_lines.shape[0], detected_lines.shape[1]
        )  # [B, 2, 2]
        # print(f"checking the shape of detected and projected lines {detected_lines}, {projected_lines}")  # [2, 2] [2, 2]
        cylinder_loss = cylinder_loss_params(detected_lines, projected_lines)  # [B]
        # parallelism = angle_between_lines(projected_lines, detected_lines)
        # print(f"testing line angle...{parallelism}")
        return cylinder_loss

    def optimize_batch(
        self,
        iterations=300,
        grid_search=False,
        ld1=3,
        ld2=3,
        ld3=3,
    ):
        """
        Runs multiple optimization steps on cTr_batch.
        - We do 1 forward/backward pass per iteration.
        - Each pass sums (or averages) the single-item loss over B => single scalar.

        Returns:
          final_cTr: shape (B,6)
          final_losses: shape (B,) each sample's final loss after last iteration
        """

        for _ in tqdm(range(iterations), desc="optimizing batch samples...."):
            self.optimizer.zero_grad()

            pred_masks_b = self.model.render_robot_mask_batch_nvdiffrast(
                self.cTr_batch, self.verts, self.faces, self.robot_renderer
            )  # shape (B,H,W)

            ref_masks_b = self.ref_mask_b  # shape (B,H,W)
            per_sample_loss = self.batch_loss(
                pred_masks_b,
                ref_masks_b,
                self.cTr_batch,
                self.ref_keypoints,
                ld1,
                ld2,
                ld3,
            )  # [B]

            total_loss = per_sample_loss.mean()
            total_loss.backward()
            # th.nn.utils.clip_grad_norm_([self.cTr_batch], max_norm = 1)
            self.optimizer.step()

        # Compute the final per-sample losses
        with th.no_grad():
            pred_masks_b = self.model.render_robot_mask_batch_nvdiffrast(
                self.cTr_batch, self.verts, self.faces, self.robot_renderer
            )  # shape (B,H,W)

            final_losses = self.batch_loss(
                pred_masks_b,
                self.ref_mask_b,
                self.cTr_batch,
                self.ref_keypoints,
                ld1,
                ld2,
                ld3,
            )  # shape [B]

        return self.cTr_batch.clone().detach(), final_losses


class BatchOptimize:
    def __init__(
        self,
        cTr_batch,  # (B,6) angle-axis+translation
        joint_angles,  # shape (4) or (B,4) if each sample has different joint angles
        model,
        robot_mesh,
        robot_renderer,
        ref_keypoints=None,  # shape np array (B,2,2) or (2,2)
        fx=None,
        fy=None,
        px=None,
        py=None,
        lr=1e-4,
        batch_size=None,
    ):
        """
        If ref_keypoints is the same for all items, store shape (2,2). If each item has separate keypoints, store (B,2,2).
        """
        self.model = model
        self.robot_mesh = robot_mesh
        self.robot_renderer = robot_renderer
        self.device = model.device

        # Store cTr_batch as a trainable parameter
        self.cTr_batch = th.nn.Parameter(cTr_batch.clone().detach()).to(
            self.device
        )  # shape (B,6)
        self.optimizer = th.optim.Adam([self.cTr_batch], lr=lr)

        self.joint_angles = joint_angles  # shape (4) or (B,4)
        self.ref_mask = None

        self.distance_map = None  # Ref only
        self.weighting_mask = None
        self.batch_size = batch_size
        # Convert reference mask to shape (H,W) => store it in float
        # If you have a single reference for the entire batch, we just broadcast later
        # If you want separate references, store shape (B,H,W).
        self.ref_mask_b = None
        if th.is_tensor(ref_keypoints):
            self.ref_keypoints = ref_keypoints.clone().squeeze().to(self.device).float()
        else:
            self.ref_keypoints = th.tensor(ref_keypoints).squeeze().to(self.device).float()
        self.fx = fx
        self.fy = fy
        self.px = px
        self.py = py
        self.det_line_params = None
        self.line_image = None  # just for visualize
        self.longest_lines = None  # just for visualize
        # We'll define constants from your single-sample code
        # (like "10, 0.7" etc) inside a method or pass them as arguments
        self.gamma = 1

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

    # @th.compile
    def batch_loss(
        self,
        pred_mask_2d,  # shape (B, H, W)
        ref_mask_2d,  # shape (B, H, W)
        cTr_b,  # shape (B, 6)
        ref_kps_2d,  # shape (2, 2), but will be broadcasted
        ld1=3,
        ld2=3,
        ld3=3,
    ):
        B = pred_mask_2d.shape[0]
        # weighting mask
        weighting_mask = self.weighting_mask
        weighting_mask = weighting_mask.unsqueeze(0).expand(
            B, *[-1 for _ in weighting_mask.shape]
        )
        distance_map_ref = self.distance_map.unsqueeze(0).expand(
            B, *[-1 for _ in self.distance_map.shape]
        )  # [B, H, W]

        # MSE
        mse_val = F.mse_loss(
            pred_mask_2d * weighting_mask,
            ref_mask_2d * weighting_mask,
            reduction="none",
        )  # [B, H, W]
        mse_val = mse_val.mean(dim=(1, 2))  # should be [B]
        # print(f"debugging mse scales: {mse_val}")
        # distance loss
        dist_val = distance_loss(
            pred_mask_2d * weighting_mask, distance_map_ref * weighting_mask
        )  # [B]

        # appearance
        app_val = appearance_loss(pred_mask_2d, ref_mask_2d)  # [B]

        # keypoint
        if ref_kps_2d is not None:
            intr = th.tensor(
                [[self.fx, 0, self.px], [0, self.fy, self.py], [0, 0, 1]],
                device="cuda",
                dtype=self.joint_angles.dtype,
            )

            pose_matrix_b = self.model.cTr_to_pose_matrix(cTr_b)  # [B, 4, 4]

            R_list, t_list = lndFK(self.joint_angles)
            R_list = R_list.to(self.model.device)
            t_list = t_list.to(self.model.device)
            p_local1 = (
                th.tensor([0.0, 0.0004, 0.009])
                .to(self.joint_angles.dtype)
                .to(self.model.device)
            )
            p_local2 = (
                th.tensor([0.0, -0.0004, 0.009])
                .to(self.joint_angles.dtype)
                .to(self.model.device)
            )
            p_img1 = get_img_coords_batch(
                p_local1,
                R_list[2],
                t_list[2],
                pose_matrix_b.squeeze().to(self.joint_angles.dtype),
                intr,
            )
            p_img2 = get_img_coords_batch(
                p_local2,
                R_list[3],
                t_list[3],
                pose_matrix_b.squeeze().to(self.joint_angles.dtype),
                intr,
            )
            # They are both B, 2

            proj_pts = th.stack((p_img1, p_img2), dim=1)  # [B, 2, 2]

            ref_kps_2d = ref_kps_2d.unsqueeze(0).expand(
                B, ref_kps_2d.shape[0], ref_kps_2d.shape[1]
            )  # [B, 2, 2]
            pts_val = keypoint_loss_batch(proj_pts, ref_kps_2d)  # [B]
        else:
            pts_val = th.tensor(0.0, device=self.device)

        # TODO Batchfy cylinder projection and computation, projection is done.

        position = th.zeros((B, 3), dtype=th.float32, device=self.device)  # (B, 3)
        direction = th.zeros((B, 3), dtype=th.float32, device=self.device)  # (B, 3)
        direction[:, 2] = 1.0
        pose_matrix_b = self.model.cTr_to_pose_matrix(cTr_b).squeeze(
            0
        )  # shape(B, 4, 4)
        radius = 0.0085 / 2

        # We'll do a small custom function "cylinder_loss_single" that returns (loss, angle).
        cylinder_val = self.cylinder_loss_batch(
            position,
            direction,
            pose_matrix_b,
            radius,
            self.fx,
            self.fy,
            self.px,
            self.py,
        )  # [B]

        # normalize
        ld1, ld2, ld3 = np.array([ld1, ld2, ld3]) / np.linalg.norm(
            np.array([ld1, ld2, ld3])
        )

        item_loss = (
            10 * ld1 * mse_val
            + ld2 * (1e-7) * dist_val
            + ld3 * (1e-5) * app_val
            + 5 * (1e-3) * pts_val
            + 10 * (1e-3) * cylinder_val
        )
        item_loss_mask = th.isnan(item_loss)
        if item_loss_mask.any():
            print(
                f"Found NaN in per_sample_loss for samples: {[mse_val], [dist_val], [app_val], [pts_val], [cylinder_val]}"
            )

        return item_loss  # [B]

    # @th.compile
    def cylinder_loss_batch(
        self, position, direction, pose_matrix, radius, fx, fy, px, py
    ):
        """
        Batchfied Input:
        position [B ,3]
        direction [B ,3]
        pose_matrix [B, 4, 4]
        radius  [1]

        Batchfied Output:
        Cylinder loss [B,]
        """

        # get the projected cylinder lines parameters
        ref_mask = self.ref_mask
        intr = th.tensor(
            [[fx, 0, px], [0, fy, py], [0, 0, 1]],
            device=self.model.device,
            dtype=th.float32,
        )

        _, cam_pts_3d_position = transform_points_b(position, pose_matrix, intr)
        _, cam_pts_3d_norm = transform_points_b(direction, pose_matrix, intr)
        cam_pts_3d_norm = th.nn.functional.normalize(
            cam_pts_3d_norm
        )  # NORMALIZE !!!!!!

        # print(f"checking shape of cylinder input: {cam_pts_3d_position.shape, cam_pts_3d_norm.shape}")  both [B,3]
        e_1, e_2 = projectCylinderTorch(
            cam_pts_3d_position, cam_pts_3d_norm, radius, fx, fy, px, py
        )  # [B,2], [B,2]
        projected_lines = th.stack((e_1, e_2), dim=1)  # [B, 2, 2]

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
                detected_lines = th.from_numpy(np.stack((a, b), axis=-1)).to(
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
        # print(f"checking the shape of detected and projected lines {detected_lines}, {projected_lines}")  # [2, 2] [2, 2]
        cylinder_loss = cylinder_loss_params(detected_lines, projected_lines)  # [B]
        # parallelism = angle_between_lines(projected_lines, detected_lines)
        # print(f"testing line angle...{parallelism}")
        return cylinder_loss

    # TODO finish the optimization pipeline

    def optimize_batch(
        self,
        iterations=300,
        grid_search=False,
        ld1=3,
        ld2=3,
        ld3=3,
    ):
        """
        Runs multiple optimization steps on cTr_batch.
        - We do 1 forward/backward pass per iteration.
        - Each pass sums (or averages) the single-item loss over B => single scalar.

        Returns:
          final_cTr: shape (B,6)
          final_losses: shape (B,) each sample's final loss after last iteration
        """

        B = self.cTr_batch.shape[0]

        initial_cTr_batch = self.cTr_batch.clone().detach()

        for it in tqdm(range(iterations), desc="optimizing batch samples...."):
            self.optimizer.zero_grad()

            # If you'd like a "coordinate descent" approach for angle vs xyz,
            # you can do so by toggling the .requires_grad of self.cTr_batch[:, :3] or something.
            # For demonstration, let's just treat them as all free:

            # We do one forward pass => (B,H,W)
            # print(f"debugging ctr batches: {self.cTr_batch.shape}")
            # import time
            # start_time = time.time()
            pred_masks_b = self.model.render_robot_mask_batch(
                self.cTr_batch, self.robot_mesh, self.robot_renderer
            )  # shape (B,H,W)
            # print(f"Rendering time: {time.time() - start_time:.4f} seconds")

            ref_masks_b = self.ref_mask_b  # shape (B,H,W)
            per_sample_loss = self.batch_loss(
                pred_masks_b,
                ref_masks_b,
                self.cTr_batch,
                self.ref_keypoints,
                ld1,
                ld2,
                ld3,
            )  # [B]

            total_loss = per_sample_loss.mean()
            total_loss.backward()
            # th.nn.utils.clip_grad_norm_([self.cTr_batch], max_norm = 1)
            self.optimizer.step()

        # Compute the final per-sample losses
        with th.no_grad():
            pred_masks_b = self.model.render_robot_mask_batch(
                self.cTr_batch, self.robot_mesh, self.robot_renderer
            )

            final_losses = self.batch_loss(
                pred_masks_b,
                self.ref_mask_b,
                self.cTr_batch,
                self.ref_keypoints,
                ld1,
                ld2,
                ld3,
            )  # shape [B]

        # Avoid nan:
        # Mask to identify valid (non-NaN) losses
        valid_mask = th.isfinite(final_losses).to(device=final_losses.device)

        # Ensure there are valid losses to pick from
        if th.any(valid_mask):
            # Find the index of the lowest loss among valid losses
            valid_losses = final_losses[valid_mask]
            valid_indices = th.arange(len(final_losses), device=final_losses.device)[
                valid_mask
            ]
            lowest_loss_idx = valid_indices[th.argmin(valid_losses)]
            # print(f"debugging the filter: {final_losses},  {valid_losses}")
            # Pick the cTr with the lowest valid loss
            best_initial_cTr = initial_cTr_batch[lowest_loss_idx].detach()
        else:
            # Handle the case where all losses are NaN
            raise ValueError(
                "All losses are NaN. Cannot determine the best initial cTr."
            )

        return best_initial_cTr, final_losses[lowest_loss_idx].item()
