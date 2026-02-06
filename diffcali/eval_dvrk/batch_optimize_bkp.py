import cv2
import numpy as np
import torch as th
import skfmm
from scipy.spatial.distance import directed_hausdorff
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from diffcali.models.mark_kp import lndFK, get_img_coords
from diffcali.utils.cylinder_projection_utils import (
    projectCylinderTorch,
    transform_points,
)


def cylinder_loss_params(detected_lines, projected_lines):
    def to_theta_rho(lines):
        # lines shape: [2,2] with each line: (a,b)
        a = lines[:, 0]
        b = lines[:, 1]
        n = th.sqrt(a**2 + b**2) + 1e-9
        rho = 1.0 / n
        theta = th.atan2(b, a)  # [2] range from -pi to +pi
        return theta, rho

    # Define a helper to compute line-line difference:
    def line_difference(theta1, rho1, theta2, rho2):
        # Compute angle difference
        delta_theta = th.abs(
            theta1 - theta2
        )  # I guess we dont need to handle periodicity here
        delta_theta = th.min(delta_theta, 2 * np.pi - delta_theta)  # handle periodicity
        delta_theta = th.min(delta_theta, np.pi - delta_theta)
        # Compute distance difference
        delta_rho = th.abs(rho1 - rho2)
        return delta_rho.mean() + 0.7 * delta_theta.mean(), delta_theta.mean()

    # Try both permutations:
    # Pairing 1: Det[0]↔Proj[0], Det[1]↔Proj[1]
    theta_det, rho_det = to_theta_rho(detected_lines)
    theta_proj, rho_proj = to_theta_rho(projected_lines)
    loss_1_0, theta_1_0 = line_difference(
        theta_det[0:1], rho_det[0:1], theta_proj[0:1], rho_proj[0:1]
    )
    loss_1_1, theta_1_1 = line_difference(
        theta_det[1:2], rho_det[1:2], theta_proj[1:2], rho_proj[1:2]
    )
    total_loss_1 = loss_1_0 + loss_1_1
    avg_theta_1 = (theta_1_0 + theta_1_1) / 2.0

    # Pairing 2: Det[0]↔Proj[1], Det[1]↔Proj[0]
    loss_2_0, theta_2_0 = line_difference(
        theta_det[0:1], rho_det[0:1], theta_proj[1:2], rho_proj[1:2]
    )
    loss_2_1, theta_2_1 = line_difference(
        theta_det[1:2], rho_det[1:2], theta_proj[0:1], rho_proj[0:1]
    )
    total_loss_2 = loss_2_0 + loss_2_1
    avg_theta_2 = (theta_2_0 + theta_2_1) / 2.0

    # print(f"debugging the theta...{ theta_1_0,  theta_1_1}, { theta_2_0, theta_2_1}")

    # centerline:
    theta_det_mean = th.mean(theta_det)
    theta_proj_mean = th.mean(theta_proj)
    rho_det_mean = th.mean(rho_det)
    rho_proj_mean = th.mean(rho_proj)
    centerline_loss, _ = line_difference(
        theta_det_mean, rho_det_mean, theta_proj_mean, rho_proj_mean
    )
    # Select the minimal loss pairing        # parallelism = angle_between_lines(projected_lines, detected_lines)

    if total_loss_1 < total_loss_2:
        line_loss = total_loss_1
        angle_mean = avg_theta_1
    else:
        line_loss = total_loss_2
        angle_mean = avg_theta_2
    line_loss = line_loss + centerline_loss
    angle_degree = th.rad2deg(angle_mean)
    return line_loss, angle_degree


# th.autograd.set_detect_anomaly(True)
def keypoint_chamfer_loss(keypoints_a, keypoints_b, pts=True):
    """
    Computes the Chamfer distance between two sets of keypoints.

    Args:
        keypoints_a (torch.Tensor): Tensor of keypoints (shape: [N, 2]).
        keypoints_b (torch.Tensor): Tensor of keypoints (shape: [M, 2]).

    Returns:
        torch.Tensor: The computed Chamfer distance.
    """
    # Expand dimensions for broadcasting
    pts_a = keypoints_a
    pts_b = keypoints_b

    if keypoints_a.size(0) != 2 or keypoints_b.size(0) != 2:
        raise ValueError(
            "This brute force method only works for exactly two keypoints in each set."
        )

    # Compute distances for both possible permutations:
    # Permutation 1: A0->B0 and A1->B1
    dist_1 = th.norm(keypoints_a[0] - keypoints_b[0]) + th.norm(
        keypoints_a[1] - keypoints_b[1]
    )

    # Permutation 2: A0->B1 and A1->B0
    dist_2 = th.norm(keypoints_a[0] - keypoints_b[1]) + th.norm(
        keypoints_a[1] - keypoints_b[0]
    )

    # Choose the pairing that results in minimal distance
    min_dist = th.min(dist_1, dist_2)

    """TODO: replace chamfer loss with brute force like in the cylinder loss"""
    # Align the centerline:
    centerline_loss = th.norm(th.mean(pts_a, dim=0) - th.mean(pts_b, dim=0))

    # print(f"debugging loss scale....{chamfer_loss, parallelism_loss, distance_constraint_loss}")
    if pts == True:
        return min_dist + centerline_loss
    else:
        # print(f"checking the cylinder loss scale: {chamfer_loss, centerline_loss}")
        return min_dist


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
    # Compute distance map from reference to predicted
    loss_ref_to_pred = th.sum(pred_mask * distance_map_ref)
    total_loss = loss_ref_to_pred
    return total_loss


def appearance_loss(pred_mask, ref_mask):
    sum_pred = th.sum(pred_mask)
    sum_ref = th.sum(ref_mask)
    return th.abs(sum_pred - sum_ref)


def hausdorff_distance(pred_mask, ref_mask):
    # Find the non-zero points (i.e., the contour points) in the masks
    pred_mask_np = pred_mask.detach().cpu().numpy().astype(np.float32)
    ref_mask_np = ref_mask.detach().cpu().numpy().astype(np.float32)
    pred_points = np.argwhere(pred_mask_np > 0)
    ref_points = np.argwhere(ref_mask_np > 0)

    # Calculate directed Hausdorff distances in both directions
    hausdorff_1_to_2 = directed_hausdorff(pred_points, ref_points)[0]
    hausdorff_2_to_1 = directed_hausdorff(ref_points, pred_points)[0]

    # The final Hausdorff distance is the maximum of the two directed distances
    hausdorff_dist = max(hausdorff_1_to_2, hausdorff_2_to_1)

    return th.tensor(hausdorff_dist)


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


# TODO Batchfy all the loss computation


class BatchOptimize:
    """
    This class replicates your single-sample approach for an entire batch of cTr transforms [B,6].
    Each iteration:
      1) Renders [B,H,W]
      2) For each item in the batch, computes the same 'mse + dist + app + keypoints + cylinder'
         (like your _loss function).
      3) Averages them into a single scalar => backprop => .step()

    At the end, we can return the final cTr, the per-item losses, and angles for analysis.
    """

    def __init__(
        self,
        cTr_batch,  # (B,6) angle-axis+translation
        joint_angles,  # shape (4) or (B,4) if each sample has different joint angles
        model,
        robot_mesh,
        robot_renderer,
        ref_mask,  # (H,W) or optionally (B,H,W) if each sample has different ref
        ref_keypoints=None,  # shape (B,2,2) or (2,2)
        fx=None,
        fy=None,
        px=None,
        py=None,
        lr=1e-4,
    ):
        """
        If ref_keypoints is the same for all items, store shape (2,2). If each item has separate keypoints, store (B,2,2).
        """
        self.model = model
        self.robot_mesh = robot_mesh
        self.robot_renderer = robot_renderer
        self.device = model.device

        # Store cTr_batch as a trainable parameter
        self.cTr_batch = th.nn.Parameter(cTr_batch.clone()).to(
            self.device
        )  # shape (B,6)
        self.optimizer = th.optim.Adam([self.cTr_batch], lr=lr)

        self.joint_angles = joint_angles  # shape (4) or (B,4)
        self.ref_mask = None
        self.distance_map = None  # Ref only
        self.weighting_mask = None
        # Convert reference mask to shape (H,W) => store it in float
        # If you have a single reference for the entire batch, we just broadcast later
        # If you want separate references, store shape (B,H,W).
        if len(ref_mask.shape) == 2:
            # Single mask for all
            self.ref_mask = ref_mask.unsqueeze(0).to(self.device)  # => (1,H,W)
        else:
            # If you already have shape (B,H,W), store directly
            self.ref_mask = ref_mask.to(self.device)

        self.ref_keypoints = th.tensor(ref_keypoints).to(self.device).float()
        self.fx = fx
        self.fy = fy
        self.px = px
        self.py = py
        self.det_line_params = None
        # We'll define constants from your single-sample code
        # (like "10, 0.7" etc) inside a method or pass them as arguments
        self.gamma = 1

    def readRefImage(self, ref_mask):
        cv_img = cv2.imread(ref_mask)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = cv2.inRange(cv_img, np.ones(3) * 128, np.ones(3) * 255) / 255.0
        self.ref_mask = th.Tensor(img).to(self.model.device)
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

    def single_item_loss(
        self,
        idx,
        pred_mask_2d,  # shape (H,W)
        ref_mask_2d,  # shape (H,W)
        cTr_6,
        ref_kps_2d,  # shape (2,2) or None
        ld1=3,
        ld2=3,
        ld3=3,
    ):
        """
        This replicates your '_loss' logic for a single item in the batch.
        pred_mask_2d: predicted silhouette for sample idx => shape (H,W)
        ref_mask_2d: reference mask => shape (H,W)
        cTr_6:  shape (6,) for this sample
        ref_kps_2d: shape (2,2) if you're using the same keypoints for all or unique per sample
        Return:
            item_loss (scalar),
            angle (scalar) from cylinder loss
        """

        # weighting mask
        weighting_mask = self.weighting_mask

        # MSE
        mse_val = F.mse_loss(
            pred_mask_2d * weighting_mask, ref_mask_2d * weighting_mask
        )

        # distance loss
        dist_val = distance_loss(
            pred_mask_2d * weighting_mask, self.distance_map * weighting_mask
        )

        # appearance
        app_val = appearance_loss(pred_mask_2d, ref_mask_2d)

        # keypoint
        if ref_kps_2d is not None:
            intr = th.tensor(
                [
                    [882.99611514, 0, 445.06146749],
                    [0, 882.99611514, 190.24049547],
                    [0, 0, 1],
                ],
                device="cuda",
                dtype=self.joint_angles.dtype,
            )
            cTr_6_b = cTr_6.unsqueeze(0)
            pose_matrix = self.model.cTr_to_pose_matrix(cTr_6_b)
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
            p_img1 = get_img_coords(
                p_local1,
                R_list[2],
                t_list[2],
                pose_matrix.squeeze().to(self.joint_angles.dtype),
                intr,
            )
            p_img2 = get_img_coords(
                p_local2,
                R_list[3],
                t_list[3],
                pose_matrix.squeeze().to(self.joint_angles.dtype),
                intr,
            )
            proj_pts = th.stack([p_img1, p_img2], dim=0)
            pts_val = keypoint_chamfer_loss(proj_pts, ref_kps_2d)
        else:
            pts_val = th.tensor(0.0, device=self.device)

        position = th.zeros((1, 3), dtype=th.float32, device=self.device)
        direction = th.zeros((1, 3), dtype=th.float32, device=self.device)
        direction[:, 2] = 1.0
        pose_mat = self.model.cTr_to_pose_matrix(cTr_6_b).squeeze(0)  # shape(4,4)
        radius = 0.0085 / 2
        # We'll do a small custom function "cylinder_loss_single" that returns (loss, angle).
        cylinder_val, angle_val = self.cylinder_loss_single(
            position, direction, pose_mat, radius, self.fx, self.fy, self.px, self.py
        )
        # normalize
        ld1, ld2, ld3 = np.array([ld1, ld2, ld3]) / np.linalg.norm(
            np.array([ld1, ld2, ld3])
        )
        item_loss = (
            10 * ld1 * mse_val
            + ld2 * (1e-7) * dist_val
            + ld3 * (1e-5) * app_val
            + 5 * (1e-3) * pts_val
            + 1 * (1e-3) * cylinder_val
        )
        return item_loss, angle_val

    def cylinder_loss_single(
        self, position, direction, pose_matrix, radius, fx, fy, px, py
    ):
        """
        Computes the cylinder alignment loss

        Args:
            position: Tensor (B, 3) in mesh local frame, a point on the center line
            direction: Tensor (B, 3) in mesh local frame, the direction of the center line
            ctr: Tensor (6), transform to pose
            radius: the radius of the cylinder
            fx, fy, cx, cy: intr
            ref_mask: the ref_mask path
        Returns:
            torch.Tensor: The distance alignment loss
        """

        # get the projected cylinder lines parameters
        ref_mask = self.ref_mask

        intr = th.tensor(
            [[fx, 0, px], [0, fy, py], [0, 0, 1]],
            device=self.model.device,
            dtype=th.float32,
        )

        _, cam_pts_3d_position = transform_points(position, pose_matrix, intr)
        _, cam_pts_3d_norm = transform_points(direction, pose_matrix, intr)
        # print(f"checking shape of cylinder input: {cam_pts_3d_position.shape, cam_pts_3d_norm.shape}")  both [1,3]
        e_1, e_2 = projectCylinderTorch(
            cam_pts_3d_position, cam_pts_3d_norm, radius, fx, fy, px, py
        )  # [1,2], [1,2]
        projected_lines = th.cat((e_1, e_2), dim=0)

        # get the detected reference lines parameters
        if self.det_line_params == None:
            ref_mask_np = ref_mask.detach().cpu().numpy()
            assert ref_mask_np.dtype != np.uint8
            ref_mask_np = (ref_mask_np * 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(ref_mask_np, (3, 3), 0)
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=50,
                maxLineGap=10,
            )

            # Draw the lines
            # Check if any lines are detected
            if lines is not None:
                # Extract all line endpoints (shape: [num_lines, 1, 4])
                lines = lines[:, 0, :]  # Shape now: [num_lines, 4]

                # Vectorized calculation of line lengths
                x1 = lines[:, 0]
                y1 = lines[:, 1]
                x2 = lines[:, 2]
                y2 = lines[:, 3]

                # Compute lengths for all lines (no loop)
                lengths = np.hypot(x2 - x1, y2 - y1)

                # Get the indices of the lines sorted by length (descending order)
                sorted_indices = np.argsort(-lengths)

                # Select the top N longest lines
                N = 2
                top_indices = sorted_indices[:N]

                # Extract the top N longest lines
                longest_lines = lines[top_indices]
            else:
                longest_lines = []

            longest_lines = np.array(longest_lines, dtype=np.float64)
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
            norm = c + 1e-6  # Ensure no division by zero
            a /= norm
            b /= norm

            # Stack line parameters into a tensor and normalize to match au + bv = 1 form
            detected_lines = th.from_numpy(np.stack((a, b), axis=-1)).to(
                self.model.device
            )
            self.det_line_params = detected_lines
        # print(f"debugging the line params: {detected_lines}")
        self.proj_line_params = projected_lines
        detected_lines = self.det_line_params

        # print(f"checking the shape of detected and projected lines {detected_lines}, {projected_lines}")  # [2, 2] [2, 2]
        cylinder_loss, angle = cylinder_loss_params(detected_lines, projected_lines)
        # parallelism = angle_between_lines(projected_lines, detected_lines)
        # print(f"testing line angle...{parallelism}")
        return cylinder_loss, angle

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
          final_angles: shape (B,) each sample's last cylinder angle
        """
        B = self.cTr_batch.shape[0]
        (H, W) = self.ref_mask.shape[-2:]  # either (1,H,W) or (B,H,W)

        for it in range(iterations):
            self.optimizer.zero_grad()

            # If you'd like a "coordinate descent" approach for angle vs xyz,
            # you can do so by toggling the .requires_grad of self.cTr_batch[:, :3] or something.
            # For demonstration, let's just treat them as all free:

            # We do one forward pass => (B,H,W)
            # print(f"debugging ctr batches: {self.cTr_batch.shape}")
            pred_masks_b = self.model.render_robot_mask_batch(
                self.cTr_batch, self.robot_mesh, self.robot_renderer
            )  # shape (B,H,W)

            # Now compute the total loss over B
            # We'll do it item-by-item, then average
            item_losses = []
            item_angles = []
            for i in range(B):
                # pred_mask_2d => shape (H,W)
                pred_mask_2d = pred_masks_b[i]
                # broadcast or index reference
                # if we have a single mask => self.ref_mask[0]; if B separate => self.ref_mask[i]
                ref_mask_2d = self.ref_mask  # shape (H,W)

                # keypoints => if shape(2,2), use that; if shape(B,2,2), do self.ref_keypoints[i]
                if self.ref_keypoints is not None:
                    ref_kps_2d = self.ref_keypoints
                else:
                    ref_kps_2d = None

                cTr_6 = self.cTr_batch[i]  # shape (6,)

                # print(f"debugging ctr 6 b {cTr_6.shape}")

                item_loss, angle_val = self.single_item_loss(
                    idx=i,
                    pred_mask_2d=pred_mask_2d,
                    ref_mask_2d=ref_mask_2d,
                    cTr_6=cTr_6,
                    ref_kps_2d=ref_kps_2d,
                    ld1=ld1,
                    ld2=ld2,
                    ld3=ld3,
                )
                item_losses.append(item_loss.unsqueeze(0))
                item_angles.append(angle_val.unsqueeze(0))

            # Combine => shape (B,)
            item_losses_t = th.cat(item_losses, dim=0)
            # We do a single scalar for backprop => average or sum
            total_loss = item_losses_t.mean()

            total_loss.backward()
            self.optimizer.step()

        # after final iteration, compute final losses & angles
        # do a final forward pass, or re-use the last pass if you want
        with th.no_grad():
            pred_masks_b = self.model.render_robot_mask_batch(
                self.cTr_batch, self.robot_mesh, self.robot_renderer
            )
            final_losses_list = []
            final_angles_list = []
            for i in range(B):
                pred_mask_2d = pred_masks_b[i]
                ref_mask_2d = self.ref_mask
                if self.ref_keypoints is not None:
                    if len(self.ref_keypoints.shape) == 2:
                        ref_kps_2d = self.ref_keypoints
                    else:
                        ref_kps_2d = self.ref_keypoints[i]
                else:
                    ref_kps_2d = None

                cTr_6 = self.cTr_batch[i]
                item_loss, angle_val = self.single_item_loss(
                    idx=i,
                    pred_mask_2d=pred_mask_2d,
                    ref_mask_2d=ref_mask_2d,
                    cTr_6=cTr_6,
                    ref_kps_2d=ref_kps_2d,
                    ld1=ld1,
                    ld2=ld2,
                    ld3=ld3,
                )
                final_losses_list.append(item_loss.unsqueeze(0))
                final_angles_list.append(angle_val.unsqueeze(0))

            final_losses = th.cat(final_losses_list, dim=0)  # (B,)
            final_angles = th.cat(final_angles_list, dim=0)  # (B,)

        return self.cTr_batch.detach().clone(), final_losses, final_angles
