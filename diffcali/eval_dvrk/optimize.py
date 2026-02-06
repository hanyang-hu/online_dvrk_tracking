import os
import cv2
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from tqdm import tqdm
import skfmm
from scipy.spatial.distance import directed_hausdorff

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from diffcali.eval_dvrk.LND_fk import lndFK
from diffcali.models.mark_kp import get_img_coords
import diffcali.eval_dvrk.plot_utils as plot_utils
from diffcali.utils.cylinder_projection_utils import (
    projectCylinderTorch,
    transform_points,
)
from diffcali.utils.detection_utils import detect_lines


def angle_between_lines(params_set1, params_set2):
    """
    Compute the average angle between corresponding lines in two sets using PyTorch.

    Args:
        params_set1: torch.Tensor of shape (2, 2), each row is [a, b] for set 1.
        params_set2: torch.Tensor of shape (2, 2), each row is [a, b] for set 2.

    Returns:
        torch.Tensor: Average angle (in degrees) between corresponding lines in the two sets.
    """
    # Extract parameters
    # print(f"debugging the line params {params_set1}, {params_set2}")
    a1, b1 = params_set1[:, 0], params_set1[:, 1]
    a2, b2 = params_set2[:, 0], params_set2[:, 1]

    # Compute dot products
    dot_products = a1 * a2 + b1 * b2

    # Compute magnitudes of the vectors
    norms1 = th.sqrt(a1**2 + b1**2)
    norms2 = th.sqrt(a2**2 + b2**2)

    # Compute cosine similarity
    cos_theta = dot_products / (
        norms1 * norms2 + 1e-6
    )  # Add epsilon for numerical stability

    # Clip values to avoid numerical issues with arccos
    cos_theta = th.clamp(cos_theta, -1.0, 1.0)

    # Compute angles in radians and convert to degrees
    angles = th.acos(cos_theta)  # Radians
    angles_degree = th.rad2deg(angles)[0]  # Convert to degrees
    if angles_degree > 90:
        angles_degree = 180 - angles_degree
    return angles_degree


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

    # distance_loss_1 = th.norm(detected_lines[0] - projected_lines[0]) + th.norm(
    #     detected_lines[1] - projected_lines[1]
    # )
    # distance_loss_2 = th.norm(detected_lines[0] - projected_lines[1]) + th.norm(
    #     detected_lines[1] - projected_lines[0]
    # )
    # centerline_loss_distance = th.norm(
    #     th.mean(detected_lines, dim=0) - th.mean(projected_lines, dim=0)
    # )
    # if distance_loss_1 > distance_loss_2:
    #     line_loss_distance = distance_loss_2 + centerline_loss_distance
    # else:
    #     line_loss_distance = distance_loss_1 + centerline_loss_distance
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

    # """experiement on parallelism loss as well"""
    # # Parallelism constraint: Compute slopes and enforce similarity
    # if (
    #     pts_a.size(0) >= 2 and pts_b.size(0) >= 2
    # ):  # Ensure at least two points in each set
    #     slope_a = (pts_a[1, 1] - pts_a[0, 1]) / (pts_a[1, 0] - pts_a[0, 0] + 1e-6)
    #     slope_b = (pts_b[1, 1] - pts_b[0, 1]) / (pts_b[1, 0] - pts_b[0, 0] + 1e-6)
    #     parallelism_loss = th.abs(slope_a - slope_b)

    #     # Calculate distances between the two points in each set
    #     distance_a = th.norm(pts_a[1] - pts_a[0])  # Distance in set A
    #     distance_b = th.norm(pts_b[1] - pts_b[0])  # Distance in set B

    #     # Minimize the difference between distances (distance constraint loss)
    #     distance_constraint_loss = th.abs(distance_a - distance_b)
    # else:
    #     parallelism_loss = th.tensor(0.0, device=keypoints_a.device)
    #     distance_constraint_loss = th.tensor(0.0, device=keypoints_a.device)
    # """experiement on parallelism loss as well"""

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


def distance_loss(pred_mask, ref_mask, gamma):
    # Compute distance map from reference to predicted
    distance_map_ref = compute_distance_map(ref_mask, gamma)
    loss_ref_to_pred = th.sum(pred_mask * distance_map_ref)

    # Compute distance map from predicted to reference

    # distance_map_pred = compute_distance_map(pred_mask, gamma)
    # loss_pred_to_ref = th.sum(ref_mask * distance_map_pred)

    # Combine the losses
    # total_loss = (loss_ref_to_pred + loss_pred_to_ref)/2
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

    # Clip points to image boundaries
    # x1, y1 = np.clip([x1, y1], [0, 0], [image_width - 1, image_height - 1])
    # x2, y2 = np.clip([x2, y2], [0, 0], [image_width - 1, image_height - 1])

    return (x1, y1), (x2, y2)


class Optimize(object):
    def __init__(
        self,
        cTr_train,
        model,
        robot_mesh,
        robot_renderer,
        lr=1e-4,
        cTr_nontrain=None,
        buildcTr=None,
    ):
        """
        param lr: learning rate of the optimizer
        """
        self.reset_lr = lr
        self.cTr_train = cTr_train
        self.model = model
        self.robot_mesh = robot_mesh
        self.robot_renderer = robot_renderer

        self.optimizer = th.optim.Adam(self.cTr_train, lr=lr)
        # self.optimizer = th.optim.SGD(self.cTr_train, lr=lr, momentum=0.9)
        # self.optimizer = th.optim.LBFGS(self.cTr_train, lr=lr)
        self.cTr_nontrain = cTr_nontrain
        self.buildcTr = buildcTr
        self.loss = th.nn.MSELoss().cuda()
        self.proj_keypoints = None
        self.ref_keypoints = None
        self.fx = None
        self.fy = None
        self.px = None
        self.py = None
        self.proj_line_params = None
        self.det_line_params = None
        # self._loss = th.compile(self._loss)
        # self.loss = th.nn.SmoothL1Loss().cuda()

    def readRefImage(self, ref_img_file):
        cv_img = cv2.imread(ref_img_file)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = cv2.inRange(cv_img, np.ones(3) * 128, np.ones(3) * 255) / 255.0

        self.ref_img = th.Tensor(img).to(self.model.device)

    def optimize(
        self,
        iterations=200,
        save_fig_dir=None,
        ld1=15,
        ld2=10,
        ld3=4,
        set2=[],
        xyz_steps=10,
        angles_steps=100,
        saving_interval=30,
        coarse_step_num=300,
        grid_search=False,
    ):

        # rendered_image = self.model.render_single_robot_mask(
        #     self.buildcTr(self.cTr_train, self.cTr_nontrain),
        #     self.robot_mesh,
        #     self.robot_renderer,
        # )

        "cTr_train: [aa, xy, z]"
        # scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)
        losses = np.zeros([iterations, 2])
        angles_over_time = []
        intr = th.tensor(
            [[self.fx, 0, self.px], [0, self.fy, self.py], [0, 0, 1]],
            device="cuda",
            dtype=self.cTr_train[3].dtype,
        )

        progress_bar = tqdm(range(iterations), desc="optimizing the pose...")
        for idx in progress_bar:
            self.model.get_joint_angles(self.cTr_train[3])
            # import time
            # start_time = time.time()
            self.robot_mesh = self.robot_renderer.get_robot_mesh(self.cTr_train[3])
            # print(f"Elapsed time for get_robot_mesh: {time.time() - start_time:.4f} seconds")
            self.optimizer.zero_grad()

            # Alternating optimization - Coordinate Descent
            full_circle = xyz_steps + angles_steps
            if (idx % full_circle) < xyz_steps:  # Optimize `xyz` first
                self.cTr_train[0].requires_grad = False  # Angles frozen
                self.cTr_train[1].requires_grad = True  # Optimize xyz
                self.cTr_train[2].requires_grad = True
                # self.cTr_train[3].requires_grad = True
            else:  # Optimize angles after `xyz`
                self.cTr_train[0].requires_grad = True  # Angles
                self.cTr_train[1].requires_grad = False  # Freeze xyz
                self.cTr_train[2].requires_grad = False
                # self.cTr_train[3].requires_grad = True
            # set up the rendering target, partial & full body

            if idx <= coarse_step_num:
                # make sure after first 300 there will be some overlapping
                ld1, ld2, ld3 = np.array([ld1, ld2, ld3]) / np.linalg.norm(
                    np.array([ld1, ld2, ld3])
                )
                self.cTr_train[0].requires_grad = True
                self.cTr_train[1].requires_grad = True
                self.cTr_train[2].requires_grad = True
                grid_search = True  # Try to find the good init first
            else:
                # self.optimizer = th.optim.Adam(self.cTr_train, lr=self.reset_lr)
                ld1, ld2, ld3 = set2
                ld1, ld2, ld3 = np.array([ld1, ld2, ld3]) / np.linalg.norm(
                    np.array([ld1, ld2, ld3])
                )
                grid_search = False

            if idx == coarse_step_num:
                print("Start coordinate decent.....")

            # import time
            # start_time = time.time()
            # self.model.args.use_nvdiffrast = False
            rendered_image = self.model.render_single_robot_mask(
                self.buildcTr(self.cTr_train, self.cTr_nontrain),
                self.robot_mesh,
                self.robot_renderer
            )
            # print(f"Rendering time: {time.time() - start_time:.4f} seconds")
            self.compute_weighting_mask(rendered_image.squeeze().shape) # compute weighting mask in advance

            # if grid_search == False:
            #     if idx == 0:
            #         ref_pts = get_reference_keypoints(self.ref_img, num_keypoints=2)
            #         # Convert to torch tensor
            #         ref_pts = th.tensor(ref_pts, device= self.model.device, dtype=th.float32)
            # else:
            #     ref_pts = None

            # if idx == 0 and grid_search and self.ref_keypoints is None:
            #     # Click reference keypoints once and save them
            #     self.ref_keypoints = get_reference_keypoints(self.ref_img, num_keypoints=2)
            #     # Convert to torch tensor
            #     self.ref_keypoints = th.tensor(self.ref_keypoints, device=self.model.device, dtype=th.float32)

            # Use the stored reference keypoints
            self.ref_keypoints = th.tensor(
                self.ref_keypoints, device=self.model.device, dtype=th.float32
            ).squeeze()
            ref_pts = self.ref_keypoints

            """project the keypoints here"""
            # print(f"debug build ctr{(self.buildcTr(self.cTr_train, self.cTr_nontrain)).shape}")

            pose_matrix = self.model.cTr_to_pose_matrix(
                (self.buildcTr(self.cTr_train, self.cTr_nontrain)).unsqueeze(0)
            )
            R_list, t_list = lndFK(self.cTr_train[3])
            R_list = R_list.to(self.model.device)
            t_list = t_list.to(self.model.device)
            p_local1 = (
                th.tensor([0.0, 0.0004, 0.009])
                .to(self.cTr_train[3].dtype)
                .to(self.model.device)
            )
            p_local2 = (
                th.tensor([0.0, -0.0004, 0.009])
                .to(self.cTr_train[3].dtype)
                .to(self.model.device)
            )
            p_img1 = get_img_coords(
                p_local1,
                R_list[2],
                t_list[2],
                pose_matrix.squeeze().to(self.cTr_train[3].dtype),
                intr,
            )
            p_img2 = get_img_coords(
                p_local2,
                R_list[3],
                t_list[3],
                pose_matrix.squeeze().to(self.cTr_train[3].dtype),
                intr,
            )
            proj_pts = th.stack([p_img1, p_img2], dim=0)

            # print(f"debugging input shape: {proj_pts.shape}")
            loss, angle = self._loss(
                rendered_image, proj_pts, ref_pts, ld1, ld2, ld3, grid_search
            )

            # loss_nan_mask = th.isnan(loss)
            # if loss_nan_mask.any():
            #     print(f"Found NaN in losses: {loss_nan_mask.nonzero().flatten().tolist()}")
            #     # Option A: Zero out the NaNs
            #     loss = th.where(loss_nan_mask,
            #                             th.zeros_like(loss),
            #                             loss)

            angles_over_time.append(angle.item())
            # print(f"debugging loss {loss}")
            losses[idx] = np.array([idx + 1, loss.detach().cpu()])
            # loss.backward()
            if idx < iterations - 1:
                loss.backward(
                    retain_graph=True
                )  # Retain graph for future backward passes
            else:
                loss.backward()  # On the last iteration, no need to retain the graph
            # th.nn.utils.clip_grad_norm_([self.cTr_train], max_norm = 1)
            self.optimizer.step()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # scheduler.step()

            if ((idx + 1) % saving_interval == 0 or idx == 0 or idx == iterations - 1):
                if not save_fig_dir is None:
                    os.makedirs(save_fig_dir, exist_ok=True)

                    # Convert tensors to NumPy arrays for visualization
                    rendered_np = rendered_image.squeeze().detach().cpu().numpy()
                    ref_np = self.ref_img.squeeze().detach().cpu().numpy()
                    # Overlay the rendered mask with the reference mask
                    det_line_params = self.det_line_params
                    proj_line_params = self.proj_line_params
                    overlay = self._overlay_masks(
                        rendered_np,
                        ref_np,
                        ref_pts,
                        proj_pts,
                        det_line_params,
                        proj_line_params,
                    )

                    # Save the overlay image
                    # plt.imshow(overlay)
                    # plt.axis('off')
                    # plt.savefig(os.path.join(save_fig_dir, f'{idx}.png'))

                    save_path = os.path.join(save_fig_dir, f"{idx}.png")
                    self.save_image_cv(overlay, save_path)

                # Plot the loss curve
            if False and (idx + 1) == iterations and iterations >= 1000:
                plt.figure()  # Create a new figure
                plot_utils.curve2D(losses[: idx + 1], ["b"], "iteration", "loss")
                plt.savefig(f"loss_curve_ir{idx}.png")

                plt.plot(angles_over_time)
                plt.xlabel("Iteration")
                plt.ylabel("Angle Difference (degrees)")
                plt.title("Cylinder Angles Over Time")
                plt.show()

        return (
            self.buildcTr(self.cTr_train, self.cTr_nontrain),
            loss.detach().cpu(),
            angle.detach().cpu(),
        )

    def save_image_cv(self, overlay, save_path):
        overlay_bgr = (overlay * 255).astype(
            np.uint8
        )  # Convert to uint8 and BGR format
        cv2.imwrite(save_path, overlay_bgr)

    def compute_weighting_mask(self, shape, center_weight=1.0, edge_weight=0.5):
        """
        Create a weighting mask with a smooth transition from center to edge.

        param shape: Tuple representing the shape of the image (height, width)
        param center_weight: The weight at the center of the image
        param edge_weight: The weight at the edges (should be less than center_weight)
        return: A tensor representing the weighting mask
        """

        h, w = shape
        y, x = np.ogrid[:h, :w]

        # Calculate distance to the center
        center_y, center_x = h / 2, w / 2
        distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

        # Normalize distances to range from 0 (center) to 1 (edge)
        max_distance = np.sqrt(center_y**2 + center_x**2)
        normalized_distance = distance / max_distance

        # Invert distances to have highest weight at the center, with an edge weight bias
        weights = edge_weight + (center_weight - edge_weight) * (
            1 - normalized_distance
        )

        # Convert to tensor
        self.weighting_mask = th.from_numpy(weights).float().to(self.model.device).requires_grad_(False).contiguous()

    def cylinder_loss(self, position, direction, pose_matrix, radius, fx, fy, cx, cy):
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
        ref_mask = self.ref_img

        intr = th.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            device=self.model.device,
            dtype=th.float32,
        )

        _, cam_pts_3d_position = transform_points(position, pose_matrix, intr)
        _, cam_pts_3d_norm = transform_points(direction, pose_matrix, intr)
        cam_pts_3d_norm = th.nn.functional.normalize(cam_pts_3d_norm)
        e_1, e_2 = projectCylinderTorch(
            cam_pts_3d_position, cam_pts_3d_norm, radius, fx, fy, cx, cy
        )  # [1,2], [1,2]
        projected_lines = th.cat((e_1, e_2), dim=0)

        # get the detected reference lines parameters
        if self.det_line_params == None:
            ref_mask_np = ref_mask.detach().cpu().numpy()
            longest_lines = detect_lines(ref_mask_np, output=True)
            # if ref_mask_np.dtype != np.uint8:
            #     ref_mask_np = (ref_mask_np * 255).astype(np.uint8)

            # # filtered_mask = cv2.medianBlur(ref_mask_np, 13)
            # blurred = cv2.GaussianBlur(ref_mask_np, (3, 3), 0)
            # edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            # lines = cv2.HoughLinesP(edges, rho=0.5, theta = np.pi / 360, threshold=50, minLineLength=50, maxLineGap=20)

            # # Draw the lines
            # # Check if any lines are detected
            # if lines is not None:
            #     # Extract all line endpoints (shape: [num_lines, 1, 4])
            #     lines = lines[:, 0, :]  # Shape now: [num_lines, 4]

            #     # Vectorized calculation of line lengths
            #     x1 = lines[:, 0]
            #     y1 = lines[:, 1]
            #     x2 = lines[:, 2]
            #     y2 = lines[:, 3]

            #     # Compute lengths for all lines (no loop)
            #     lengths = np.hypot(x2 - x1, y2 - y1)

            #     # Get the indices of the lines sorted by length (descending order)
            #     sorted_indices = np.argsort(-lengths)

            #     # Select the top N longest lines
            #     N = 2
            #     top_indices = sorted_indices[:N]

            #     # Extract the top N longest lines
            #     longest_lines = lines[top_indices]
            # else:
            #     longest_lines = []

            # print(f"debugging the longest lines {longest_lines.shape}") # [2, 4]

            # longest_lines = longest_lines[:, 0, :]  # Shape now: [num_lines, 4]
            longest_lines = np.array(longest_lines, dtype=np.float64)
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

    def _loss(self, rendered_img, proj_pts, ref_pts, ld1, ld2, ld3, grid_search):
        gamma = 1
        # weighting_mask = self.compute_weighting_mask(rendered_img.squeeze().shape).to(
        #     self.model.device
        # )
        weighting_mask = self.weighting_mask
        # weighting_mask = th.ones_like(rendered_img.squeeze()).to(self.model.device)
        # weighting_mask = self.compute_weighting_mask(rendered_img.squeeze().shape).to(self.model.device)
        mse = self.loss(
            rendered_img.squeeze() * weighting_mask, self.ref_img * weighting_mask
        )
        # print(f"debugging mse scales: {mse}")
        dist = distance_loss(
            rendered_img.squeeze() * weighting_mask,
            self.ref_img * weighting_mask,
            gamma,
        )

        app = appearance_loss(rendered_img.squeeze(), self.ref_img)
        # if grid_search == False:
        # print(f"debugging ref_kps_2d { ref_pts}")
        pts = keypoint_chamfer_loss(proj_pts, ref_pts)

        """test the cylinder loss"""
        position = th.zeros(
            (1, 3), dtype=th.float32, device=self.model.device
        )  # (B, 3)
        # The direction of the cylinder is aligned along the z-axis
        direction = th.zeros((1, 3), dtype=th.float32, device=self.model.device)
        direction[:, 2] = 1.0  # Aligned along z-axis
        pose_matrix = self.model.cTr_to_pose_matrix(
            (self.buildcTr(self.cTr_train, self.cTr_nontrain)).unsqueeze(0)
        ).squeeze()

        radius = 0.0085 / 2  # adjust radius if needed
        # proj_position_2d, cam_pts_3d_position = transform_points(position, pose_matrix, intr)
        # proj_norm_2d, cam_pts_3d_norm = transform_points(direction, pose_matrix, intr)
        cylinder, angle = self.cylinder_loss(
            position, direction, pose_matrix, radius, self.fx, self.fy, self.px, self.py
        )
        # print(f"Show the cylinder loss scale.....{cylinder} and pts loss {pts}")
        """test the cylinder loss"""
        # print(f"check loss scales:{ld1*mse, ld2*(1e-7)*dist, ld3*(1e-5)*app, pts}")
        # print(f"debugging the loss scale {ld1*mse, ld2*(1e-7)*dist, ld3*(1e-5)*app, 7*(1e-4)*pts, 10*(1e-2)*cylinder}")
        # 10, 3, 6

        """TODO: optimize the angle calculation, get the avg???"""
        if grid_search == False:
            # print(f"debugging the loss scale {ld1*mse, ld2*(1e-7)*dist, ld3*(1e-5)*app, 7*(1e-4)*pts, (1e-4)*cylinder}")
            return (
                10 * ld1 * mse
                + ld2 * (1e-7) * dist
                + ld3 * (1e-5) * app
                + 5 * (1e-3) * pts
                + 10 * (1e-3) * cylinder,
                angle,
            )
            # return 2*(1e-3)*cylinder, angle

        else:
            return (
                10 * ld1 * mse
                + ld2 * (1e-7) * dist
                + ld3 * (1e-5) * app
                + 5 * (1e-3) * pts
                + 10 * (1e-3) * cylinder,
                angle,
            )
            # return 2*(1e-3)*cylinder, angle

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
        center_ref_pt = th.mean(ref_pts, dim=0)
        center_proj_pt = th.mean(proj_pts, dim=0)
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


# if __name__ == "__main__":
#     ref_img_file = "images/dvrk/left0000-mask.jpg"

#     opt = Optimize()
#     opt.readRefImage(ref_img_file)

# plt.imshow(self.ref_img.cpu().numpy())
# plt.show()
