import torch
import torch as th
import torch.nn.functional as F
from torch.quasirandom import SobolEngine
import kornia
import FastGeodis
import numpy as np
import math
import time
import cv2
import os
import gc
import tqdm
import inspect

from diffcali.eval_dvrk.LND_fk import lndFK, batch_lndFK
from diffcali.utils.projection_utils import get_img_coords, get_img_coords_batch
from diffcali.utils.angle_transform_utils import (
    mix_angle_to_axis_angle,
    axis_angle_to_mix_angle,
    mix_angle_to_rotmat,
)
from diffcali.utils.cma_es import (
    CMAES_cus, 
    CMAES_bd_cus,
    CMAES_bi_manual_cus,
    CMAES_bi_manual_bd_cus,
    generate_sigma_normal,
    generate_low_discrepancy_normal,
)
from diffcali.utils.pose_tracker import OneEuroFilter, KalmanFilter
from diffcali.utils.contour_tip_net import ContourTipNet, detect_keypoints, Tip2DNet, detect_keypoints_2d

from evotorch import Problem, SolutionBatch
from evotorch.algorithms import SearchAlgorithm, XNES
from evotorch.logging import Logger, StdOutLogger


torch.set_default_dtype(torch.float32)
torch._functorch.config.donated_buffer=False
# torch.autograd.set_detect_anomaly(True)


# @torch.compile()
def keypoint_loss_batch(keypoints_a, keypoints_b, p=1, sqrt=False, thres=10.):
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
    dist_1 = torch.norm(keypoints_a[:, 0] - keypoints_b[:, 0], dim=1, p=p) + torch.norm(
        keypoints_a[:, 1] - keypoints_b[:, 1], dim=1, p=p
    )
    dist_1 = torch.clamp(dist_1 - thres, min=0.)  # apply threshold

    # Permutation 2: A0->B1 and A1->B0
    dist_2 = torch.norm(keypoints_a[:, 0] - keypoints_b[:, 1], dim=1, p=p) + torch.norm(
        keypoints_a[:, 1] - keypoints_b[:, 0], dim=1, p=p
    )
    dist_2 = torch.clamp(dist_2 - thres, min=0.)  # apply threshold

    # Choose the pairing that results in minimal distance for each batch
    min_dist = torch.min(dist_1, dist_2)  # [B]

    # Align the centerline for each batch
    centerline_loss = torch.norm(
        torch.mean(keypoints_a, dim=1) - torch.mean(keypoints_b, dim=1), dim=1, p=p
    )  # [B]
    centerline_loss = torch.clamp(centerline_loss - thres, min=0.)  # apply threshold

    return min_dist + centerline_loss if not sqrt else torch.sqrt(min_dist + centerline_loss)


class DummyLogger(Logger):
    """
    A dummy logger that only maintains the best solution during the optimization.
    """
    def __init__(
        self, 
        searcher, 
        *, 
        interval: int = 1, 
        after_first_step: bool = False
    ):
        # Call the super constructor
        super().__init__(searcher, interval=interval, after_first_step=after_first_step)

        self.best_solution, self.best_eval = None, float('inf')
        
        self.has_warned_about_pop_best_eval = False

    def __call__(self, status: dict):
        # Update best value and evaluation
        try:
            if status["pop_best_eval"] < self.best_eval:
                self.best_solution = status["pop_best"].values.clone()
                self.best_eval = status["pop_best_eval"]
                # print(f"New best solution found: {self.best_solution}, evaluation: {self.best_eval}")
        except:
            if self.has_warned_about_pop_best_eval:
                return
            print(f"[Warning] status['pop_best_eval']: {status['pop_best_eval']} is not comparable.")
            self.has_warned_about_pop_best_eval = True

        self._steps_count += 1


class BiManualLogger(Logger):
    """
    A dummy logger that only maintains the best solution during the optimization.
    """
    def __init__(
        self, 
        searcher, 
        *, 
        interval: int = 1, 
        after_first_step: bool = False
    ):
        # Call the super constructor
        super().__init__(searcher, interval=interval, after_first_step=after_first_step)

        self.best_solution_left, self.best_solution_right = None, None
        self.best_eval_left, self.best_eval_right = float('inf'), float('inf')
        
        self.has_warned_about_pop_best_eval = False

    @property
    def best_eval(self):
        return self.best_eval_left + self.best_eval_right

    @property
    def best_solution(self):
        return torch.cat((self.best_solution_left, self.best_solution_right), dim=0) # concatenate along solution dimension

    def __call__(self, status: dict):
        # Update best value and evaluation
        if status["pop_best_eval_left"] < self.best_eval_left:
            self.best_solution_left = status["pop_best_left"].clone()
            self.best_eval_left = status["pop_best_eval_left"].clone()
        if status["pop_best_eval_right"] < self.best_eval_right:    
            self.best_solution_right = status["pop_best_right"].clone()
            self.best_eval_right = status["pop_best_eval_right"].clone()

        self._steps_count += 1


class PoseEstimationProblem(Problem):
    def __init__(
        self, model, robot_renderer, ref_mask, intr, p_local1, p_local2, stdev_init, args,
        joint_angles_lb=torch.tensor([-1.5707, -1.3963, 0.,     0.    ]),
        joint_angles_ub=torch.tensor([ 1.5707,  1.3963, 1.5707, 1.5707]),
    ):
        if not isinstance(self, BiManualPoseEstimationProblem):
            solution_length = 9 if args.symmetric_jaw else 10
        else:
            solution_length = 18 if args.symmetric_jaw else 20 

        if isinstance(self, BiManualPoseEstimationProblem):
            self.separate_loss = args.separate_loss
            self.soft_separation = args.soft_separation
        else:
            self.separate_loss = False
            self.soft_separation = False

        super().__init__(
            objective_sense="min" if not self.separate_loss else ["min", "min"], # minimize single or two objectives
            solution_length=solution_length, 
            device=model.device,
            initial_bounds=([-float('inf')] * solution_length, [float('inf')] * solution_length)
        )

        self.model = model
        self.robot_renderer = robot_renderer
        self.ref_mask = ref_mask

        self.intr = intr  
        self.p_local1 = p_local1  
        self.p_local2 = p_local2

        self.fx, self.fy, self.px, self.py = intr[0, 0].item(), intr[1, 1].item(), intr[0, 2].item(), intr[1, 2].item()
        
        self.ref_keypoints = None
        self.det_line_params = None

        self.args = args

        self.render_loss = self.args.use_render_loss
        self.dist_loss = self.args.dist_weight > 0.
        self.kpts_loss = self.args.use_pts_loss
        # self.tip_emd_loss = self.args.use_tip_emd_loss

        # assert self.tip_emd_loss is False, "The tip EMD loss is not supported currently."

        self.mse_weight = self.args.mse_weight  # weight for the MSE loss
        self.dist_weight = self.args.dist_weight  # weight for the distance loss
        self.app_weight = self.args.app_weight  # weight for the appearance loss
        self.pts_weight = self.args.pts_weight  # weight for the keypoint loss
        # self.tip_emd_weight = self.args.tip_emd_weight  # weight for the tip EMD loss

        # Convert stdev_init to lengthscales
        if torch.is_tensor(stdev_init):
            self.lengthscales = stdev_init.clone().detach()
        else:
            self.lengthscales = stdev_init

        self.pose_dim = None

        if args.downscale_factor != 1:
            self.resolution = self.model.resolution
            # assert self.resolution[0] % args.downscale_factor == 0 and self.resolution[1] % args.downscale_factor == 0
            self.render_resolution = (self.resolution[0] // args.downscale_factor, self.resolution[1] // args.downscale_factor)

        else:
            self.render_resolution = self.model.resolution

        self.joint_angles_lb = joint_angles_lb.to(self.model.device)
        self.joint_angles_ub = joint_angles_ub.to(self.model.device)

        self._prev_cTr = None
        self._prev_joint_angles = None

        print("Loss weights:")
        if self.render_loss:
            print(f"    Render loss weight: {self.mse_weight}")
            print(f"    Distance loss weight: {self.dist_weight}")
            print(f"    Appearance loss weight: {self.app_weight}")
        if self.kpts_loss:
            print(f"    Keypoint loss weight: {self.pts_weight}")

    @torch.no_grad()
    def update_problem(
        self, ref_mask, ref_keypoints, cTr_init, stdev_init
    ):
        self.ref_mask = ref_mask
        self.ref_keypoints = ref_keypoints

        if self.args.downscale_factor != 1:
            # Downscale the reference mask to the render resolution
            self.ref_mask = F.interpolate(
                self.ref_mask.float().unsqueeze(0).unsqueeze(0), 
                size=self.render_resolution, 
                mode='bilinear'
            ).squeeze(0).squeeze(0)

        # print(stdev_init)
        if torch.is_tensor(stdev_init):
            self.lengthscales = stdev_init.clone().detach()
        else:
            self.lengthscales = stdev_init

        # Compute distance map
        if self.dist_loss:
            mask = 1 - self.ref_mask.float()  # Invert the mask for the distance transform
            v, lamb, iterations = 1e10, 0.0, 2 # Use Euclidean distance transform only
            self.dist_map = FastGeodis.generalised_geodesic2d(
                torch.ones_like(mask).unsqueeze(0).unsqueeze(0),
                mask.unsqueeze(0).unsqueeze(0),
                v, 
                lamb,
                iterations
            ).squeeze().detach()

            if self.args.downscale_factor != 1:
                # Downscale the distance map to the render resolution
                self.dist_map = F.interpolate(
                    self.dist_map.unsqueeze(0).unsqueeze(0), 
                    size=self.render_resolution,
                    mode='bilinear'
                ).squeeze(0).squeeze(0)

            # # Use kornia cascaded convolution distance transform
            # self.dist_map_size = (self.ref_mask.shape[0] // 10, self.ref_mask.shape[1] // 10)
            # mask = F.interpolate(self.ref_mask.float().unsqueeze(0).unsqueeze(0), size=self.dist_map_size, mode='bilinear')
            # self.dist_map = kornia.contrib.distance_transform(mask, kernel_size=33)
            # self.dist_map = F.interpolate(self.dist_map, size=self.ref_mask.shape, mode='bilinear').squeeze(0).squeeze(0).detach() # [H, W]
            # self.dist_map *= 10.0 # scale the distance map

        # Transform the initial rotation representations
        self.cTr_init = cTr_init
        self.pose_dim = 6

        # if self.args.use_mix_angle:
        #     axis_angle = cTr_init[:3].unsqueeze(0)  # shape (1, 3)
        #     mix_angle = axis_angle_to_mix_angle(axis_angle)  # Convert to [alpha, beta, gamma] representation
        #     self.cTr_init[:3] = mix_angle.squeeze(0)  # Replace the first 3 elements with the [alpha, beta, gamma] representation

    def compute_loss(self, raw_values):
        values = raw_values * self.lengthscales # scale the values by the lengthscales
            
        cTr_batch = values[:, :self.pose_dim]  # shape (B, 6)
        joint_angles = values[:, self.pose_dim:]  # shape (B, 3) or (B, 4)
        B = cTr_batch.shape[0]

        if self.args.symmetric_jaw:
            joint_angles = torch.cat([joint_angles[:, :3], joint_angles[:, -1:]], dim=1)  # make jaws symmetric

        if self.args.use_mix_angle:
            R_batched = mix_angle_to_rotmat(cTr_batch[:, :3])  # shape (B, 3, 3)
            T_batched = cTr_batch[:, 3:]
        else:
            R_batched = kornia.geometry.conversions.axis_angle_to_rotation_matrix(cTr_batch[:, :3])  # shape (B, 3, 3)
            T_batched = cTr_batch[:, 3:]

        # Convert joint angles to the bounded representation
        # if self.args.searcher == 'CMA-ES':
        if self.args.cos_reparams:
            joint_angles_R = self.joint_angles_ub - self.joint_angles_lb
            joint_angles = self.joint_angles_lb + \
                            0.5 * joint_angles_R * (1 - torch.cos(math.pi * (joint_angles - self.joint_angles_lb) / joint_angles_R))
        else:
            joint_angles = torch.clamp(joint_angles, self.joint_angles_lb, self.joint_angles_ub)

        assert self.render_loss, "The render loss must be enabled."

        R_list, t_list = None, None

        self.ref_mask_b = self.ref_mask.unsqueeze(0).expand(
            B, self.ref_mask.shape[0], self.ref_mask.shape[1]
        )

        verts, faces, R_list, t_list = self.robot_renderer.batch_get_robot_verts_and_faces(joint_angles, ret_lndFK=True)
        pred_masks_b = self.model.render_robot_mask_batch_nvdiffrast_rotmat(
            R_batched, T_batched, verts, faces, self.robot_renderer, self.render_resolution
        ) # shape (B,H,W)

        # if self.args.downscale_factor != 1:
        #     pred_masks_b = F.interpolate(pred_masks_b.unsqueeze(1), size=self.resolution, mode='bilinear').squeeze(1)

        mse = F.mse_loss(
            pred_masks_b,
            self.ref_mask_b,
            reduction="none",
        ).mean(dim=(1, 2))

        # Compute distance loss
        if self.dist_weight > 0.:
            dist_map_ref = self.dist_map.unsqueeze(0).expand(
                B, *[-1 for _ in self.dist_map.shape]
            )  # [B, H, W]
            dist = torch.sum(
                (pred_masks_b) * (dist_map_ref), 
                dim=(1, 2)
            )

        else:
            dist = 0.0

        # Compute appearance loss
        if self.app_weight > 0.:
            sum_pred = torch.sum(pred_masks_b, dim=(1, 2))  # [B]
            sum_ref = torch.sum(self.ref_mask_b, dim=(1, 2))
            app = torch.abs(sum_pred - sum_ref) 
        else:
            app = 0.0

        if self.kpts_loss and (self.ref_keypoints is not None):
            # print("Computing keypoint loss...")
            pose_matrix_b = torch.zeros((B, 4, 4), device=self.model.device)
            pose_matrix_b[:, :3, :3] = R_batched
            pose_matrix_b[:, :3, 3] = T_batched
            pose_matrix_b[:, 3, 3] = 1.0
            
            p_img1 = get_img_coords_batch(
                self.p_local1,
                R_list[:,2,...],
                t_list[:,2,...],
                pose_matrix_b.to(joint_angles.dtype),
                self.intr,
                ret_cam_coords=False
            )
            p_img2 = get_img_coords_batch(
                self.p_local2,
                R_list[:,3,...],
                t_list[:,3,...],
                pose_matrix_b.to(joint_angles.dtype),
                self.intr,
                ret_cam_coords=False
            )

            proj_pts = torch.stack((p_img1, p_img2), dim=1)  # [B, 2, 2]

            ref_kps_2d = self.ref_keypoints.unsqueeze(0).expand(B, -1, -1)  
            pts_val = keypoint_loss_batch(proj_pts, ref_kps_2d)  # [B]

        else:
            pts_val = 0.0

        loss = (
            1. * (
                self.mse_weight * mse
                + self.dist_weight * dist
                + self.app_weight * app
            )
            + self.pts_weight * pts_val 
        )

        if torch.any(torch.isnan(loss)):
            loss[torch.isnan(loss)] = float('inf')
            # print("[Warning] NaN loss encountered, setting to inf.")
            # # Print the inputs that caused NaN
            # nan_indices = torch.isnan(loss).nonzero(as_tuple=True)[0]
            # for idx in nan_indices:
            #     print(f"NaN loss for input: cTr = {cTr_batch[idx]}, joint_angles = {joint_angles[idx]}")

        if torch.any(torch.isnan(cTr_batch)) or torch.any(torch.isnan(joint_angles)):
            loss[torch.isnan(cTr_batch).any(dim=1) | torch.isnan(joint_angles).any(dim=1)] = float('inf') # set loss to inf if inputs are NaN
            # print("[Warning] NaN in cTr or joint angles.")
            # nan_indices = torch.unique(torch.cat((
            #     torch.isnan(cTr_batch).any(dim=1).nonzero(as_tuple=True)[0],
            #     torch.isnan(joint_angles).any(dim=1).nonzero(as_tuple=True)[0]
            # )))
            # for idx in nan_indices:
            #     print(f"NaN input: cTr = {cTr_batch[idx]}, joint_angles = {joint_angles[idx]}") 

        # print(loss)

        return loss

    def _evaluate_batch(self, batch: SolutionBatch) -> SolutionBatch:
        values = batch.values.clone()  # extract the values

        losses = self.compute_loss(values)  # shape (B,)

        batch.set_evals(losses)  # set the evaluations for the batch


class BiManualPoseEstimationProblem(PoseEstimationProblem):
    @torch.no_grad()
    def update_problem(
        self, ref_mask, ref_keypoints, cTr_init, stdev_init
    ):
        """
        ref_mask: [2, H, W] tensor, masks for left and right arms
        ref_keypoints: [2, 2, 2] tensor, keypoints for left and right arms
        """
        if self.args.downscale_factor != 1:
            # Downscale the reference mask to the render resolution
            ref_mask = F.interpolate(
                ref_mask.float().unsqueeze(0), 
                size=self.render_resolution, 
                mode='bilinear'
            ).squeeze(0) # [2, H, W]

        self.ref_mask = torch.max(ref_mask[0], ref_mask[1])  # Combine the two masks
        self.ref_keypoints_left = ref_keypoints[0]
        self.ref_keypoints_right = ref_keypoints[1]

        # Compute distance map for left and right masks
        if self.separate_loss:
            if self.soft_separation or self.dist_loss:
                v, lamb, iterations = 1e10, 0.0, 1 # Use Euclidean distance transform only
                mask_left = 1 - ref_mask[0]  # Invert the mask for the distance transform
                self.dist_map_left = FastGeodis.generalised_geodesic2d(
                    torch.ones_like(mask_left).unsqueeze(0).unsqueeze(0),
                    mask_left.unsqueeze(0).unsqueeze(0),
                    v, 
                    lamb,
                    iterations
                ).squeeze().detach()
                mask_right = 1 - ref_mask[1]  # Invert the mask for the distance transform
                self.dist_map_right = FastGeodis.generalised_geodesic2d(
                    torch.ones_like(mask_right).unsqueeze(0).unsqueeze(0),
                    mask_right.unsqueeze(0).unsqueeze(0),
                    v,
                    lamb,
                    iterations
                ).squeeze().detach()

            if self.soft_separation:
                tol = 1. # tolerance in pixels
                self.mask_left = (self.dist_map_left <= self.dist_map_right + tol).to(self.ref_mask.dtype).to(self.ref_mask.device)
                self.mask_right = (self.dist_map_right <= self.dist_map_left + tol).to(self.ref_mask.dtype).to(self.ref_mask.device)

                self.ref_mask = torch.stack((self.ref_mask, self.ref_mask), dim=-1) # [H, W, 2]
                self.dist_map = torch.stack((self.dist_map_left, self.dist_map_right), dim=-1)  # [H, W, 2]
                self.partition_mask = torch.stack((self.mask_left, self.mask_right), dim=-1)  # [H, W, 2]
            
            else:
                self.ref_mask = ref_mask.permute(1, 2, 0)  # [H, W, 2]
                if self.dist_loss:
                    self.dist_map = torch.stack((self.dist_map_left, self.dist_map_right), dim=-1)  # [H, W, 2]


        elif self.dist_loss:
            v, lamb, iterations = 1e10, 0.0, 2 # Use Euclidean distance transform only
            mask = 1 - self.ref_mask.float()  # Invert the mask for the distance transform
            self.dist_map = FastGeodis.generalised_geodesic2d(
                torch.ones_like(mask).unsqueeze(0).unsqueeze(0),
                mask.unsqueeze(0).unsqueeze(0),
                v, 
                lamb,
                iterations
            ).squeeze().detach()

            if self.args.downscale_factor != 1:
                # Downscale the distance map to the render resolution
                self.dist_map = F.interpolate(
                    self.dist_map.unsqueeze(0).unsqueeze(0), 
                    size=self.render_resolution,
                    mode='bilinear'
                ).squeeze(0).squeeze(0)

        # print(stdev_init)
        if torch.is_tensor(stdev_init):
            self.lengthscales = stdev_init.clone().detach()
        else:
            self.lengthscales = stdev_init

        # Transform the initial rotation representations
        self.cTr_init = cTr_init
        self.pose_dim = 6

    def compute_loss(self, raw_values):
        # print("Raw values:", raw_values.shape)
        # print("Lengthscales:", self.lengthscales.shape)
        values = raw_values * self.lengthscales # scale the values by the lengthscales, shape (B, 18) or (B, 20)
        values = values.view(-1, 2, values.shape[1] // 2)  # shape (B, 2, 9) or (B, 2, 10)
        cTr_batch = values[:, :, :self.pose_dim] # shape (B, 2, 6)
        joint_angles = values[:, :, self.pose_dim:]  # shape (B, 2, 3) or (B, 2, 4)
        # print("cTr_batch shape:", cTr_batch.shape)
        # print("joint_angles shape:", joint_angles.shape)

        B = cTr_batch.shape[0]

        if self.args.symmetric_jaw:
            joint_angles = torch.cat([joint_angles[:, :, :3], joint_angles[:, :, -1:]], dim=2)  # make jaws symmetric

        if self.args.use_mix_angle:
            R_batched = mix_angle_to_rotmat(cTr_batch[:, :, :3].view(-1, 3)).view(-1, 2, 3, 3)  # shape (B, 2, 3, 3)
            T_batched = cTr_batch[:, :, 3:] # shape (B, 2, 3)
        else:
            R_batched = kornia.geometry.conversions.axis_angle_to_rotation_matrix(cTr_batch[:, :, :3].view(-1, 3)).view(-1, 2, 3, 3)  # shape (B, 2, 3, 3)
            T_batched = cTr_batch[:, :, 3:] # shape (B, 2, 3)

        # Convert joint angles to the bounded representation
        # if self.args.searcher == 'CMA-ES':
        if self.args.cos_reparams:
            joint_angles_R = self.joint_angles_ub - self.joint_angles_lb
            joint_angles = self.joint_angles_lb + \
                            0.5 * joint_angles_R * (1 - torch.cos(math.pi * (joint_angles - self.joint_angles_lb) / joint_angles_R))
        else:
            joint_angles = torch.clamp(joint_angles, self.joint_angles_lb, self.joint_angles_ub)

        assert self.render_loss, "The render loss must be enabled."

        R_list, t_list = None, None

        if self.separate_loss and not self.soft_separation:
            if self.args.share_depth_buffer:
                # Render different arms in different channels
                verts, faces, R_list, t_list, color = self.robot_renderer.batch_get_robot_verts_and_faces(joint_angles, ret_lndFK=True, bi_manual=True, ret_col=True)
                pred_masks_b = self.model.render_robot_mask_batch_nvdiffrast_rotmat(
                    R_batched, T_batched, verts, faces, self.robot_renderer, self.render_resolution, bi_manual=True, color=color
                ) # shape (B,H,W,2)

            else:
                # Render different arms as different samples in the batch
                joint_angles_flat = joint_angles.view(-1, joint_angles.shape[-1])  # shape (B*2, 3) or (B*2, 4)
                R_batched_flat = R_batched.view(-1, 3, 3)  # shape (B*2, 3, 3)
                T_batched_flat = T_batched.view(-1, 3)  # shape (B*2, 3)

                verts, faces, R_list_flat, t_list_flat = self.robot_renderer.batch_get_robot_verts_and_faces(joint_angles_flat, ret_lndFK=True, bi_manual=False)
                pred_masks_b_flat = self.model.render_robot_mask_batch_nvdiffrast_rotmat(
                    R_batched_flat, T_batched_flat, verts, faces, self.robot_renderer, self.render_resolution, bi_manual=False
                ) # shape (B*2,H,W)

                pred_masks_b = pred_masks_b_flat.view(B, 2, pred_masks_b_flat.shape[1], pred_masks_b_flat.shape[2]).permute(0, 2, 3, 1)  # shape (B, H, W, 2)
                R_list = R_list_flat.view(B, 2, R_list_flat.shape[1], R_list_flat.shape[2], R_list_flat.shape[3])  # shape (B, 2, N, 3, 3)
                t_list = t_list_flat.view(B, 2, t_list_flat.shape[1], t_list_flat.shape[2])  # shape (B, 2, N, 3)
        else:
            verts, faces, R_list, t_list = self.robot_renderer.batch_get_robot_verts_and_faces(joint_angles, ret_lndFK=True, bi_manual=True)
            pred_masks_b = self.model.render_robot_mask_batch_nvdiffrast_rotmat(
                R_batched, T_batched, verts, faces, self.robot_renderer, self.render_resolution, bi_manual=True
            ) # shape (B,H,W)

        if self.separate_loss:
            # --------------------------------------------------
            # Shapes:
            #   pred_masks_b   : [B, H, W]
            #   ref_mask       : [H, W, 2]
            #   partition_mask : [H, W, 2]
            #   dist_map       : [H, W, 2]
            # --------------------------------------------------

            B = pred_masks_b.shape[0]
            ref_mask_b = self.ref_mask.unsqueeze(0).expand(B, -1, -1, -1)      # [B, H, W, 2]

            if self.soft_separation:
                # Expand reference + masks once
                part_mask_b = self.partition_mask.unsqueeze(0)                    # [1, H, W, 2]

                # Mask predictions and references
                pred_masked = pred_masks_b.unsqueeze(-1) * part_mask_b             # [B, H, W, 2]
                ref_masked  = ref_mask_b   * part_mask_b                           # [B, H, W, 2]
            
            else:
                pred_masked = pred_masks_b                          # [B, H, W, 2]
                ref_masked = ref_mask_b

            # --------------------------------------------------
            # MSE loss
            # --------------------------------------------------
            mse = F.mse_loss(
                pred_masked,
                ref_masked,
                reduction="none"
            ).mean(dim=(1, 2))                                                  # [B, 2]

            mse_left  = mse[:, 0]
            mse_right = mse[:, 1]

            # --------------------------------------------------
            # Distance loss
            # --------------------------------------------------
            if self.dist_weight > 0.:
                dist_map_b = self.dist_map.unsqueeze(0)                         # [1, H, W, 2]

                dist = torch.sum(
                    pred_masked * dist_map_b,
                    dim=(1, 2)
                )                                                                # [B, 2]

                dist_left  = dist[:, 0]
                dist_right = dist[:, 1]
            else:
                dist_left = dist_right = 0.0

            # --------------------------------------------------
            # Appearance loss
            # --------------------------------------------------
            if self.app_weight > 0.:
                sum_pred = torch.sum(pred_masked, dim=(1, 2))                   # [B, 2]
                sum_ref  = torch.sum(ref_masked,  dim=(1, 2))                   # [B, 2]

                app = torch.abs(sum_pred - sum_ref)                              # [B, 2]

                app_left  = app[:, 0]
                app_right = app[:, 1]
            else:
                app_left = app_right = 0.0

        else:
            self.ref_mask_b = self.ref_mask.unsqueeze(0).expand(
                B, self.ref_mask.shape[0], self.ref_mask.shape[1]
            )

            mse = F.mse_loss(
                pred_masks_b,
                self.ref_mask_b,
                reduction="none",
            ).mean(dim=(1, 2))

            # Compute distance loss
            if self.dist_weight > 0.:
                dist_map_ref = self.dist_map.unsqueeze(0).expand(
                    B, *[-1 for _ in self.dist_map.shape]
                )  # [B, H, W]
                dist = torch.sum(
                    (pred_masks_b) * (dist_map_ref), 
                    dim=(1, 2)
                )
            else:
                dist = 0.0

            # Compute appearance loss
            if self.app_weight > 0.:
                sum_pred = torch.sum(pred_masks_b, dim=(1, 2))  # [B]
                sum_ref = torch.sum(self.ref_mask_b, dim=(1, 2))
                app = torch.abs(sum_pred - sum_ref) 
            else:
                app = 0.0

        if self.kpts_loss:
            if self.ref_keypoints_left is not None:
                # Compute keypoint loss for left and right arms separately
                pose_matrix_b_left = torch.zeros((B, 4, 4), device=self.model.device)
                pose_matrix_b_left[:, :3, :3] = R_batched[:, 0, ...]
                pose_matrix_b_left[:, :3, 3] = T_batched[:, 0, ...]
                pose_matrix_b_left[:, 3, 3] = 1.0

                p_img1_left = get_img_coords_batch(
                    self.p_local1,
                    R_list[:,0,2,...],
                    t_list[:,0,2,...],
                    pose_matrix_b_left.to(joint_angles.dtype),
                    self.intr,
                    ret_cam_coords=False
                )
                p_img2_left = get_img_coords_batch(
                    self.p_local2,
                    R_list[:,0,3,...],
                    t_list[:,0,3,...],
                    pose_matrix_b_left.to(joint_angles.dtype),
                    self.intr,
                    ret_cam_coords=False
                )
                proj_pts_left = torch.stack((p_img1_left, p_img2_left), dim=1)  # [B, 2, 2]
                ref_kps_2d_left = self.ref_keypoints_left.unsqueeze(0).expand(B, -1, -1)
                pts_val_left = keypoint_loss_batch(proj_pts_left, ref_kps_2d_left)  # [B]
            else:
                pts_val_left = 0.0

            if self.ref_keypoints_right is not None:
                pose_matrix_b_right = torch.zeros((B, 4, 4), device=self.model.device)
                pose_matrix_b_right[:, :3, :3] = R_batched[:, 1, ...]
                pose_matrix_b_right[:, :3, 3] = T_batched[:, 1, ...]
                pose_matrix_b_right[:, 3, 3] = 1.0
                p_img1_right = get_img_coords_batch(
                    self.p_local1,
                    R_list[:,1,2,...],
                    t_list[:,1,2,...],
                    pose_matrix_b_right.to(joint_angles.dtype),
                    self.intr,
                    ret_cam_coords=False
                )
                p_img2_right = get_img_coords_batch(
                    self.p_local2,
                    R_list[:,1,3,...],
                    t_list[:,1,3,...],
                    pose_matrix_b_right.to(joint_angles.dtype),
                    self.intr,
                    ret_cam_coords=False
                )
                proj_pts_right = torch.stack((p_img1_right, p_img2_right), dim=1)  # [B, 2, 2]
                ref_kps_2d_right = self.ref_keypoints_right.unsqueeze(0).expand(B, -1, -1)
                pts_val_right = keypoint_loss_batch(proj_pts_right, ref_kps_2d_right)  # [B]
            else:
                pts_val_right = 0.0

        else:
            pts_val_left, pts_val_right = 0.0, 0.0

        if self.separate_loss:
            loss_left = (
                1. * (
                    self.mse_weight * mse_left
                    + self.dist_weight * dist_left
                    + self.app_weight * app_left
                )
                + self.pts_weight * pts_val_left
            )

            loss_right = (
                1. * (
                    self.mse_weight * mse_right
                    + self.dist_weight * dist_right
                    + self.app_weight * app_right
                )
                + self.pts_weight * pts_val_right
            )

            if torch.any(torch.isnan(loss_left)):
                loss_left[torch.isnan(loss_left)] = float('inf')
            if torch.any(torch.isnan(loss_right)):
                loss_right[torch.isnan(loss_right)] = float('inf')

            # if torch.any(torch.isnan(cTr_batch)) or torch.any(torch.isnan(joint_angles)):
            #     loss_left[torch.isnan(cTr_batch).any(dim=1) | torch.isnan(joint_angles).any(dim=1)] = float('inf')
            #     loss_right[torch.isnan(cTr_batch).any(dim=1) | torch.isnan(joint_angles).any(dim=1)] = float('inf')

            return torch.stack((loss_left, loss_right), dim=1) # shape (B, 2)

        else:
            loss = (
                1. * (
                    self.mse_weight * mse
                    + self.dist_weight * dist
                    + self.app_weight * app
                )
                + self.pts_weight * (pts_val_left + pts_val_right)
            )

            if torch.any(torch.isnan(loss)):
                loss[torch.isnan(loss)] = float('inf')

            # if torch.any(torch.isnan(cTr_batch)) or torch.any(torch.isnan(joint_angles)):
            #     loss[torch.isnan(cTr_batch).any(dim=1) | torch.isnan(joint_angles).any(dim=1)] = float('inf')

            return loss  # shape (B,)


class GradientDescentSearcher(SearchAlgorithm):
    def __init__(self, problem: Problem, stdev_init=1., center_init=None, popsize=None, random_initialization=False):
        SearchAlgorithm.__init__(
            self,
            problem=problem, 
            pop_best=self._get_pop_best,
            pop_best_eval=self._get_pop_best_eval,
            pop_best_eval_left=self._get_pop_best_eval_left,
            pop_best_eval_right=self._get_pop_best_eval_right,
            pop_best_left=self._get_pop_best_left,
            pop_best_right=self._get_pop_best_right,
        )

        # Turn on antialiasing if using NvDiffRast renderer
        if self.problem.model.args.use_nvdiffrast and not self.problem.model.use_antialiasing:
            print("[Antialiasing is not enabled in the NvDiffRast renderer. This may lead to inaccurate gradients.]")
            print("[Enabling antialiasing for better gradients.]")
            self.problem.model.use_antialiasing = True # use antialiasing for better gradients

        if random_initialization:
            # Sample from Gaussian(center_init, stdev_init) and use the best candidate as the starting point
            with torch.no_grad():
                N_candidates = 100
                candidates = torch.randn((N_candidates, self.problem.solution_length), device=self.problem.device) * stdev_init + center_init.unsqueeze(0)  # [100, D]
                losses = self.problem.compute_loss(candidates)  # [100]
                best_idx = torch.argmin(losses)
                center_init = candidates[best_idx]  # [D]

        self.vars = center_init.detach().clone().unsqueeze(0).requires_grad_(True)
        self.optimizer = torch.optim.Adam([self.vars], lr=1e-1)

        # Dummy data for the logger to process
        self.batch = self.problem.generate_batch(1)
        self._pop_best: Optional[Solution] = None

        self._pop_best_eval_left = float('inf')
        self._pop_best_eval_right = float('inf')
        self._pop_best_left = None
        self._pop_best_right = None

    def _get_pop_best_eval_left(self):
        return self._pop_best_eval_left

    def _get_pop_best_eval_right(self):
        return self._pop_best_eval_right

    def _get_pop_best_left(self):
        return self._pop_best_left

    def _get_pop_best_right(self):
        return self._pop_best_right

    def _get_pop_best(self):
        return self._pop_best

    def _get_pop_best_eval(self):
        return self._pop_best.evals[0].item()

    def _step(self):
        # Back-propagate the loss
        self.optimizer.zero_grad()

        loss = self.problem.compute_loss(self.vars).squeeze()
        
        if loss.shape[0] == 2:
            # Bi-manual case, sum the two losses
            loss_sum = loss[0] + loss[1]

            loss_sum.backward()

            # Update pop best left and right
            if loss[0] < self._pop_best_eval_left:
                self._pop_best_left = self.vars[:self.problem.solution_length//2].detach().clone()
                self._pop_best_eval_left = loss[0].detach().clone()
            if loss[1] < self._pop_best_eval_right:
                self._pop_best_right = self.vars[self.problem.solution_length//2:].detach().clone()
                self._pop_best_eval_right = loss[1].detach().clone()

            self.batch.set_values(self.vars.detach().clone())
            self.batch.set_evals(loss.unsqueeze(0).detach().clone())
            self._pop_best = self.batch[0]
        
        else:
            loss.backward()
            # Update dummy data
            self.batch.set_values(self.vars.detach().clone())
            self.batch.set_evals(loss.unsqueeze(0).detach().clone())
            self._pop_best = self.batch[0]

        self.optimizer.step()


class Tracker:
    def __init__(
        self, model, robot_renderer, init_cTr, init_joint_angles, 
        num_iters=5, intr=None, p_local1=None, p_local2=None, 
        stdev_init=1., searcher="CMA-ES", args=None
    ):
        self.model = model
        self.robot_renderer = robot_renderer

        self._prev_cTr = init_cTr
        self._prev_joint_angles = init_joint_angles
        self.num_iters = num_iters

        self.num_iters = num_iters # number of iterations for optimization

        self.intr = intr  
        self.p_local1 = p_local1  
        self.p_local2 = p_local2 

        self.args = args

        self.fx, self.fy, self.px, self.py = intr[0, 0].item(), intr[1, 1].item(), intr[0, 2].item(), intr[1, 2].item()

        if args.symmetric_jaw:
            stdev_init = stdev_init[:9]  # Use 9 dimensions if symmetric jaws
        self.stdev_init = stdev_init  # Initial standard deviation for the optimization

        if self.model.args.use_nvdiffrast is not True:
            print("[Warning] NvDiffRast renderer is not enabled. Automatically enabling it for better performance.]")
            self.model.args.use_nvdiffrast = True

        if self.args.searcher != "Gradient":
            print("[Antialiasing is not enabled in the NvDiffRast renderer for black box optimization.]")
            self.model.use_antialiasing = False

        self.problem = PoseEstimationProblem(model, robot_renderer, None, intr, p_local1, p_local2, self.stdev_init, args)

        if args.use_filter:
            if args.filter_option == "OneEuro":
                self.filter = OneEuroFilter(
                    # f_min=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), # minimum cutoff frequency for each dimension (the larger, the less smooth)
                    f_min=0.5,
                    alpha_d=0.3, # smoothing factor for the derivative
                    beta=0.3, # speed coefficient
                    kappa=0.6
                )
            elif args.filter_option == "OneEuro_orig":
                self.filter = OneEuroFilter(
                    f_min=0.8, # higher to reduce lag
                    alpha_d=0.3, # smoothing factor for the derivative
                    beta=0.9, # higher to reduce lag
                    kappa=0.
                )
            elif args.filter_option == "Kalman":
                ms_factor = 1. if not args.searcher == "Gradient" else 0.01 # reduce measurement noise for gradient-based searcher as it produces less noisy results
                self.filter = KalmanFilter(
                    process_noise_pos=np.array([2e-5, 1e-4, 2e-5, 2e-5, 2e-5, 2e-5, 1e-4, 1e-4, 1e-4, 1e-4]),      # scalar or (D,)
                    process_noise_vel=np.array([2e-4, 1e-3, 2e-4, 2e-4, 2e-4, 2e-4, 1e-3, 1e-3, 1e-3, 1e-3]),      # scalar or (D,)
                    measurement_noise=np.array([2e-3, 1e-2, 2e-3, 2e-3, 2e-3, 2e-3, 5e-3, 5e-3, 5e-3, 5e-3]) * ms_factor * 0.5,      # scalar or (D,)
                )
            else:
                raise ValueError(f"Unknown filter option: {args.filter_option}")

        optimizer_dict = {
            "CMA-ES": CMAES_cus, # customized CMA-ES implementation
            "Gradient": GradientDescentSearcher,
            "XNES": XNES,
        }
        self.optimizer = optimizer_dict[searcher]

        # Transform the intial cTr to Euler angle if required
        if self.args.use_mix_angle:
            print("[Using transformed angle space for optimization.]")
            self._prev_cTr[:3] = axis_angle_to_mix_angle(self._prev_cTr[:3].unsqueeze(0)).squeeze(0)
        else:
            print("[Using axis-angle space for optimization.]")

        self.sobol = SobolEngine(dimension=self.problem.solution_length, scramble=True) # Sobol sequence generator for low-discrepancy sampling

        if args.use_contour_tip_net:
            # self.contour_tip_net = ContourTipNet(
            #     feature_dim=6,
            #     max_len=200
            # ).to(self.model.device)
            # self.contour_tip_net.load_state_dict(
            #     th.load(
            #         args.contour_tip_net_path, 
            #         map_location=self.model.device
            #     )
            # )
            # self.contour_tip_net.eval()
            self.tip_2d_net = Tip2DNet().to(self.model.device)
            self.tip_2d_net.load_state_dict(
                th.load(
                    args.contour_tip_net_path, 
                    map_location=self.model.device
                )
            )
            self.tip_2d_net.eval()
            self.tip_2d_net.compile()

    def overlay_mask(self, ref_mask, pred_mask, ref_pts=None, proj_pts=None):
        """
        Overlay the predicted mask on the reference mask for visualization.
        """
        # Convert masks to grayscale images
        ref_mask = ref_mask.float().cpu().numpy()
        pred_mask = pred_mask.float().cpu().numpy()

        w, h = ref_mask.shape[1], ref_mask.shape[0]
        
        # Create a color overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)  # Create an empty overlay image
        overlay[..., 0] = (ref_mask * 255).astype(np.uint8) 
        overlay[..., 2] = (pred_mask * 255).astype(np.uint8)  

        if ref_pts != None:
            center_ref_pt = th.mean(ref_pts, dim=0)
            for ref_pt in ref_pts:
                u_ref, v_ref = int(ref_pt[0]), int(ref_pt[1])
                cv2.circle(
                    overlay,
                    (u_ref, v_ref),
                    radius=5,
                    color=(255, 0.6 * 255, 0.1 * 255),
                    thickness=-1,
                )  

            u_ref, v_ref = int(center_ref_pt[0]), int(center_ref_pt[1])
            cv2.circle(
                overlay,
                (u_ref, v_ref),
                radius=5,
                color=(255, 0.6 * 255, 0.1 * 255),
                thickness=-1,
            )  

        if proj_pts is not None:
            center_proj_pt = th.mean(proj_pts, dim=0)
            for proj_pt in proj_pts.squeeze():
                u_proj, v_proj = int(proj_pt[0].item()), int(proj_pt[1].item())
                cv2.circle(
                    overlay, (u_proj, v_proj), radius=5, color=(255*0.1, 255*0.6, 255), thickness=-1
                )  #

            u_ref, v_ref = int(center_proj_pt[0]), int(center_proj_pt[1])
            cv2.circle(
                overlay, (u_ref, v_ref), radius=5, color=(255*0.1, 255*0.6, 255), thickness=-1
            ) 

        return overlay

    def project_keypoints(self, cTr, joint_angles):
        pose_matrix = self.model.cTr_to_pose_matrix(cTr.unsqueeze(0)).squeeze()
        R_list, t_list = lndFK(joint_angles)
        R_list = R_list.to(self.model.device)
        t_list = t_list.to(self.model.device)
        p_img1 = get_img_coords(
            self.p_local1,
            R_list[2],
            t_list[2],
            pose_matrix.to(joint_angles.dtype),
            self.intr,
        )
        p_img2 = get_img_coords(
            self.p_local2,
            R_list[3],
            t_list[3],
            pose_matrix.to(joint_angles.dtype),
            self.intr,
        )
        return th.stack([p_img1, p_img2], dim=0)

    def visualize(self, mask, cTr, joint_angles, ref_keypoints):
        # Render the predicted mask for visualization
        robot_mesh = self.robot_renderer.get_robot_mesh(joint_angles)
        rendered_mask = self.model.render_single_robot_mask(cTr, robot_mesh, self.robot_renderer, self.problem.render_resolution).squeeze(0)

        if self.args.downscale_factor != 1:
            rendered_mask = F.interpolate(rendered_mask.unsqueeze(0).unsqueeze(0), size=self.problem.resolution, mode='bilinear').squeeze(0).squeeze(0)

        # Project keypoints
        proj_keypoints = self.project_keypoints(cTr, joint_angles)

        return self.overlay_mask(
            mask.detach(), 
            rendered_mask.detach(),
            ref_pts=ref_keypoints,
            proj_pts=proj_keypoints,
        )
    
    def track_frame(self, ref_mask, joint_angles, is_init=False, keypoints=None):
        if is_init:
            # Initialization settings
            use_dist_loss = self.problem.dist_loss
            dist_weight = self.problem.dist_weight
            kpts_weight = self.problem.pts_weight

            self.problem.dist_loss = True # enable distance loss during initialization
            self.problem.dist_weight = 12e-7 # distance weight during initialization
            self.problem.pts_weight = 5e-3 # increase the keypoint weight during initialization
    
        # # Update the problem with the new inputs
        # if ref_mask.shape[0] != self.problem.resolution[0] or ref_mask.shape[1] != self.problem.resolution[1]:
        #     ref_mask = F.interpolate(
        #         ref_mask.float().unsqueeze(0).unsqueeze(0), 
        #         size=self.problem.resolution, 
        #         mode='bilinear'
        #     ).squeeze(0).squeeze(0)

        # Detect keypoints from the reference mask if using contour tip net
        if keypoints is not None: 
            ref_keypoints = keypoints
        elif self.args.use_contour_tip_net:
            with torch.no_grad():
                ref_keypoints = detect_keypoints_2d(
                    model=self.tip_2d_net,
                    mask=ref_mask,
                )
        else:
            ref_keypoints = None
        
        self.problem.update_problem(
            ref_mask, ref_keypoints, self._prev_cTr.clone(), self.stdev_init.clone()
        )

        # Initialize the solution with the previous cTr and joint angles
        cTr = self.problem.cTr_init
        try: 
            joint_angles = self._prev_joint_angles.clone() if self.args.use_prev_joint_angles else joint_angles.clone()
            # if self.args.use_prev_joint_angles:
            #     joint_angles = self._prev_joint_angles.clone()
            # else:
            #     joint_angles = joint_angles.clone()
            #     # Flip both wrist yaw and wrist pitch if they are closer to the previous angles after flipping (to handle pose ambiguity)
            #     _prev_wrist_pitch_yaw = self._prev_joint_angles[:2]
            #     _curr_wrist_pitch_yaw = joint_angles[:2]
            #     flipped_wrist_pitch_yaw = -_curr_wrist_pitch_yaw
            #     if torch.norm(flipped_wrist_pitch_yaw - _prev_wrist_pitch_yaw) < torch.norm(_curr_wrist_pitch_yaw - _prev_wrist_pitch_yaw):
            #         joint_angles[:2] = flipped_wrist_pitch_yaw
        except:
            raise ValueError(f"Error in cloning joint angles. joint_angles: {joint_angles}, turn on --use_prev_joint_angles to use previous joint angles as initialization.")
        
        if self.args.cos_reparams:
            joint_angles_R = self.problem.joint_angles_ub - self.problem.joint_angles_lb
            joint_angles = self.problem.joint_angles_lb + \
                        joint_angles_R / math.pi * \
                        torch.acos(1 - 2 * (joint_angles - self.problem.joint_angles_lb) / joint_angles_R)
            
        if self.args.symmetric_jaw:
            center_init = torch.cat([cTr, joint_angles[:3]], dim=0)
        else:
            center_init = torch.cat([cTr, joint_angles], dim=0)

        kwargs = dict(
            problem=self.problem,
            stdev_init=1.,
            center_init=center_init / self.problem.lengthscales,
            popsize=self.args.popsize if not is_init else min(self.args.popsize, 30),
            sobol=self.sobol,
        )
        sig = inspect.signature(self.optimizer.__init__)
        accepted = set(sig.parameters.keys())
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k in accepted
        }

        searcher = self.optimizer(**filtered_kwargs)
        logger = DummyLogger(searcher, interval=1, after_first_step=False)

        searcher.run(self.num_iters if not is_init else max(self.args.final_iters, self.num_iters))

        with torch.no_grad():
            # Extract the best solution and evaluation from the logger
            best_solution = logger.best_solution

            best_solution = best_solution * self.problem.lengthscales

            cTr, joint_angles = best_solution[:self.problem.pose_dim], best_solution[self.problem.pose_dim:]
            loss = logger.best_eval

            if self.args.symmetric_jaw:
                joint_angles = torch.cat([joint_angles[:3], joint_angles[-1:]], dim=0)  # make jaws symmetric

            # joint_angles = torch.clamp(joint_angles, self.problem.joint_angles_lb, self.problem.joint_angles_ub)
            # if self.args.searcher == 'CMA-ES':
            if self.args.cos_reparams:
                joint_angles = self.problem.joint_angles_lb + \
                        0.5 * joint_angles_R * (1 - torch.cos(math.pi * (joint_angles - self.problem.joint_angles_lb) / joint_angles_R))
            else:
                joint_angles = torch.clamp(joint_angles, self.problem.joint_angles_lb, self.problem.joint_angles_ub)

            # Filter the results via a low-pass filter
            if self.args.use_filter:
                full_state = torch.cat([cTr, joint_angles], dim=0).cpu().numpy()

                if is_init:
                    self.filter.reset(full_state)
                else:
                    # np.unwrap rotation angles before filtering
                    for j in range(3):
                        prev_cTr_np = self._prev_cTr[j].cpu().numpy()
                        curr_cTr_np = full_state[j]
                        unwrapped_cTr_np = np.unwrap(
                            np.array([prev_cTr_np, curr_cTr_np]), axis=0
                        )[1]
                        full_state[j] = unwrapped_cTr_np

                        self.filter.update(full_state)

                    filtered_state = self.filter.get_x_hat()
                    cTr = torch.from_numpy(filtered_state[:self.problem.pose_dim]).to(self.model.device).type(cTr.dtype)
                    joint_angles = torch.from_numpy(filtered_state[self.problem.pose_dim:]).to(self.model.device).type(joint_angles.dtype)

                self._prev_cTr = cTr.detach().clone()
                self._prev_joint_angles = joint_angles.detach().clone()

            if is_init:
                self.problem.dist_loss = use_dist_loss # restore distance loss
                self.problem.dist_weight = dist_weight # restore distance weight
                self.problem.pts_weight = kpts_weight # restore pts weight

            if self.args.use_mix_angle:
                cTr[:3] = mix_angle_to_axis_angle(cTr[:3].unsqueeze(0)).squeeze(0)

        return cTr, joint_angles, loss
        
    def track_sequence(
        self, mask_lst, joint_angles_lst, ref_keypoints_lst, det_line_params_lst, visualization=False
    ):
        frame_num = mask_lst.shape[0]  # number of frames
        cTr_seq = []
        joint_angles_seq = []
        loss_seq = []
        time_seq = []
        overlay_seq = []

        pbar = tqdm.tqdm(range(frame_num), desc="Tracking frames")
        for i in pbar:
            start_time = time.time()

            if i == 0:
                # Initialization settings
                use_dist_loss = self.problem.dist_loss
                dist_weight = self.problem.dist_weight
                kpts_weight = self.problem.pts_weight

                self.problem.dist_loss = True # enable distance loss during initialization
                self.problem.dist_weight = 12e-7 # distance weight during initialization
                self.problem.pts_weight = 5e-3 # increase the keypoint weight during initialization

            ref_mask = mask_lst[i].to(self.model.device)  # shape (H, W)
            if ref_mask.shape[0] != self.problem.resolution[0] or ref_mask.shape[1] != self.problem.resolution[1]:
                ref_mask = F.interpolate(
                    ref_mask.float().unsqueeze(0).unsqueeze(0), 
                    size=self.problem.resolution, 
                    mode='bilinear'
                ).squeeze(0).squeeze(0)

            joint_angles = joint_angles_lst[i]
            ref_keypoints = ref_keypoints_lst[i][:2]

            with torch.no_grad():
                if i != 0 and self.args.use_contour_tip_net:
                    # # print(self.contour_tip_net, ref_mask.shape)
                    # ref_keypoints = detect_keypoints(
                    #     model=self.contour_tip_net,
                    #     mask=ref_mask,
                    # ) 
                    # # print(f"Detected keypoints: {ref_keypoints}")
                    ref_keypoints = detect_keypoints_2d(
                        model=self.tip_2d_net,
                        mask=ref_mask,
                    )

            stdev_init = self.stdev_init.clone()
            # if ref_keypoints is None and self.problem.kpts_loss:
            #     stdev_init[6:9] = 5e-2 # decrease the stdev for joint angles if no keypoints are available
            #     stdev_init[9:] = 1e-2

            self.problem.update_problem(
                ref_mask, ref_keypoints, self._prev_cTr.clone(), stdev_init
            )

            # Initialize the solution with the previous cTr and joint angles
            cTr = self.problem.cTr_init 
            joint_angles = self._prev_joint_angles.clone() if self.args.use_prev_joint_angles else joint_angles.clone()
            # if self.args.use_prev_joint_angles:
            #     joint_angles = self._prev_joint_angles.clone()
            # else:
            #     joint_angles = joint_angles.clone()
            #     # Flip both wrist yaw and wrist pitch if they are closer to the previous angles after flipping (to handle pose ambiguity)
            #     _prev_wrist_pitch_yaw = self._prev_joint_angles[:2]
            #     _curr_wrist_pitch_yaw = joint_angles[:2]
            #     flipped_wrist_pitch_yaw = -_curr_wrist_pitch_yaw
            #     if torch.norm(flipped_wrist_pitch_yaw - _prev_wrist_pitch_yaw) < torch.norm(_curr_wrist_pitch_yaw - _prev_wrist_pitch_yaw):
            #         joint_angles[:2] = flipped_wrist_pitch_yaw

            debug_joint_angles_lst = []

            # print(f"Frame {i}: Initial joint angles: {joint_angles}")
            debug_joint_angles_lst.append(joint_angles.clone())

            # Need to make sure joint angles are bounded before transformation
            joint_angles = torch.clamp(joint_angles, self.problem.joint_angles_lb, self.problem.joint_angles_ub)

            # Reverse transform of joint angles (instead of sigmoid)
            # if self.args.searcher == 'CMA-ES':
            if self.args.cos_reparams:
                joint_angles_R = self.problem.joint_angles_ub - self.problem.joint_angles_lb
                joint_angles = self.problem.joint_angles_lb + \
                            joint_angles_R / math.pi * \
                            torch.acos(1 - 2 * (joint_angles - self.problem.joint_angles_lb) / joint_angles_R)

            # print(f"Frame {i}: Transformed joint angles: {joint_angles}")
            debug_joint_angles_lst.append(joint_angles.clone())

            # # Inject jaw angle prior (must be after this transformation)
            # prior_weight = 0.5
            # jaw_prior = 0.5
            # joint_angles[-2:] = (1 - prior_weight) * joint_angles[-2:] + prior_weight * jaw_prior

            if self.args.symmetric_jaw:
                center_init = torch.cat([cTr, joint_angles[:3]], dim=0) 
            else:
                center_init = torch.cat([cTr, joint_angles], dim=0)

            # If using synthetic data, do not need initialization
            init_flag = not (i > 0 or self.args.data_dir.startswith("./data/synthetic"))

            kwargs = dict(
                problem=self.problem,
                stdev_init=1.,
                center_init=center_init / self.problem.lengthscales,
                popsize=self.args.popsize if not init_flag else min(self.args.popsize, 30),
                sobol=self.sobol,
            )
            sig = inspect.signature(self.optimizer.__init__)
            accepted = set(sig.parameters.keys())
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in accepted
            }

            searcher = self.optimizer(**filtered_kwargs)
            logger = DummyLogger(searcher, interval=1, after_first_step=False)

            searcher.run(self.num_iters if not init_flag else max(self.args.final_iters, self.num_iters))

            # if logger.best_eval > 0.05:
            #     searcher.run(self.num_iters)
            
            with torch.no_grad():
                # Extract the best solution and evaluation from the logger
                best_solution = logger.best_solution

                best_solution = best_solution * self.problem.lengthscales

                cTr, joint_angles = best_solution[:self.problem.pose_dim], best_solution[self.problem.pose_dim:]
                loss = logger.best_eval

                # print("Optimized joint angles (before clamping):", joint_angles)
                debug_joint_angles_lst.append(joint_angles.clone())

                if self.args.symmetric_jaw:
                    joint_angles = torch.cat([joint_angles[:3], joint_angles[-1:]], dim=0)  # make jaws symmetric

                # joint_angles = torch.clamp(joint_angles, self.problem.joint_angles_lb, self.problem.joint_angles_ub)
                # if self.args.searcher == 'CMA-ES':
                if self.args.cos_reparams:
                    joint_angles = self.problem.joint_angles_lb + \
                            0.5 * joint_angles_R * (1 - torch.cos(math.pi * (joint_angles - self.problem.joint_angles_lb) / joint_angles_R))
                else:
                    joint_angles = torch.clamp(joint_angles, self.problem.joint_angles_lb, self.problem.joint_angles_ub)

                # print(f"Frame {i}: Optimized joint angles (after clamping): {joint_angles}")
                debug_joint_angles_lst.append(joint_angles.clone())

                display_debug = False
                for idx, ja in enumerate(debug_joint_angles_lst):
                    if ja.isnan().any():
                        display_debug = True
                        print(f"[Debug] NaN detected in joint angles at step {idx} for frame {i}: {ja}")
                        break
                if display_debug:
                    print(f"[Debug] Initial joint angles: {debug_joint_angles_lst[0]}")
                    print(f"[Debug] Transformed joint angles: {debug_joint_angles_lst[1]}")
                    print(f"[Debug] Optimized joint angles (before clamping): {debug_joint_angles_lst[2]}")
                    print(f"[Debug] Optimized joint angles (after clamping): {debug_joint_angles_lst[3]}")

                # Filter the results via a low-pass filter
                if self.args.use_filter:
                    full_state = torch.cat([cTr, joint_angles], dim=0).cpu().numpy()

                    if i == 0:
                        self.filter.reset(full_state)
                    else:
                        # np.unwrap rotation angles before filtering
                        for j in range(3):
                            prev_cTr_np = self._prev_cTr[j].cpu().numpy()
                            curr_cTr_np = full_state[j]
                            unwrapped_cTr_np = np.unwrap(
                                np.array([prev_cTr_np, curr_cTr_np]), axis=0
                            )[1]
                            full_state[j] = unwrapped_cTr_np

                        # # unwrap pose ambiguity (beta +/- pi, wrist pitch and yaw * +/- 1)
                        # assert self.args.use_mix_angle, "Pose ambiguity handling only implemented for mix-angle representation."
                        # beta_idx = 1
                        # wrist_pitch_idx = 6
                        # wrist_yaw_idx = 7
                        # prev_beta = self._prev_cTr[beta_idx].cpu().numpy()
                        # curr_beta = full_state[beta_idx]
                        # if abs(curr_beta - (prev_beta + np.pi)) < abs(curr_beta - prev_beta):
                        #     full_state[beta_idx] = curr_beta - np.pi
                        #     full_state[wrist_pitch_idx] = -full_state[wrist_pitch_idx]
                        #     full_state[wrist_yaw_idx] = -full_state[wrist_yaw_idx]
                        # elif abs(curr_beta - (prev_beta - np.pi)) < abs(curr_beta - prev_beta):
                        #     full_state[beta_idx] = curr_beta + np.pi
                        #     full_state[wrist_pitch_idx] = -full_state[wrist_pitch_idx]
                        #     full_state[wrist_yaw_idx] = -full_state[wrist_yaw_idx]

                        self.filter.update(full_state)

                    filtered_state = self.filter.get_x_hat()
                    cTr = torch.from_numpy(filtered_state[:self.problem.pose_dim]).to(self.model.device).type(cTr.dtype)
                    joint_angles = torch.from_numpy(filtered_state[self.problem.pose_dim:]).to(self.model.device).type(joint_angles.dtype)

                self._prev_cTr = cTr.detach().clone()
                self._prev_joint_angles = joint_angles.detach().clone()

            if i == 0:
                self.problem.dist_loss = use_dist_loss # restore distance loss
                self.problem.dist_weight = dist_weight # restore distance weight
                self.problem.pts_weight = kpts_weight # restore pts weight

            end_time = time.time()

            cTr_seq.append(self._prev_cTr.detach().clone())
            joint_angles_seq.append(self._prev_joint_angles.detach().clone())
            loss_seq.append(loss)
            time_seq.append(end_time - start_time)

            pbar.set_postfix({'Loss': f'{loss:.4f}'})

        if self.args.use_mix_angle:
            # Always convert output back to axis-angle representation
            for i in range(frame_num):
                mix_angle = cTr_seq[i][:3].unsqueeze(0)
                axis_angle = mix_angle_to_axis_angle(mix_angle) # Convert to axis-angle
                cTr_seq[i][:3] = axis_angle.squeeze(0)       # Replace the first 3 elements with axis-angle

        if visualization:
            for i in range(frame_num):
                ref_mask = mask_lst[i].to(self.model.device)  # shape (1, H, W)
                if ref_mask.shape[0] != self.problem.resolution[0] or ref_mask.shape[1] != self.problem.resolution[1]:
                    ref_mask = F.interpolate(
                        ref_mask.float().unsqueeze(0).unsqueeze(0), 
                        size=self.problem.resolution, 
                        mode='bilinear'
                    ).squeeze(0).squeeze(0)
                    
                joint_angles = joint_angles_seq[i]
                with torch.no_grad():
                    if self.args.use_contour_tip_net:
                        # ref_keypoints = detect_keypoints(
                        #     model=self.contour_tip_net,
                        #     mask=ref_mask.clone().detach(),
                        # )
                        ref_keypoints = detect_keypoints_2d(
                            model=self.tip_2d_net,
                            mask=ref_mask.clone().detach(),
                        )
                    else:
                        ref_keypoints = ref_keypoints_lst[i][:2] 
                det_line_params = det_line_params_lst[i] if det_line_params_lst is not None else None

                overlay = self.visualize(
                    (ref_mask > 0.5).float(), cTr_seq[i], joint_angles, ref_keypoints
                )
                overlay_seq.append(overlay)

        else:
            overlay_seq = [None] * frame_num

        return (
            th.stack(cTr_seq),
            th.stack(joint_angles_seq),
            th.tensor(loss_seq),
            th.tensor(time_seq),
            overlay_seq,
        )


class BiManualTracker(Tracker):
    def __init__(
        self, model, robot_renderer, init_cTr, init_joint_angles, 
        num_iters=5, intr=None, p_local1=None, p_local2=None, 
        stdev_init=1., searcher="CMA-ES", args=None
    ):
        self.model = model
        self.robot_renderer = robot_renderer

        self._prev_cTr = init_cTr # shape (2, 6)
        self._prev_joint_angles = init_joint_angles # shape (2, 4) or (2, 3)
        self.num_iters = num_iters

        self.num_iters = num_iters # number of iterations for optimization

        self.intr = intr  
        self.p_local1 = p_local1  
        self.p_local2 = p_local2 

        self.args = args

        self.fx, self.fy, self.px, self.py = intr[0, 0].item(), intr[1, 1].item(), intr[0, 2].item(), intr[1, 2].item()

        if args.symmetric_jaw:
            stdev_init = stdev_init.reshape(2, -1)[:, :9].reshape(-1)  # Use 9 dimensions if symmetric jaws
        self.stdev_init = stdev_init  # Initial standard deviation for the optimization

        if self.model.args.use_nvdiffrast is not True:
            print("[Warning] NvDiffRast renderer is not enabled. Automatically enabling it for better performance.]")
            self.model.args.use_nvdiffrast = True

        if self.args.searcher != "Gradient":
            print("[Antialiasing is not enabled in the NvDiffRast renderer for black box optimization.]")
            self.model.use_antialiasing = False

        self.problem = BiManualPoseEstimationProblem(model, robot_renderer, None, intr, p_local1, p_local2, self.stdev_init, args)

        self.separate_loss = args.separate_loss
        self.soft_separation = args.soft_separation

        if args.filter_option == "Kalman":
            self.filter_left = self.filter = KalmanFilter(
                process_noise_pos=np.array([2e-5, 1e-4, 2e-5, 2e-5, 2e-5, 2e-5, 1e-4, 1e-4, 1e-4, 1e-4]),      # scalar or (D,)
                process_noise_vel=np.array([2e-4, 1e-3, 2e-4, 2e-4, 2e-4, 2e-4, 1e-3, 1e-3, 1e-3, 1e-3]),      # scalar or (D,)
                measurement_noise=np.array([2e-3, 1e-2, 2e-3, 2e-3, 2e-3, 2e-3, 5e-3, 5e-3, 5e-3, 5e-3]) * 0.5,      # scalar or (D,)
            )
            self.filter_right = self.filter = KalmanFilter(
                process_noise_pos=np.array([2e-5, 1e-4, 2e-5, 2e-5, 2e-5, 2e-5, 1e-4, 1e-4, 1e-4, 1e-4]),      # scalar or (D,)
                process_noise_vel=np.array([2e-4, 1e-3, 2e-4, 2e-4, 2e-4, 2e-4, 1e-3, 1e-3, 1e-3, 1e-3]),      # scalar or (D,)
                measurement_noise=np.array([2e-3, 1e-2, 2e-3, 2e-3, 2e-3, 2e-3, 5e-3, 5e-3, 5e-3, 5e-3]) * 0.5,      # scalar or (D,)
            )
        elif args.filter_option == "OneEuro":
            self.filter_left = self.filter = OneEuroFilter(
                f_min=0.5,
                alpha_d=0.3, 
                beta=0.3, 
                kappa=0.6
            )
            self.filter_right = self.filter = OneEuroFilter(
                f_min=0.5,
                alpha_d=0.3, 
                beta=0.3, 
                kappa=0.6
            )
        elif args.filter_option == "OneEuro_orig":
            self.filter_left = self.filter = OneEuroFilter(
                f_min=0.8, 
                alpha_d=0.3, 
                beta=0.9, 
                kappa=0.
            )
            self.filter_right = self.filter = OneEuroFilter(
                f_min=0.8, 
                alpha_d=0.3, 
                beta=0.9, 
                kappa=0.
            )

        # Different variants of CMA-ES for different settings
        if self.separate_loss:
            if self.args.use_bd_cmaes:
                CMAES_searcher = CMAES_bi_manual_bd_cus
                print("[Optimizer: Using bi-manual CMA-ES with separate loss and block-diagonal covariance.]")
            else:
                CMAES_searcher = CMAES_bi_manual_cus
                print("[Optimizer: Using bi-manual CMA-ES with separate loss and FULL covariance.]")
        else:
            if self.args.use_bd_cmaes:
                CMAES_searcher = CMAES_bd_cus
                print("[Optimizer: Using CMA-ES with block-diagonal covariance.]")
            else:
                CMAES_searcher = CMAES_cus
                print("[Optimizer: Using CMA-ES with FULL covariance.]")

        optimizer_dict = {
            "CMA-ES": CMAES_searcher, # customized CMA-ES implementation
            "XNES": XNES,
            "Gradient": GradientDescentSearcher,
        }
        self.optimizer = optimizer_dict[searcher]

        # Transform the intial cTr to Euler angle if required
        if self.args.use_mix_angle:
            print("[Using transformed angle space for optimization.]")
            self._prev_cTr[:,:3] = axis_angle_to_mix_angle(self._prev_cTr[:,:3])

        else:
            print("[Using axis-angle space for optimization.]")

        self.sobol = SobolEngine(dimension=self.problem.solution_length, scramble=True) # Sobol sequence generator for low-discrepancy sampling

        if args.use_contour_tip_net:
            self.tip_2d_net = Tip2DNet().to(self.model.device)
            self.tip_2d_net.load_state_dict(
                th.load(
                    args.contour_tip_net_path, 
                    map_location=self.model.device
                )
            )
            self.tip_2d_net.eval()
            self.tip_2d_net.compile()

    def overlay_mask(self, ref_mask, pred_mask, ref_pts_left=None, ref_pts_right=None, proj_pts_left=None, proj_pts_right=None):
        """
        Overlay the predicted mask on the reference mask for visualization.
        """
        # Convert masks to grayscale images
        ref_mask = ref_mask.float().cpu().numpy()
        pred_mask = pred_mask.float().cpu().numpy()

        w, h = ref_mask.shape[1], ref_mask.shape[0]
        
        # Create a color overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)  # Create an empty overlay image
        overlay[..., 0] = (ref_mask * 255).astype(np.uint8) 
        overlay[..., 2] = (pred_mask * 255).astype(np.uint8)  

        if ref_pts_left is not None:
            center_ref_pt = th.mean(ref_pts_left, dim=0)
            for ref_pt in ref_pts_left:
                u_ref, v_ref = int(ref_pt[0]), int(ref_pt[1])
                cv2.circle(
                    overlay,
                    (u_ref, v_ref),
                    radius=5,
                    color=(255, 0.6 * 255, 0.1 * 255),
                    thickness=-1,
                )  

            u_ref, v_ref = int(center_ref_pt[0]), int(center_ref_pt[1])
            cv2.circle(
                overlay,
                (u_ref, v_ref),
                radius=5,
                color=(255, 0.6 * 255, 0.1 * 255),
                thickness=-1,
            )  

        if ref_pts_right is not None:
            center_ref_pt = th.mean(ref_pts_right, dim=0)
            for ref_pt in ref_pts_right:
                u_ref, v_ref = int(ref_pt[0]), int(ref_pt[1])
                cv2.circle(
                    overlay,
                    (u_ref, v_ref),
                    radius=5,
                    color=(255, 0.6 * 255, 0.1 * 255),
                    thickness=-1,
                )  

            u_ref, v_ref = int(center_ref_pt[0]), int(center_ref_pt[1])
            cv2.circle(
                overlay,
                (u_ref, v_ref),
                radius=5,
                color=(255, 0.6 * 255, 0.1 * 255),
                thickness=-1,
            )

        if proj_pts_left is not None:
            center_proj_pt = th.mean(proj_pts_left, dim=0)
            for proj_pt in proj_pts_left.squeeze():
                u_proj, v_proj = int(proj_pt[0].item()), int(proj_pt[1].item())
                cv2.circle(
                    overlay, (u_proj, v_proj), radius=5, color=(255*0.1, 255*0.6, 255), thickness=-1
                )  #

            u_ref, v_ref = int(center_proj_pt[0]), int(center_proj_pt[1])
            cv2.circle(
                overlay, (u_ref, v_ref), radius=5, color=(255*0.1, 255*0.6, 255), thickness=-1
            ) 

        if proj_pts_right is not None:
            center_proj_pt = th.mean(proj_pts_right, dim=0)
            for proj_pt in proj_pts_right.squeeze():
                u_proj, v_proj = int(proj_pt[0].item()), int(proj_pt[1].item())
                cv2.circle(
                    overlay, (u_proj, v_proj), radius=5, color=(255*0.1, 255*0.6, 255), thickness=-1
                )  #

            u_ref, v_ref = int(center_proj_pt[0]), int(center_proj_pt[1])
            cv2.circle(
                overlay, (u_ref, v_ref), radius=5, color=(255*0.1, 255*0.6, 255), thickness=-1
            )

        return overlay

    def project_keypoints(self, cTr, joint_angles):
        pose_matrix = self.model.cTr_to_pose_matrix(cTr.unsqueeze(0)).squeeze()
        R_list, t_list = lndFK(joint_angles)
        R_list = R_list.to(self.model.device)
        t_list = t_list.to(self.model.device)
        p_img1 = get_img_coords(
            self.p_local1,
            R_list[2],
            t_list[2],
            pose_matrix.to(joint_angles.dtype),
            self.intr,
        )
        p_img2 = get_img_coords(
            self.p_local2,
            R_list[3],
            t_list[3],
            pose_matrix.to(joint_angles.dtype),
            self.intr,
        )
        return th.stack([p_img1, p_img2], dim=0)

    def visualize(self, mask, cTr_left, cTr_right, joint_angles_left, joint_angles_right, ref_keypoints):
        # Render the predicted mask for visualization
        robot_mesh_left = self.robot_renderer.get_robot_mesh(joint_angles_left)
        rendered_mask_left = self.model.render_single_robot_mask(cTr_left, robot_mesh_left, self.robot_renderer, self.problem.render_resolution).squeeze(0)
        robot_mesh_right = self.robot_renderer.get_robot_mesh(joint_angles_right)
        rendered_mask_right = self.model.render_single_robot_mask(cTr_right, robot_mesh_right, self.robot_renderer, self.problem.render_resolution).squeeze(0)
        rendered_mask = th.maximum(rendered_mask_left, rendered_mask_right)

        if self.args.downscale_factor != 1:
            rendered_mask = F.interpolate(rendered_mask.unsqueeze(0).unsqueeze(0), size=self.problem.resolution, mode='bilinear').squeeze(0).squeeze(0)

        # Project keypoints
        proj_keypoints_left = self.project_keypoints(cTr_left, joint_angles_left)
        proj_keypoints_right = self.project_keypoints(cTr_right, joint_angles_right)

        return self.overlay_mask(
            mask.detach(), 
            rendered_mask.detach(),
            ref_pts_left=ref_keypoints[0],
            ref_pts_right=ref_keypoints[1],
            proj_pts_left=proj_keypoints_left,
            proj_pts_right=proj_keypoints_right,
        )

    def track(self, ref_mask, joint_angles=None, ref_keypoints=None):
        pass

    def track_sequence(
        self, mask_lst, joint_angles_lst, ref_keypoints_lst, det_line_params_lst, visualization=False
    ):
        frame_num = mask_lst.shape[0]  # number of frames
        cTr_seq = []
        joint_angles_seq = []
        loss_seq = []
        time_seq = []
        overlay_seq = []

        pbar = tqdm.tqdm(range(frame_num), desc="Tracking frames")
        for i in pbar:
            start_time = time.time()

            if i == 0:
                # Initialization settings
                kpts_weight = self.problem.pts_weight
                self.problem.pts_weight = 5e-3 # increase the keypoint weight during initialization

            ref_mask = mask_lst[i].to(self.model.device)  # shape (2, H, W)
            if ref_mask.shape[1] != self.problem.resolution[0] or ref_mask.shape[2] != self.problem.resolution[1]:
                ref_mask = F.interpolate(
                    ref_mask.float().unsqueeze(1), 
                    size=self.problem.resolution, 
                    mode='bilinear'
                ).squeeze(1)
            joint_angles = joint_angles_lst[i] # shape (2, D)
            ref_keypoints = ref_keypoints_lst[i][:,:2]

            with torch.no_grad():
                if i != 0 and self.args.use_contour_tip_net:
                    ref_keypoints_left = detect_keypoints_2d(
                        model=self.tip_2d_net,
                        mask=ref_mask[0].detach(),
                    )
                    ref_keypoints_right = detect_keypoints_2d(
                        model=self.tip_2d_net,
                        mask=ref_mask[1].detach(),
                    )
                    ref_keypoints = [ref_keypoints_left, ref_keypoints_right]

            stdev_init = self.stdev_init.clone()

            self.problem.update_problem(
                ref_mask, ref_keypoints, self._prev_cTr.clone(), stdev_init
            )

            # Initialize the solution with the previous cTr and joint angles
            cTr = self.problem.cTr_init 
            joint_angles = joint_angles.clone() if not self.args.use_prev_joint_angles else self._prev_joint_angles.clone()

            # Need to make sure joint angles are bounded before transformation
            joint_angles = torch.clamp(joint_angles, self.problem.joint_angles_lb, self.problem.joint_angles_ub)

            # Reverse transform of joint angles (instead of sigmoid)
            # if self.args.searcher == 'CMA-ES':
            if self.args.cos_reparams:
                joint_angles_R = self.problem.joint_angles_ub - self.problem.joint_angles_lb
                joint_angles = self.problem.joint_angles_lb + \
                            joint_angles_R / math.pi * \
                            torch.acos(1 - 2 * (joint_angles - self.problem.joint_angles_lb) / joint_angles_R)

            if self.args.symmetric_jaw:
                center_init = torch.cat([cTr, joint_angles[:, :3]], dim=1).reshape(-1)
            else:
                center_init = torch.cat([cTr, joint_angles], dim=1).reshape(-1)

            # If using synthetic data, do not need initialization
            init_flag = not (i > 0 or self.args.data_dir.startswith("./data/synthetic"))

            kwargs = dict(
                problem=self.problem,
                stdev_init=1.,
                center_init=center_init / self.problem.lengthscales,
                popsize=self.args.popsize if not init_flag else min(self.args.popsize, 30),
                sobol=self.sobol,
            )
            sig = inspect.signature(self.optimizer.__init__)
            accepted = set(sig.parameters.keys())
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in accepted
            }

            searcher = self.optimizer(**filtered_kwargs)
            logger = BiManualLogger(searcher, interval=1, after_first_step=False) if self.separate_loss else DummyLogger(searcher, interval=1, after_first_step=False)

            searcher.run(self.num_iters if not init_flag else max(self.args.final_iters, self.num_iters))

            # # Run the searcher for the specified number of iterations
            # # If using synthetic data, do not need initialization
            # if i > 0 or self.args.data_dir.startswith("./data/synthetic"):
            #     # Define the searcher and logger
            #     searcher = self.optimizer(
            #         problem=self.problem,
            #         stdev_init=1.,
            #         center_init=center_init / self.problem.lengthscales,
            #         popsize=self.args.popsize,
            #         sobol=self.sobol,
            #     )
            #     logger = BiManualLogger(searcher, interval=1, after_first_step=False) if self.separate_loss else DummyLogger(searcher, interval=1, after_first_step=False)
            #     searcher.run(self.num_iters) 
            # else:
            #     # Define the searcher and logger
            #     searcher = self.optimizer(
            #         problem=self.problem,
            #         stdev_init=1.,
            #         center_init=center_init / self.problem.lengthscales,
            #         popsize=min(self.args.popsize, 30),
            #         sobol=self.sobol,
            #     )
            #     logger = BiManualLogger(searcher, interval=1, after_first_step=False) if self.separate_loss else DummyLogger(searcher, interval=1, after_first_step=False)
            #     # self.problem.lengthscales *= 10. # increase stdev_init for initialization
            #     searcher.run(max(self.args.final_iters, self.num_iters))

            # if logger.best_eval > 0.05:
            #     searcher.run(self.num_iters)
            
            with torch.no_grad():
                # Extract the best solution and evaluation from the logger
                best_solution = logger.best_solution

                best_solution = best_solution * self.problem.lengthscales

                best_solution = best_solution.reshape(2, -1)
                cTr, joint_angles = best_solution[:, :self.problem.pose_dim], best_solution[:, self.problem.pose_dim:]
                loss = logger.best_eval

                if self.args.symmetric_jaw:
                    joint_angles = torch.cat([joint_angles[:, :3], joint_angles[:, -1:]], dim=1)  # make jaws symmetric

                # joint_angles = torch.clamp(joint_angles, self.problem.joint_angles_lb, self.problem.joint_angles_ub)
                # if self.args.searcher == 'CMA-ES':
                if self.args.cos_reparams:
                    joint_angles = self.problem.joint_angles_lb + \
                            0.5 * joint_angles_R * (1 - torch.cos(math.pi * (joint_angles - self.problem.joint_angles_lb) / joint_angles_R))
                else:
                    joint_angles = torch.clamp(joint_angles, self.problem.joint_angles_lb, self.problem.joint_angles_ub)

                # Filter the results via a low-pass filter
                if self.args.use_filter:
                    full_state_left = torch.cat([cTr[0,:], joint_angles[0,:]], dim=0).cpu().numpy()

                    if i == 0:
                        self.filter_left.reset(full_state_left)
                    else:
                        # np.unwrap rotation angles before filtering
                        for j in range(3):
                            prev_cTr_np = self._prev_cTr[0][j].cpu().numpy()
                            curr_cTr_np = full_state_left[j]
                            unwrapped_cTr_np = np.unwrap(
                                np.array([prev_cTr_np, curr_cTr_np]), axis=0
                            )[1]
                            full_state_left[j] = unwrapped_cTr_np
                        self.filter_left.update(full_state_left)

                    filtered_state = self.filter_left.get_x_hat()
                    cTr_left = torch.from_numpy(filtered_state[:self.problem.pose_dim]).to(self.model.device).type(cTr.dtype)
                    joint_angles_left = torch.from_numpy(filtered_state[self.problem.pose_dim:]).to(self.model.device).type(joint_angles.dtype)

                    full_state_right = torch.cat([cTr[1,:], joint_angles[1,:]], dim=0).cpu().numpy()

                    if i == 0:
                        self.filter_right.reset(full_state_right)
                    else:
                        # np.unwrap rotation angles before filtering
                        for j in range(3):
                            prev_cTr_np = self._prev_cTr[1][j].cpu().numpy()
                            curr_cTr_np = full_state_right[j]
                            unwrapped_cTr_np = np.unwrap(
                                np.array([prev_cTr_np, curr_cTr_np]), axis=0
                            )[1]
                            full_state_right[j] = unwrapped_cTr_np
                        self.filter_right.update(full_state_right)

                    filtered_state = self.filter_right.get_x_hat()
                    cTr_right = torch.from_numpy(filtered_state[:self.problem.pose_dim]).to(self.model.device).type(cTr.dtype)
                    joint_angles_right = torch.from_numpy(filtered_state[self.problem.pose_dim:]).to(self.model.device).type(joint_angles.dtype)

                    cTr = th.stack([cTr_left, cTr_right], dim=0) # shape (2, 6)
                    joint_angles = th.stack([joint_angles_left, joint_angles_right], dim=0) # shape (2, D)

                self._prev_cTr = cTr.detach().clone()
                self._prev_joint_angles = joint_angles.detach().clone()

            if i == 0:
                self.problem.pts_weight = kpts_weight # restore pts weight

            end_time = time.time()

            cTr_seq.append(self._prev_cTr.detach().clone())
            joint_angles_seq.append(self._prev_joint_angles.detach().clone())
            loss_seq.append(loss)
            time_seq.append(end_time - start_time)

            pbar.set_postfix({'Loss': f'{loss:.4f}'})

        if self.args.use_mix_angle:
            # Always convert output back to axis-angle representation
            for i in range(frame_num):
                mix_angle = cTr_seq[i][:,:3] # (2, 3)
                axis_angle = mix_angle_to_axis_angle(mix_angle) # Convert to axis-angle
                cTr_seq[i][:,:3] = axis_angle       # Replace the first 3 elements with axis-angle

        if visualization:
            for i in range(frame_num):
                if mask_lst[i][0].shape[0] != self.problem.resolution[0] or mask_lst[i][0].shape[1] != self.problem.resolution[1]:
                    mask_mask = F.interpolate(
                        mask_lst[i].float().unsqueeze(1), 
                        size=self.problem.resolution, 
                        mode='bilinear'
                    ).squeeze(1)
                else:
                    mask_mask = mask_lst[i]
                ref_mask = torch.max(mask_mask[0], mask_mask[1]).to(self.model.device)  # shape (H, W)

                cTr_left, cTr_right = cTr_seq[i][0], cTr_seq[i][1]
                joint_angles_left, joint_angles_right = joint_angles_seq[i][0], joint_angles_seq[i][1]
                with torch.no_grad():
                    if self.args.use_contour_tip_net:
                        ref_keypoints_left = detect_keypoints_2d(
                            model=self.tip_2d_net,
                            mask=mask_mask[0].to(self.model.device),
                        )
                        ref_keypoints_right = detect_keypoints_2d(
                            model=self.tip_2d_net,
                            mask=mask_mask[1].to(self.model.device),
                        )
                    else:
                        assert False, "Bi-manual tracker requires contour tip net for keypoint detection."

                overlay = self.visualize(
                    (ref_mask > 0.5).float(), cTr_left, cTr_right, joint_angles_left, joint_angles_right, 
                    [ref_keypoints_left, ref_keypoints_right]
                )
                overlay_seq.append(overlay)

        else:
            overlay_seq = [None] * frame_num

        return (
            th.stack(cTr_seq),
            th.stack(joint_angles_seq),
            th.tensor(loss_seq),
            th.tensor(time_seq),
            overlay_seq,
        )