import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffcali.models.CtRNet import CtRNet
from diffcali.eval_dvrk.LND_fk import lndFK, batch_lndFK
from diffcali.utils.projection_utils import *
from diffcali.utils.ui_utils import *
from diffcali.utils.cylinder_projection_utils import (
    projectCylinderTorch,
    transform_points,
    transform_points_b,
)
from diffcali.utils.angle_transform_utils import mix_angle_to_axis_angle, axis_angle_to_mix_angle

from diffcali.utils.pose_tracker import OneEuroFilter

import math
import torch
import kornia
import nvdiffrast.torch as dr
import numpy as np
import argparse
import tqdm
import imageio
import torch.nn.functional as F

def boundary_mask(mask):
    """Convert binary mask to edge in PyTorch"""
    # Use kornia to remove small holes
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    max_pool = F.max_pool2d(mask, 3, stride=1, padding=1)
    min_pool = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return (max_pool - min_pool).clamp(0, 1).squeeze()

def sample_boundary_points(boundary, num_points=50):
    """
    Returns a tensor of shape (num_points, 2) with (y,x) indices
    """
    ys, xs = torch.nonzero(boundary > 0, as_tuple=True)
    if len(ys) == 0:
        return torch.empty((0,2), device=boundary.device, dtype=torch.long)
    
    idx = torch.randint(0, len(ys), (num_points,), device=boundary.device)
    return torch.stack([ys[idx], xs[idx]], dim=1)

def add_boundary_blobs(mask, num_blobs=1, radius=5):
    mask_noisy = mask.clone()
    H, W = mask.shape
    boundary = boundary_mask(mask)
    points = sample_boundary_points(boundary, num_blobs)

    yy, xx = torch.meshgrid(torch.arange(H, device=mask.device),
                            torch.arange(W, device=mask.device),
                            indexing="ij")
    
    for p in points:
        y0, x0 = p
        circle = ((yy - y0)**2 + (xx - x0)**2) <= radius**2
        if torch.rand(1) < 0.5:
            mask_noisy[circle] = 1  # add blob
    return mask_noisy

def noisy_mask(mask):
    out = mask.clone()
    kernel_sizes = [1, 1, 3]
    probs = [0.1, 0.05, 0.05] # probabilities for different morphological noise
    for i, k in enumerate(kernel_sizes):
        if torch.rand(1) < probs[i]:
            kernel = torch.ones((k, k), device=mask.device)
            if torch.rand(1) < 0.5:
                out = kornia.morphology.dilation(out.unsqueeze(0).unsqueeze(0), kernel).squeeze()
            else:
                out = kornia.morphology.erosion(out.unsqueeze(0).unsqueeze(0), kernel).squeeze()
            break
    # Add boundary blobs
    radius_lst = [3, 7, 11]
    blob_nums_lambda = [10, 5, 3]
    for i, radius in enumerate(radius_lst):
        if torch.rand(1) < probs[i]:
            num_blobs = np.random.poisson(blob_nums_lambda[i])
            out = add_boundary_blobs(out, num_blobs=num_blobs, radius=radius)
    return out

def transform_mesh(cameras, mesh, R, T, args):
    """
    Transform the mesh from world space to clip space
    Modified from https://github.com/NVlabs/nvdiffrast/issues/148#issuecomment-2090054967
    """
    # world to view transform
    verts = mesh.verts_padded()  #  (B, N_v, 3)
    verts_view = cameras.get_world_to_view_transform(R=R, T=T).transform_points(verts)  # (B, N_v, 3)
    verts_view[...,  :3] *= -1 # due to PyTorch3D camera coordinate conventions
    verts_view_home = torch.cat([verts_view, torch.ones_like(verts_view[..., [0]])], axis=-1) # (B, N_v, 4)

    # projection
    fx, fy = cameras.focal_length[0]
    px, py = cameras.principal_point[0]
    height, width = cameras.image_size[0]
    near, far = args.znear, args.zfar
    A = (2 * fx) / width
    B = (2 * fy) / height
    C = (width - 2 * px) / width
    D = (height - 2 * py) / height
    E = (near + far) / (near - far)
    F = (2 * near * far) / (near - far)
    t_mtx = projectionMatrix = torch.tensor(
        [
            [A, 0, C, 0],
            [0, B, D, 0],
            [0, 0, E, F],
            [0, 0, -1, 0]
        ]
    ).to(verts.device)
    verts_clip = torch.matmul(verts_view_home, t_mtx.transpose(0, 1))

    faces_clip = mesh.faces_padded().to(torch.int32)

    return verts_clip, faces_clip


def render(glctx, pos, pos_idx, resolution: [int, int], antialiasing=True):
    """
    Silhouette rendering pipeline based on NvDiffRast
    """
    # Create color attributes
    col = torch.ones_like(pos[..., :1], dtype=torch.float32) # (B, N_v, 1)
    col_idx = pos_idx

    # Render the mesh
    rast_out, _ = dr.rasterize(glctx, pos, pos_idx, resolution=resolution, grad_db=False)
    color   , _ = dr.interpolate(col, rast_out, col_idx)
    if antialiasing:
        color = dr.antialias(color, rast_out, pos, pos_idx)
    return color.squeeze(-1) # (B, H, W)


def parseArgs():     
    # parser = argparse.ArgumentParser()
    # data_dir = "data/consistency_evaluation/easy/4"
    # parser.add_argument("--data_dir", type=str, default=data_dir)  # reference mask
    # parser.add_argument("--mesh_dir", type=str, default="urdfs/dVRK/meshes")
    # parser.add_argument("--arm", type=str, default="psm2")
    
    # args = parser.parse_args()

    args = argparse.Namespace()
    args.mesh_dir = "urdfs/dVRK/meshes"
    args.arm = "psm2"

    args.use_gpu = True
    args.trained_on_multi_gpus = False

    args.height = 480
    args.width = 640
    args.fx, args.fy, args.px, args.py = 1025.88223, 1025.88223, 167.919017, 234.152707
    args.scale = 1.0

    # clip space parameters
    args.znear = 0
    args.zfar = float('inf')

    # scale the camera parameters
    args.width = int(args.width * args.scale)
    args.height = int(args.height * args.scale)
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale

    args.use_nvdiffrast = False # do not use nvdiffrast in CtRNet

    return args


target_dir = "./data/synthetic_data/1006/"
data_dir = "./data/ground_truth.pt"

"""
python scripts/generate_synthetic_masks.py --target_dir ./data/synthetic_data/syn_rw1 --data_dir ./pose_results/rw1_tracking_results_iters50.pth
python scripts/generate_synthetic_masks.py --target_dir ./data/synthetic_data/syn_rw5 --data_dir ./pose_results/rw5_tracking_results_iters50.pth
python scripts/generate_synthetic_masks.py --target_dir ./data/synthetic_data/syn_rw6 --data_dir ./pose_results/rw6_tracking_results_iters50.pth
python scripts/generate_synthetic_masks.py --target_dir ./data/synthetic_data/syn_rw7 --data_dir ./pose_results/rw7_tracking_results_iters50.pth
python scripts/generate_synthetic_masks.py --target_dir ./data/synthetic_data/syn_rw8 --data_dir ./pose_results/rw8_tracking_results_iters50.pth
python scripts/generate_synthetic_masks.py --target_dir ./data/synthetic_data/syn_rw9 --data_dir ./pose_results/rw9_tracking_results_iters50.pth
python scripts/generate_synthetic_masks.py --target_dir ./data/synthetic_data/syn_rw14 --data_dir ./pose_results/rw14_tracking_results_iters50.pth
python scripts/generate_synthetic_masks.py --target_dir ./data/synthetic_data/syn_rw15 --data_dir ./pose_results/rw15_tracking_results_iters50.pth
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--target_dir", type=str, default=target_dir)
    parser.add_argument("--data_dir", type=str, default=data_dir)
    args_cmd = parser.parse_args()
    target_dir = args_cmd.target_dir
    data_dir = args_cmd.data_dir

    data = torch.load(data_dir)
    cTr_seq = data["cTr"]
    joint_angles_seq = data["joint_angles"]
    N = cTr_seq.shape[0]
    
    # smooth rotation in mixed angle representation
    mix_angle_seq = axis_angle_to_mix_angle(cTr_seq[:, :3])
    joint_angles_seq[:,-2:] = joint_angles_seq[:,-2:].mean(dim=-1, keepdim=True)  # make last two dims equal for smoothing

    # Unwrap mix angle to avoid discontinuity
    for i in range(1, N):
        for j in range(3):
            diff = mix_angle_seq[i, j] - mix_angle_seq[i - 1, j]
            if diff > math.pi:
                mix_angle_seq[i, j] -= 2 * math.pi
            elif diff < -math.pi:
                mix_angle_seq[i, j] += 2 * math.pi
    
    # Apply low-pass filter
    full_seq = torch.cat([mix_angle_seq, cTr_seq[:, 3:], joint_angles_seq], dim=-1).cpu().numpy()
    filtered_seq = np.zeros_like(full_seq)
    filter = OneEuroFilter(
        correction=False, f_min=0.1, alpha_d=0.3, beta=1.,
        lengthscales=np.array([0.5, 5., 0.5, 0.05, 0.05, 0.05, 10., 10., 10., 10.]),
    )
    filter.reset(full_seq[0])
    filtered_seq[0] = full_seq[0]
    for i in range(1, full_seq.shape[0]):
        filter.update(full_seq[i])
        filtered_seq[i] = filter.get_x_hat()
    full_seq = torch.from_numpy(filtered_seq).to(cTr_seq.device)

    cTr_seq = full_seq[:, :6]
    joint_angles_seq = full_seq[:, 6:]

    lb = torch.tensor([-1.5707,     -1.3963, 0.0,    0.0   ], dtype=torch.float32).to("cuda")
    ub = torch.tensor([+1.5707,      1.3963, 1.5707, 1.5707], dtype=torch.float32).to("cuda")
    joint_angles_seq = torch.max(torch.min(joint_angles_seq.to("cuda"), ub), lb)

    args = parseArgs()

    model = CtRNet(args)
    mesh_files = [
        f"{args.mesh_dir}/shaft_multi_cylinder.ply",
        f"{args.mesh_dir}/logo_low_res_1.ply",
        f"{args.mesh_dir}/jawright_lowres.ply",
        f"{args.mesh_dir}/jawleft_lowres.ply",
    ]

    robot_renderer = model.setup_robot_renderer(mesh_files)
    robot_renderer.set_mesh_visibility([True, True, True, True])

    glctx = dr.RasterizeCudaContext() # CUDA context (OpenGL is not available in my WSL)
    resolution = (args.height, args.width)

    # Plot the trajectory
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Plot the prediction results of the 10 dimensions over time
    fig, axs = plt.subplots(5, 2, figsize=(10, 10), sharex=True)
    fig.suptitle("Trajectory Over Time per Dimension", fontsize=16)
    axs = axs.flatten()

    for j in range(6):
        ax = axs[j]
        ax.plot(cTr_seq[:, j].cpu().numpy(), label='cTr', linewidth=1.5)
        ax.set_title(f'cTr Dimension {j}')
        ax.grid(True, alpha=0.4)

    for j in range(6, 10):
        ax = axs[j]
        ax.plot(joint_angles_seq[:, j-6].cpu().numpy(), label='Joint Angle', linewidth=1.5)
        ax.set_title(f'Joint Angle {j-6}')
        ax.grid(True, alpha=0.4)

    fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical', fontsize=14)
    fig.legend(
        ['cTr', 'Joint Angle'],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        fontsize=12
    )
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.98])
    plt.savefig(os.path.join(target_dir, "trajectory_plot.png"))
    plt.close()

    with torch.no_grad():
        pbar = tqdm.tqdm(range(N), desc="Generating synthetic data...")
        for i in pbar:
            cTr = cTr_seq[i].clone()
            # convert back to axis-angle
            cTr[:3] = mix_angle_to_axis_angle(cTr[:3].unsqueeze(0)).squeeze(0)

            joint_angles  = joint_angles_seq[i]

            model.get_joint_angles(joint_angles)
            robot_mesh = robot_renderer.get_robot_mesh(joint_angles)

            R_batched = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
                cTr[:3].unsqueeze(0)
            ) 
            R_batched = R_batched.transpose(1, 2)
            T_batched = cTr[3:].unsqueeze(0)
            negative_mask = T_batched[:, -1] < 0  #flip where negative_mask is True
            T_batched_ = T_batched.clone()
            T_batched_[negative_mask] = -T_batched_[negative_mask]
            R_batched_ = R_batched.clone()
            R_batched_[negative_mask] = -R_batched_[negative_mask]
            pos, pos_idx = transform_mesh(
                cameras=robot_renderer.cameras, mesh=robot_mesh.extend(1),
                R=R_batched_, T=T_batched_, args=args
            ) # project the batched meshes in the clip 
            
            rendered_mask = render(glctx, pos, pos_idx[0], resolution)[0] # shape (H, W)

            # Add noise to the rendered mask by signed distance function
            rendered_mask = noisy_mask(rendered_mask)

            intr = torch.tensor(
                [
                    [args.fx, 0, args.px], 
                    [0, args.fy, args.py], 
                    [0, 0, 1]
                ],
                device="cuda",
                dtype=joint_angles.dtype,
            )

            p_local1 = (
                torch.tensor([0.0, 0.0004, 0.009])
                .to(joint_angles.dtype)
                .to(model.device)
            )
            p_local2 = (
                torch.tensor([0.0, -0.0004, 0.009])
                .to(joint_angles.dtype)
                .to(model.device)
            )
            
            # Project keypoints
            pose_matrix = model.cTr_to_pose_matrix(cTr.unsqueeze(0)).squeeze()
            R_list, t_list = lndFK(joint_angles)
            R_list = R_list.to(model.device)
            t_list = t_list.to(model.device)
            p_img1 = get_img_coords(
                p_local1,
                R_list[2],
                t_list[2],
                pose_matrix.to(joint_angles.dtype),
                intr,
            )
            p_img2 = get_img_coords(
                p_local2,
                R_list[3],
                t_list[3],
                pose_matrix.to(joint_angles.dtype),
                intr,
            )
            proj_keypoints = torch.stack([p_img1, p_img2], dim=0)

            # Project cylinders
            cTr_batch = cTr.unsqueeze(0)  # shape (1, 6)
            B = 1
            position = torch.zeros((B, 3), dtype=torch.float32, device=model.device)  # (B, 3)
            direction = torch.zeros((B, 3), dtype=torch.float32, device=model.device)  # (B, 3)
            direction[:, 2] = 1.0
            pose_matrix_b = model.cTr_to_pose_matrix(cTr_batch).squeeze(0)  # shape(B, 4, 4)
            radius = 0.0085 / 2
            fx, fy, px, py = intr[0, 0].item(), intr[1, 1].item(), intr[0, 2].item(), intr[1, 2].item()

            _, cam_pts_3d_position = transform_points(position, pose_matrix, intr)
            _, cam_pts_3d_norm = transform_points(direction, pose_matrix, intr)
            cam_pts_3d_norm = th.nn.functional.normalize(cam_pts_3d_norm)
            e_1, e_2 = projectCylinderTorch(
                cam_pts_3d_position, cam_pts_3d_norm, radius, fx, fy, px, py
            )  # [B,2], [B,2]
            projected_lines = torch.stack((e_1, e_2), dim=1)  # [B, 2, 2]

            # Format subdir and filenames
            subdir = os.path.join(target_dir, f"{i}")
            os.makedirs(subdir, exist_ok=True)

            mask_path = os.path.join(subdir, f"{i:05d}.png")
            ctr_path = os.path.join(subdir, "optimized_ctr.npy")
            angles_path = os.path.join(subdir, "optimized_joint_angles.npy")
            joint_path = os.path.join(subdir, f"joint_{i:04d}.npy")
            jaw_path = os.path.join(subdir, f"jaw_{i:04d}.npy")
            kpts_path = os.path.join(subdir, f"keypoints_{i:04d}.npy")
            cyd_path = os.path.join(subdir, f"cylinders_{i:04d}.npy")

            # Save mask
            rendered_mask_np = rendered_mask.detach().cpu().numpy()
            rendered_mask_np = (rendered_mask_np * 255).astype(np.uint8)
            imageio.imwrite(mask_path, rendered_mask_np)

            joint_angles_np = joint_angles.detach().cpu().numpy()

            # Save pose and joint angles
            np.save(ctr_path, cTr.detach().cpu().numpy())
            np.save(angles_path, joint_angles_np)

            # Compute fake jaw and joints with noise
            jaw_val = (joint_angles_np[2] + joint_angles_np[3])

            joints = np.random.randn(10).astype(np.float32)  # dummy
            jaw = np.random.randn(1).astype(np.float32)      # dummy

            joints[4] = joint_angles_np[0]
            joints[5] = joint_angles_np[1]
            jaw[0] = jaw_val

            # Add some noise to saved jaw/joint for variety
            if i == 0:
                noisy_joint = joints
                noisy_jaw = jaw

                joint_noise = np.random.normal(scale=0.2, size=10).astype(np.float32)
                jaw_noise = np.random.normal(scale=0.15, size=1).astype(np.float32)
            else:
                noisy_joint = joints + joint_noise
                noisy_jaw = jaw + jaw_noise
                joint_noise = joint_noise * 0.6 + np.random.normal(scale=0.2, size=10).astype(np.float32) * (1-0.6**2)**0.5
                jaw_noise = jaw_noise * 0.6 + np.random.normal(scale=0.15, size=1).astype(np.float32) * (1-0.6**2)**0.5

            # Save
            np.save(joint_path, noisy_joint)
            np.save(jaw_path, noisy_jaw)

            # Save keypoints and cylinders
            ref_keypoints_np = proj_keypoints.cpu().numpy()
            ref_cylinders_np = projected_lines.cpu().numpy

            np.save(kpts_path, ref_keypoints_np)
            np.save(cyd_path, ref_cylinders_np)

        # # Save cache initialization
        # data_dir, bag_number = args_cmd.target_dir.split("/")[-2], args_cmd.target_dir.split("/")[-1]
        # cache_filename = f"./data/cached_initialization/{data_dir}_{bag_number}_0.pth"
        # cache_data = {
        #     "cTr": cTr_seq[0].cpu(),
        #     "joint_angles": joint_angles_seq[0].cpu(),
        # }
        # torch.save(cache_data, cache_filename)
    



