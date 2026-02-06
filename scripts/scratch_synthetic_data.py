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

from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator

import math
import torch
import kornia
import nvdiffrast.torch as dr
import numpy as np
import argparse
import tqdm
import imageio
import torch.nn.functional as F
from scipy.signal import butter, filtfilt

def downsample_contour_pchip(main_contour, fixed_length=200):
    """
    Arc-length–parametrized PCHIP downsampling for an open contour.
    """
    pts = np.asarray(main_contour, dtype=np.float64)
    N = len(pts)

    if N < fixed_length:
        return None

    # ---- arc-length parameter ----
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])

    if s[-1] == 0:
        return None

    # ---- uniform resampling ----
    s_new = np.linspace(0, s[-1], fixed_length)

    # ---- PCHIP interpolation ----
    fx = PchipInterpolator(s, pts[:, 0], extrapolate=False)
    fy = PchipInterpolator(s, pts[:, 1], extrapolate=False)

    x_new = fx(s_new)
    y_new = fy(s_new)

    # Safety (should not trigger)
    if np.any(np.isnan(x_new)) or np.any(np.isnan(y_new)):
        return None

    return np.stack((x_new, y_new), axis=1)


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


def render(glctx, pos, pos_idx, resolution: [int, int], antialiasing=False, col=None):
    """
    Silhouette rendering pipeline based on NvDiffRast
    if col is None, render silhouette mask
    otherwise (col is (1, N_v, 3)), render colored image (three channels)
    """
    # Create color attributes
    if col is None:
        col = torch.ones_like(pos[..., :1], dtype=torch.float32) # (B, N_v, 1)
    col_idx = pos_idx

    # Render the mesh
    rast_out, _ = dr.rasterize(glctx, pos, pos_idx, resolution=resolution)
    color   , _ = dr.interpolate(col, rast_out, col_idx)
    if antialiasing:
        color = dr.antialias(color, rast_out, pos, pos_idx)
    return color.squeeze(-1) # (B, H, W)


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
    blob_nums_lambda = [20, 5, 3] # average number of blobs for different radius
    probs_blob = [0.2, 0.1, 0.05]
    for i, radius in enumerate(radius_lst):
        if torch.rand(1) < probs_blob[i]:
            num_blobs = np.random.poisson(blob_nums_lambda[i])
            out = add_boundary_blobs(out, num_blobs=num_blobs, radius=radius)
    return out


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
    args.znear = 1e-3
    args.zfar = 1e9

    # scale the camera parameters
    args.width = int(args.width * args.scale)
    args.height = int(args.height * args.scale)
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale

    args.use_nvdiffrast = False # do not use nvdiffrast in CtRNet

    return args


def extract_contour(mask: torch.Tensor, contour_length=300):
    """
    mask: (H,W) torch.float {0,1}, single component
    Returns: contour points as np.ndarray (N,2) in (y,x)
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)
    mask_np = (mask_np > 0.5).astype(np.uint8) * 255

    # External contours only, no holes
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None

    # Pick the longest contour
    main_contour = max(contours, key=lambda c: len(c))
    main_contour = main_contour.squeeze(1)  # shape (N,2), columns = (x,y)
    main_contour = main_contour[:, [1,0]]   # convert to (y,x)

    # Filter out the point on the image border
    H, W = mask.shape
    N = len(main_contour)
    valid_indices = []
    for idx, (y, x) in enumerate(main_contour):
        if y <= 1 or y >= H - 1 or x <= 1 or x >= W - 1:
            continue
        valid_indices.append(idx)

    if not valid_indices:
        return None

    valid_indices = np.asarray(valid_indices, dtype=np.int32)

    # # Sorted in circular order (original contour order)
    # valid_indices.sort()

    # Differences with wrap-around
    # diffs = (np.roll(valid_indices, -1) - valid_indices) % N
    diffs = (np.roll(valid_indices, -1) - valid_indices) % N

    # Adjacency means diff == 1
    # breaks = np.where(diffs != 1)[0]
    # print(0.02 * len(valid_indices))
    breaks = np.where(diffs > 1)[0]

    if len(breaks) != 1:
        return None

    cut = breaks[0] + 1   # first index AFTER the deleted segment

    # Rotate indices
    valid_indices = np.roll(valid_indices, -cut)

    # Apply re-ordering to contour
    main_contour = main_contour[valid_indices]

    # Use interpolation to obtain downsampled contour
    # fixed_length = args.contour_length
    main_contour = downsample_contour_pchip(main_contour, fixed_length=contour_length)

    return main_contour


def edge_signal(contour: np.ndarray, centroid: np.ndarray):
    vectors = contour - centroid
    distances = np.linalg.norm(vectors, axis=1)
    return distances


def intialize_pose(model, robot_renderer, args, joints_lb, joints_ub, reference_quat=None):
    """
    Generate inital pose such that the end-effector is within the camera view
    and the two tips are clearly visible
    """
    to_break = False
    while not to_break:
        xyz_lb = torch.tensor([-0.03, -0.03, 0.10], dtype=torch.float32).to("cuda")
        xyz_ub = torch.tensor([+0.03, +0.03, 0.20], dtype=torch.float32).to("cuda")

        # Randomly sample inital joint angles and positions within bounds
        joint_angles_init = torch.empty(3).uniform_(0, 1).to("cuda") * (joints_ub - joints_lb) + joints_lb
        position_init = torch.empty(3).uniform_(0, 1).to("cuda") * (xyz_ub - xyz_lb) + xyz_lb

        if reference_quat is None:
            # Sample initial rotation from pose hypoetheses space
            camera_roll_local = torch.empty(1).uniform_(
                0, 360
            )  # Random values in [0, 360]
            camera_roll = torch.empty(1).uniform_(0, 360)  # Random values in [0, 360]
            azimuth = torch.empty(1).uniform_(0, 360)  # Random values in [0, 360]
            elevation = torch.empty(1).uniform_(
                90 - 60, 90 - 30
            )  # Random values in [90-25, 90+25]
            # elevation = 30

            distance = torch.empty(1).uniform_(0.10, 0.17)

            pose_matrix = model.from_lookat_to_pose_matrix(
                distance, elevation, camera_roll_local
            )
            roll_rad = torch.deg2rad(camera_roll)  # Convert roll angle to radians
            roll_matrix = torch.tensor(
                [
                    [torch.cos(roll_rad), -torch.sin(roll_rad), 0],
                    [torch.sin(roll_rad), torch.cos(roll_rad), 0],
                    [0, 0, 1],
                ]
            )
            pose_matrix[:, :3, :3] = torch.matmul(roll_matrix, pose_matrix[:, :3, :3])
            cTr = model.pose_matrix_to_cTr(pose_matrix).squeeze(0)
    
        else:
            # Perturb reference quaternion and convert to cTr
            try: 
                quat = reference_quat + 0.01 * torch.randn(4).cuda()
            except:
                quat = reference_quat.q + 0.01 * torch.randn(4).cuda()
            quat = quat / torch.norm(quat)
            axis_angle = kornia.geometry.conversions.quaternion_to_axis_angle(quat.unsqueeze(0)).squeeze(0)
            cTr = torch.cat([axis_angle, position_init], dim=0)

        cTr[3:] = position_init

        joint_angles = torch.cat([joint_angles_init, joint_angles_init[-1:].clone()], dim=0)  # jaw1=jaw2

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

        # Extract contour
        contour = extract_contour(rendered_mask * 255., contour_length=200)

        if contour is None:
            continue

        # Project keypoints
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
            torch.tensor([0.0, 0.0004, 0.0096])
            .to(joint_angles.dtype)
            .to(model.device)
        )
        p_local2 = (
            torch.tensor([0.0, -0.0004, 0.0096])
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
        
        # Determine if keypoints is detectable
        # Step 1: Project another point on the tool tip
        p_local3 = (
            torch.tensor([0.0, 0.0004, 0.0075])
            .to(joint_angles.dtype)
            .to(model.device)
        )
        p_local4 = (
            torch.tensor([0.0, -0.0004, 0.0075])
            .to(joint_angles.dtype)
            .to(model.device)
        )
        p_img3 = get_img_coords(
            p_local3,
            R_list[2],
            t_list[2],
            pose_matrix.to(joint_angles.dtype),
            intr,
        )
        p_img4 = get_img_coords(
            p_local4,
            R_list[3],
            t_list[3],
            pose_matrix.to(joint_angles.dtype),
            intr,
        )

        # Step 2: Choose the reference center as the intersection of the two tips
        # First compute the intersection of the two lines (p_img1, p_img3) and (p_img2, p_img4)
        # If the reference is in the direction of (p_img3 -> p_img1) and (p_img4 -> p_img2) flip it to the opposite side
        def line_params(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            return A, B, C
        A1, B1, C1 = line_params(p_img1, p_img3)
        A2, B2, C2 = line_params(p_img2, p_img4)
        D = A1 * B2 - A2 * B1
        if D == 0:
            # Lines are parallel, use midpoint between p_img1 and p_img2
            ref_center = torch.tensor([(p_img1[1] + p_img2[1]).item() / 2,
                                        (p_img1[0] + p_img2[0]).item() / 2])
        else:
            x0 = (B1 * C2 - B2 * C1) / D
            y0 = (C1 * A2 - C2 * A1) / D
            ref_center = torch.tensor([y0.item(), x0.item()])  # (y,x)

        # reference should be closer to p_img3 and p_img4 than p_img1 and p_img2
        ref_center_swapped = torch.tensor([ref_center[1].item(), ref_center[0].item()])
        if (torch.norm(ref_center_swapped - p_img3.cpu()).item() > torch.norm(ref_center_swapped - p_img1.cpu()).item()) or \
            (torch.norm(ref_center_swapped - p_img4.cpu()).item() > torch.norm(ref_center_swapped - p_img2.cpu()).item()):
            ref_center = 2 * torch.tensor([(p_img1[1] + p_img2[1]).item() / 2,
                                            (p_img1[0] + p_img2[0]).item() / 2]) - ref_center

        ref_center = ref_center.cpu().numpy()
        
        # Step 3: Compute star-skeletonization signal and use it to determine keypoint visibility
        keypoints_mask = torch.zeros((proj_keypoints.shape[0],), dtype=torch.bool, device=proj_keypoints.device)

        signal = edge_signal(contour, ref_center)

        # Smooth signal by DFT
        signal_fft = np.fft.fft(signal)
        freq_cutoff_ratio = 0.1
        N = len(signal)
        freq_cutoff = int(N * freq_cutoff_ratio)
        signal_fft[freq_cutoff:(N - freq_cutoff)] = 0
        signal_smooth = np.fft.ifft(signal_fft).real
        signal = signal_smooth

        # Find local maxima of the signal (do not consider circularity here)
        local_max_indices = (np.diff(np.sign(np.diff(signal))) < 0).nonzero()[0] + 1  # +1 due to diff reducing length by 1
        local_max_values = signal[local_max_indices]

        # Check if projected keypoints are close to any local maxima
        if len(local_max_indices) > 0:
            for k in range(proj_keypoints.shape[0]):
                xk, yk = proj_keypoints[k]
                dists = np.linalg.norm(contour[local_max_indices] - np.array([yk.item(), xk.item()]), axis=1)
                if np.min(dists) < 3:  # pixels
                    keypoints_mask[k] = True

        # Check if both keypoints are visible and correspond to distinct local maxima
        if torch.all(keypoints_mask) and (local_max_indices.shape[0] >= 2) and (np.abs(local_max_indices[0] - local_max_indices[1]) > 15):
            to_break = True

        # # Chekck if p_img1 and p_img2 are close to the contour and not overlapped
        # tip1_dist = np.min(np.linalg.norm(contour - np.array([p_img1[1].item(), p_img1[0].item()]), axis=1))
        # tip2_dist = np.min(np.linalg.norm(contour - np.array([p_img2[1].item(), p_img2[0].item()]), axis=1))
        # tips_dist = np.linalg.norm(p_img1.cpu().numpy() - p_img2.cpu().numpy())
        # if tip1_dist < 5 and tip2_dist < 5 and tips_dist > 15:
        #     to_break = True

        # Check if p_img3 and p_img4 are close to the contour (basically not occluded)
        tip3_dist = np.min(np.linalg.norm(contour - np.array([p_img3[1].item(), p_img3[0].item()]), axis=1))
        tip4_dist = np.min(np.linalg.norm(contour - np.array([p_img4[1].item(), p_img4[0].item()]), axis=1))
        if tip3_dist < 3 and tip4_dist < 3:
            to_break = to_break and True

    mix_angle_init = axis_angle_to_mix_angle(cTr[:3].unsqueeze(0)).squeeze(0)

    # Compute reference quaternion
    reference_lookat_init = mix_angle_to_axis_angle(mix_angle_init.unsqueeze(0)).squeeze(0)
    quat_init = kornia.geometry.conversions.axis_angle_to_quaternion(reference_lookat_init.unsqueeze(0)).squeeze(0)

    return mix_angle_init, joint_angles_init, position_init, quat_init


def cubic_hermite(p0, p1, m0, m1, t):
    """
    p0, p1: (..., D)
    m0, m1: (..., D)  (tangents)
    t:      (T,)      in [0,1]
    returns: (T, ..., D)
    """
    t = t.view(-1, *([1] * p0.dim()))
    t2 = t * t
    t3 = t2 * t

    h00 =  2*t3 - 3*t2 + 1
    h10 =      t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 =      t3 -    t2

    return h00*p0 + h10*m0 + h01*p1 + h11*m1        


import numpy as np

def ou_perturb_trajectory(
    traj,
    rho=0.6,
    sigma=0.1,
    reset_mask=None,
):
    """
    Apply OU perturbation around a reference trajectory.

    Args:
        traj: np.ndarray, shape (T, D)
              Reference trajectory
        rho: float in (0,1)
             Temporal correlation coefficient
        sigma: float or array-like of shape (D,)
               Stationary std deviation
        reset_mask: optional np.ndarray, shape (T,)
                    If True at t, reset noise to zero

    Returns:
        perturbed_traj: np.ndarray, shape (T, D)
    """
    traj = traj.astype(np.float32)
    T, D = traj.shape

    sigma = np.asarray(sigma, dtype=np.float32)
    if sigma.ndim == 0:
        sigma = np.full(D, sigma, dtype=np.float32)

    noise = np.zeros(D, dtype=np.float32)
    perturbed = np.zeros_like(traj)

    eps = np.random.randn(T, D).astype(np.float32)

    for t in range(T):
        if reset_mask is not None and reset_mask[t]:
            noise[:] = 0.0

        perturbed[t] = traj[t] + noise

        noise = (
            rho * noise
            + np.sqrt(1.0 - rho**2) * sigma * eps[t]
        )

    return perturbed                   


target_dir = "./data/synthetic_data/1006/"
data_dir = "./data/ground_truth.pt"

"""
python scripts/scratch_synthetic_data.py --target_dir ./data/synthetic_data/syn_new1
python scripts/scratch_synthetic_data.py --target_dir ./data/synthetic_data/syn_new2
python scripts/scratch_synthetic_data.py --target_dir ./data/synthetic_data/syn_new3
python scripts/scratch_synthetic_data.py --target_dir ./data/synthetic_data/syn_new4
python scripts/scratch_synthetic_data.py --target_dir ./data/synthetic_data/syn_new5
python scripts/scratch_synthetic_data.py --target_dir ./data/synthetic_data/syn_new6
python scripts/scratch_synthetic_data.py --target_dir ./data/synthetic_data/syn_new7
python scripts/scratch_synthetic_data.py --target_dir ./data/synthetic_data/syn_new8
python scripts/scratch_synthetic_data.py --target_dir ./data/synthetic_data/syn_new9
python scripts/scratch_synthetic_data.py --target_dir ./data/synthetic_data/syn_new10

./synthetic_tracking.sh
"""

if __name__ == "__main__":
    with torch.no_grad():
        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument("--target_dir", type=str, default=target_dir)
        parser.add_argument("--traj_len", type=int, default=500)
        parser.add_argument("--interval", type=int, default=50)
        args_cmd = parser.parse_args()
        target_dir = args_cmd.target_dir

        # Load model and setup renderer
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

        parser = argparse.ArgumentParser()

        # Generate reference trajectory by random sampling and interperlotion
        interval = args_cmd.interval
        assert args_cmd.traj_len % interval == 0
        device = "cuda"

        joints_lb = torch.tensor([-1.5707, -1.3963, 0.0], dtype=torch.float32).to("cuda")
        joints_ub = torch.tensor([+1.5707,  1.3963, 1.0], dtype=torch.float32).to("cuda")

        mix_angle_seq_all = []
        joint_angles_seq_all = []
        position_seq_all = []

        # Initial pose
        mix_angle_curr, joint_angles_curr, position_curr, quat_curr = intialize_pose(
            model, robot_renderer, args, joints_lb, joints_ub
        )

        axis_angle_curr = mix_angle_to_axis_angle(mix_angle_curr.unsqueeze(0)).squeeze(0)
        quat_curr = kornia.geometry.conversions.axis_angle_to_quaternion(
            axis_angle_curr.unsqueeze(0)
        ).squeeze(0)
        quat_curr = kornia.geometry.quaternion.Quaternion(quat_curr)
        quat_ref_curr = quat_curr

        num_segments = args_cmd.traj_len // interval

        for _ in range(num_segments):

            # Sample next target pose
            mix_angle_next, joint_angles_next, position_next, quat_ref_next = intialize_pose(
                model, robot_renderer, args, joints_lb, joints_ub, reference_quat=quat_ref_curr
            )

            axis_angle_next = mix_angle_to_axis_angle(mix_angle_next.unsqueeze(0)).squeeze(0)
            quat_next = kornia.geometry.conversions.axis_angle_to_quaternion(
                axis_angle_next.unsqueeze(0)
            ).squeeze(0)
            quat_next = kornia.geometry.quaternion.Quaternion(quat_next)

            t = torch.linspace(0, 1, interval, device=device)

            # --- Rotation: SLERP ---
            mix_chunk = []
            for ti in t:
                quat_t = quat_curr.slerp(quat_next, ti).q
                axis_angle_t = kornia.geometry.conversions.quaternion_to_axis_angle(
                    quat_t
                ).squeeze(0)
                mix_angle_t = axis_angle_to_mix_angle(axis_angle_t.unsqueeze(0)).squeeze(0)
                mix_chunk.append(mix_angle_t)

            mix_chunk = torch.stack(mix_chunk, dim=0)

            # --- Joints: cubic spline ---
            joint_vel = torch.zeros_like(joint_angles_curr)
            joint_chunk = cubic_hermite(
                joint_angles_curr,
                joint_angles_next,
                joint_vel,
                joint_vel,
                t
            )

            # --- Position: cubic spline ---
            pos_vel = torch.zeros_like(position_curr)
            pos_chunk = cubic_hermite(
                position_curr,
                position_next,
                pos_vel,
                pos_vel,
                t
            )

            # Append
            mix_angle_seq_all.append(mix_chunk)
            joint_angles_seq_all.append(joint_chunk)
            position_seq_all.append(pos_chunk)

            # Update current pose
            mix_angle_curr = mix_angle_next
            joint_angles_curr = joint_angles_next
            position_curr = position_next
            quat_curr = quat_next
            quat_ref_curr = quat_ref_next

        mix_angle_seq = torch.cat(mix_angle_seq_all, dim=0)
        joint_angles_seq = torch.cat(joint_angles_seq_all, dim=0)
        position_seq = torch.cat(position_seq_all, dim=0)

        # Perturb reference trajectory by OU process
        mix_angle_seq_np = mix_angle_seq.cpu().numpy()
        joint_angles_seq_np = joint_angles_seq.cpu().numpy()
        position_seq_np = position_seq.cpu().numpy()

        # Unwrap the mix angles for smoothness
        for i in range(3):
            diffs = np.diff(mix_angle_seq_np[:, i])
            for j in range(len(diffs)):
                if diffs[j] >  np.pi:
                    mix_angle_seq_np[j+1:, i] -= 2 * np.pi
                elif diffs[j] < -np.pi:
                    mix_angle_seq_np[j+1:, i] += 2 * np.pi

        mix_angle_seq_np = ou_perturb_trajectory(
            mix_angle_seq_np,
            rho=0.99,
            sigma=np.array([0.02, 0.2, 0.02], dtype=np.float32),
        )
        joint_angles_seq_np = ou_perturb_trajectory(
            joint_angles_seq_np,
            rho=0.99,
            sigma=np.array([0.2, 0.2, 0.2], dtype=np.float32),
        )
        position_seq_np = ou_perturb_trajectory(
            position_seq_np,    
            rho=0.99,
            sigma=np.array([0.005, 0.005, 0.005], dtype=np.float32),
        )

        # Apply OneEuro filter to smooth the trajectory
        full_seq = torch.from_numpy(
            np.concatenate([mix_angle_seq_np, position_seq_np, joint_angles_seq_np, joint_angles_seq_np[:, -1:]], axis=1)
        ).to(torch.float32).cpu().numpy() # [N, 10]
        filtered_seq = np.zeros_like(full_seq)
        filter = OneEuroFilter(
            correction=False, f_min=0.1, alpha_d=0.1, beta=1.,
            lengthscales=np.array([5., 5., 5., 5., 5., 5., 10., 10., 10., 10.]),
        )
        filter.reset(full_seq[0])
        filtered_seq[0] = full_seq[0]
        for i in range(1, full_seq.shape[0]):
            filter.update(full_seq[i])
            filtered_seq[i] = filter.get_x_hat()

        mix_angle_seq_np = filtered_seq[:, :3]
        position_seq_np = filtered_seq[:, 3:6]
        joint_angles_seq_np = filtered_seq[:, 6:9]

        mix_angle_seq = torch.from_numpy(mix_angle_seq_np).to(torch.float32).to("cuda")
        joint_angles_seq = torch.from_numpy(joint_angles_seq_np).to(torch.float32).to("cuda")
        position_seq = torch.from_numpy(position_seq_np).to(torch.float32).to("cuda")

        N = args_cmd.traj_len

        cTr_seq = torch.zeros((N, 6), dtype=torch.float32).to("cuda")
        cTr_seq[:, :3] = mix_angle_seq
        cTr_seq[:, 3:] = position_seq
        joint_angles_seq = torch.cat([joint_angles_seq, joint_angles_seq[:, -1:].clone()], dim=1)

        lb = torch.tensor([-1.5707,     -1.3963, 0.0,    0.0   ], dtype=torch.float32).to("cuda")
        ub = torch.tensor([+1.5707,      1.3963, 1.5707, 1.5707], dtype=torch.float32).to("cuda")
        joint_angles_seq = torch.max(torch.min(joint_angles_seq.to("cuda"), ub), lb)

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
        plt.savefig("./trajectory_plot.png")
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
                jaw_val = joint_angles_np[2]

                joints = np.random.randn(10).astype(np.float32)  # dummy
                jaw = np.random.randn(1).astype(np.float32)      # dummy

                joints[4] = joint_angles_np[0]
                joints[5] = joint_angles_np[1]
                jaw[0] = jaw_val

                # Add some noise to saved jaw/joint for variety
                if i == 0:
                    noisy_joint = joints
                    noisy_jaw = jaw

                    joint_noise = np.random.normal(scale=0.1, size=10).astype(np.float32)
                    jaw_noise = np.random.normal(scale=0.15, size=1).astype(np.float32)
                else:
                    noisy_joint = joints + joint_noise
                    noisy_jaw = jaw + jaw_noise
                    joint_noise = joint_noise * 0.6 + np.random.normal(scale=0.1, size=10).astype(np.float32) * (1-0.6**2)**0.5
                    jaw_noise = jaw_noise * 0.6 + np.random.normal(scale=0.15, size=1).astype(np.float32) * (1-0.6**2)**0.5

                # Save
                np.save(joint_path, noisy_joint)
                np.save(jaw_path, noisy_jaw)

                # Save keypoints and cylinders
                ref_keypoints_np = proj_keypoints.cpu().numpy()
                ref_cylinders_np = projected_lines.cpu().numpy

                np.save(kpts_path, ref_keypoints_np)
                np.save(cyd_path, ref_cylinders_np)
        



