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

    # Setting for SurgPose data
    args.height = 986 // 2
    args.width = 1400 // 2
    args.fx, args.fy, args.px, args.py = 1811.910046453570 / 2, 1809.640734154330 / 2, 588.5594517681759 / 2, 477.3975900383616 / 2
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

"""
Generate 12 trajectories corresponding to {0..7} {30..33}

python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000000/PSM1 --source_traj_path "./pose_results/surgpose_000000_PSM1.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000001/PSM1 --source_traj_path "./pose_results/surgpose_000001_PSM1.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000002/PSM1 --source_traj_path "./pose_results/surgpose_000002_PSM1.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000003/PSM1 --source_traj_path "./pose_results/surgpose_000003_PSM1.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000004/PSM1 --source_traj_path "./pose_results/surgpose_000004_PSM1.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000005/PSM1 --source_traj_path "./pose_results/surgpose_000005_PSM1.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000006/PSM1 --source_traj_path "./pose_results/surgpose_000006_PSM1.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000007/PSM1 --source_traj_path "./pose_results/surgpose_000007_PSM1.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000008/PSM1 --source_traj_path "./pose_results/surgpose_000030_PSM1.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000009/PSM1 --source_traj_path "./pose_results/surgpose_000031_PSM1.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000010/PSM1 --source_traj_path "./pose_results/surgpose_000032_PSM1.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000011/PSM1 --source_traj_path "./pose_results/surgpose_000033_PSM1.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"

python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000000/PSM3 --source_traj_path "./pose_results/surgpose_000000_PSM3.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000001/PSM3 --source_traj_path "./pose_results/surgpose_000001_PSM3.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000002/PSM3 --source_traj_path "./pose_results/surgpose_000002_PSM3.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000003/PSM3 --source_traj_path "./pose_results/surgpose_000003_PSM3.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000004/PSM3 --source_traj_path "./pose_results/surgpose_000004_PSM3.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000005/PSM3 --source_traj_path "./pose_results/surgpose_000005_PSM3.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000006/PSM3 --source_traj_path "./pose_results/surgpose_000006_PSM3.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000007/PSM3 --source_traj_path "./pose_results/surgpose_000007_PSM3.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000008/PSM3 --source_traj_path "./pose_results/surgpose_000030_PSM3.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000009/PSM3 --source_traj_path "./pose_results/surgpose_000031_PSM3.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000010/PSM3 --source_traj_path "./pose_results/surgpose_000032_PSM3.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"
python scripts/generate_synthetic_data.py --target_dir ./data/synthetic/000011/PSM3 --source_traj_path "./pose_results/surgpose_000033_PSM3.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth"

./synthetic_tracking.sh
"""

if __name__ == "__main__":
    with torch.no_grad():
        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument("--target_dir", type=str, default="./data/synthetic/000000/PSM1")
        parser.add_argument("--source_traj_path", type=str, default="./pose_results/surgpose_000000_PSM1.CMA-ES.1.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.pth")
        parser.add_argument("--traj_len", type=int, default=1000)
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

        ref_traj = torch.load(args_cmd.source_traj_path)
        ref_cTr_traj = ref_traj["cTr"]                 # (T, 6) axis-angle + translation
        ref_joints_traj = ref_traj["joint_angles"][:, :3]

        # ------------------------------------------------
        # Convert rotations to mix-angle space (once!)
        # ------------------------------------------------
        ref_mix_traj = axis_angle_to_mix_angle(ref_cTr_traj[:, :3])
        ref_pos_traj = ref_cTr_traj[:, 3:]

        T = ref_cTr_traj.shape[0]
        num_segments = (T - 1) // interval

        mix_angle_seq_all = []
        joint_angles_seq_all = []
        position_seq_all = []

        # ------------------------------------------------
        # Subsample indices
        # ------------------------------------------------
        sub_idx = torch.arange(0, T, interval, device=device)
        if sub_idx[-1] != T - 1:
            sub_idx = torch.cat([sub_idx, torch.tensor([T - 1], device=device)])

        # ------------------------------------------------
        # Unwrap mix-angle trajectory (IMPORTANT)
        # ------------------------------------------------
        # unwrap along time, per angle dimension
        ref_mix_sub = ref_mix_traj[sub_idx]                      # (S, 3)
        # ref_mix_sub_unwrapped = torch.unwrap(ref_mix_sub, dim=0)
        ref_mix_sub_unwrapped = torch.from_numpy(np.unwrap(ref_mix_sub.cpu().numpy(), axis=0)).to(torch.float32).to(device)

        for k in range(len(sub_idx) - 1):

            # -----------------------------
            # Current / next (from reference)
            # -----------------------------
            mix_angle_curr = ref_mix_sub_unwrapped[k]
            mix_angle_next = ref_mix_sub_unwrapped[k + 1]

            position_curr = ref_pos_traj[sub_idx[k]]
            position_next = ref_pos_traj[sub_idx[k + 1]]

            joint_angles_curr = ref_joints_traj[sub_idx[k]]
            joint_angles_next = ref_joints_traj[sub_idx[k + 1]]

            # Interpolation parameter
            t = torch.linspace(0, 1, interval, device=device)

            # ============================================================
            # Rotation: interpolate directly in mix-angle space
            # ============================================================
            mix_vel = torch.zeros_like(mix_angle_curr)
            mix_chunk = cubic_hermite(
                mix_angle_curr,
                mix_angle_next,
                mix_vel,
                mix_vel,
                t,
            )

            # ============================================================
            # Joints: cubic Hermite
            # ============================================================
            joint_vel = torch.zeros_like(joint_angles_curr)
            joint_chunk = cubic_hermite(
                joint_angles_curr,
                joint_angles_next,
                joint_vel,
                joint_vel,
                t,
            )

            # ============================================================
            # Position: cubic Hermite
            # ============================================================
            pos_vel = torch.zeros_like(position_curr)
            pos_chunk = cubic_hermite(
                position_curr,
                position_next,
                pos_vel,
                pos_vel,
                t,
            )

            # Append
            mix_angle_seq_all.append(mix_chunk)
            joint_angles_seq_all.append(joint_chunk)
            position_seq_all.append(pos_chunk)

        # ------------------------------------------------
        # Concatenate full trajectories
        # ------------------------------------------------
        mix_angle_seq = torch.cat(mix_angle_seq_all, dim=0)
        joint_angles_seq = torch.cat(joint_angles_seq_all, dim=0)
        position_seq = torch.cat(position_seq_all, dim=0)

        # Perturb reference trajectory by OU process
        mix_angle_seq_np = mix_angle_seq.cpu().numpy()
        joint_angles_seq_np = joint_angles_seq.cpu().numpy()
        position_seq_np = position_seq.cpu().numpy()

        # Unwrap the mix angles for smoothness
        for i in range(3):
            mix_angle_seq_np[:, i] = np.unwrap(mix_angle_seq_np[:, i])

        mix_angle_seq_np = ou_perturb_trajectory(
            mix_angle_seq_np,
            rho=0.99,
            sigma=np.array([0.01, 0.03, 0.01], dtype=np.float32),
        )
        joint_angles_seq_np = ou_perturb_trajectory(
            joint_angles_seq_np,
            rho=0.99,
            sigma=np.array([0.02, 0.02, 0.02], dtype=np.float32),
        )
        position_seq_np = ou_perturb_trajectory(
            position_seq_np,    
            rho=0.99,
            sigma=np.array([0.002, 0.002, 0.002], dtype=np.float32),
        )

        # Apply OneEuro filter to smooth the trajectory
        full_seq = torch.from_numpy(
            np.concatenate([mix_angle_seq_np, position_seq_np, joint_angles_seq_np, joint_angles_seq_np[:, -1:]], axis=1)
        ).to(torch.float32).cpu().numpy() # [N, 10]
        filtered_seq = np.zeros_like(full_seq)
        filter = OneEuroFilter(
            f_min=0.5, # higher to reduce lag
            alpha_d=0.3, # smoothing factor for the derivative
            beta=0.3, # higher to reduce lag
            kappa=0.
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

                    joint_noise = np.random.normal(scale=0.02, size=10).astype(np.float32)
                    jaw_noise = np.random.normal(scale=0.02, size=1).astype(np.float32)
                else:
                    noisy_joint = joints + joint_noise
                    noisy_jaw = jaw + jaw_noise
                    joint_noise = joint_noise * 0.6 + np.random.normal(scale=0.02, size=10).astype(np.float32) * (1-0.6**2)**0.5
                    jaw_noise = jaw_noise * 0.6 + np.random.normal(scale=0.02, size=1).astype(np.float32) * (1-0.6**2)**0.5

                # Save
                np.save(joint_path, noisy_joint)
                np.save(jaw_path, noisy_jaw)

                # Save keypoints and cylinders
                ref_keypoints_np = proj_keypoints.cpu().numpy()
                ref_cylinders_np = projected_lines.cpu().numpy

                np.save(kpts_path, ref_keypoints_np)
                np.save(cyd_path, ref_cylinders_np)
        



