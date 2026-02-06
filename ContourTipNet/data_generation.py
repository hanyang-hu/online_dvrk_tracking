import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffcali.models.CtRNet import CtRNet
from diffcali.utils.projection_utils import *
from diffcali.utils.ui_utils import *
from diffcali.utils.angle_transform_utils import mix_angle_to_axis_angle, axis_angle_to_mix_angle
from diffcali.eval_dvrk.LND_fk import lndFK

import math
import torch
import kornia
import nvdiffrast.torch as dr
import numpy as np
import argparse
import tqdm
import torch.nn.functional as F
import cv2
from scipy.interpolate import PchipInterpolator


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

    if len(breaks) > 1:
        return None

    if len(breaks) == 1:
        cut = breaks[0] + 1   # first index AFTER the deleted segment
    else:
        # No deletion or full contour — keep original order
        cut = 0

    # Rotate indices
    valid_indices = np.roll(valid_indices, -cut)

    # Apply re-ordering to contour
    main_contour = main_contour[valid_indices]

    # Use interpolation to obtain downsampled contour
    # fixed_length = args.contour_length
    main_contour = downsample_contour_pchip(main_contour, fixed_length=contour_length)

    return main_contour


def contour_centroid(contour: np.ndarray):
    yc = contour[:,0].mean()
    xc = contour[:,1].mean()
    return np.array([yc, xc])


def edge_signal(contour: np.ndarray, centroid: np.ndarray):
    vectors = contour - centroid
    distances = np.linalg.norm(vectors, axis=1)
    return distances


def compute_contour_properties(contour: np.ndarray):
    contour = contour.copy()
    # Compute the unit normal of the contour at each contour point as a feature
    contour_xy = np.array([[x, y] for y, x in contour], dtype=np.float32)  # [N,2] in (x,y)
    N = contour_xy.shape[0]

    tangents = np.zeros_like(contour_xy)

    # Average of forward and backward differences for interior points
    tangents[1:-1] = (contour_xy[2:] - contour_xy[:-2]) / 2.0

    # Endpoints
    tangents[0] = contour_xy[1] - contour_xy[0]
    tangents[-1] = contour_xy[-1] - contour_xy[-2]

    # Normalize tangents
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    tangents /= tangent_norms

    # Rotate 90 degrees to get unit normals
    normals = np.zeros_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]

    # Compute tangent angles
    angles = np.arctan2(tangents[:,1], tangents[:,0])  # atan2(dy, dx)

    # Curvature: difference of consecutive angles
    curvature = np.zeros(N)
    curvature[1:-1] = angles[2:] - angles[:-2]

    # Wrap angles to [-pi, pi]
    curvature[1:-1] = (curvature[1:-1] + np.pi) % (2*np.pi) - np.pi

    # Endpoints: forward/backward difference
    curvature[0] = angles[1] - angles[0]
    curvature[-1] = angles[-1] - angles[-2]
    curvature[0] = (curvature[0] + np.pi) % (2*np.pi) - np.pi
    curvature[-1] = (curvature[-1] + np.pi) % (2*np.pi) - np.pi

    return normals.copy(), curvature.copy()


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
    parser = argparse.ArgumentParser(description="Synthetic data generation for ContourTipNet")
    
    parser.add_argument('--target_dir', type=str, default="./train/", help='Directory to save generated data')
    parser.add_argument('--sample_number', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--generate_transition', action='store_true', help='Whether to generate transition frames between two poses')
    parser.add_argument('--contour_length', type=int, default=300, help='Fixed length of the contour after downsampling')

    """Example usage:
    python data_generation.py --target_dir train --sample_number 10000 --contour_length 200
    python data_generation.py --target_dir val --sample_number 3000 --contour_length 200
    python data_generation.py --target_dir test --sample_number 2000 --contour_length 200
    """

    args = parser.parse_args()

    args.target_dir = os.path.join("./data/", args.target_dir)

    # args = argparse.Namespace()
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


if __name__ == "__main__":
    args = parseArgs()
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    # Dataset directory and sample number
    sample_number = args.sample_number

    data_dir = args.target_dir
    image_dir = os.path.join(data_dir, "images")
    image_raw_dir = os.path.join(data_dir, "images_raw")
    feature_dir = os.path.join(data_dir, "features")
    label_dir = os.path.join(data_dir, "labels")
    pose_dir = os.path.join(data_dir, "poses")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_raw_dir, exist_ok=True)
    if args.generate_transition:
        os.makedirs(pose_dir, exist_ok=True)
    frame_list = sorted(os.listdir(image_dir))

    model = CtRNet(args)
    mesh_files = [
        f"../{args.mesh_dir}/shaft_multi_cylinder.ply",
        f"../{args.mesh_dir}/logo_low_res_1.ply",
        f"../{args.mesh_dir}/jawright_lowres.ply",
        f"../{args.mesh_dir}/jawleft_lowres.ply",
    ]

    robot_renderer = model.setup_robot_renderer(mesh_files)
    robot_renderer.set_mesh_visibility([True, True, True, True])

    glctx = dr.RasterizeCudaContext() # CUDA context (OpenGL is not available in my WSL)
    resolution = (args.height, args.width)

    for i in tqdm.tqdm(range(sample_number), desc="Generating synthetic data"):

        to_break = False

        while not to_break:

            # Generate random 6D pose from a reasonable distribution
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
            cTr = model.pose_matrix_to_cTr(pose_matrix)

            x, y = torch.empty(1).uniform_(-0.01, 0.01).to(device), torch.empty(1).uniform_(-0.01, 0.01).to(device)

            cTr[:,3] += x
            cTr[:,4] += y
            
            # Sample joint angles
            lb = torch.tensor([-1.3, -1.1, 0.0], dtype=torch.float32).to(device)
            ub = torch.tensor([+1.3,  1.1, 1.2], dtype=torch.float32).to(device)

            joint_angles = torch.empty((1, 3), dtype=torch.float32).uniform_(0, 1).to(device)
            joint_angles = lb + (ub - lb) * joint_angles  # scale to [lb, ub]

            # By probability 0.1, set jaw to zero
            if torch.rand(1) < 0.05:
                joint_angles[0, 2] = 0.0

            joint_angles = torch.cat([joint_angles, joint_angles[:, -1:]], dim=1)  # repeat last angle for 4th joint

            cTr, joint_angles = cTr.squeeze(0), joint_angles.squeeze(0)

            # Randomly sample previous cTr and joint angles
            joint_angles_lb=torch.tensor([-1.5707, -1.3963, 0.,       0.]).to(device)
            joint_angles_ub=torch.tensor([ 1.5707,  1.3963, 1.5707, 1.5707]).to(device)

            angle_stdev = torch.tensor([0.03, 0.1, 0.03]).to(device)
            xyz_stdev = torch.tensor([0.003, 0.003, 0.007]).to(device)
            joint_stdev = torch.tensor([0.1, 0.1, 0.2]).to(device) 
            mix_angle = axis_angle_to_mix_angle(cTr[:3].unsqueeze(0)).squeeze(0)
            mix_angle_prev = mix_angle + torch.randn(3).to(device) * angle_stdev
            xyz_prev = cTr[3:] + torch.randn(3).to(device) * xyz_stdev
            joint_angles_prev = joint_angles[:3] + torch.randn(3).to(device) * joint_stdev
            joint_angles_prev = torch.cat([joint_angles_prev, joint_angles_prev[-1:]], dim=0)  # repeat last angle for 4th joint
            joint_angles_prev = torch.clamp(joint_angles_prev, joint_angles_lb, joint_angles_ub)
            cTr_prev = torch.cat([mix_angle_to_axis_angle(mix_angle_prev.unsqueeze(0)).squeeze(0), xyz_prev], dim=0)

            pose_vec = torch.cat([mix_angle, cTr[3:], joint_angles], dim=0)
            pose_vec_prev = torch.cat([mix_angle_prev, xyz_prev, joint_angles_prev], dim=0)

            # Render the robot mask
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

            if args.generate_transition:
                # Render the previous frame mask
                model.get_joint_angles(joint_angles_prev)
                robot_mesh_prev = robot_renderer.get_robot_mesh(joint_angles_prev)
                R_batched_prev = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
                    cTr_prev[:3].unsqueeze(0)
                )
                R_batched_prev = R_batched_prev.transpose(1, 2)
                T_batched_prev = cTr_prev[3:].unsqueeze(0)
                negative_mask_prev = T_batched_prev[:, -1] < 0  #flip where negative_mask is True
                T_batched_prev_ = T_batched_prev.clone()
                T_batched_prev_[negative_mask_prev] = -T_batched_prev_[negative_mask_prev]
                R_batched_prev_ = R_batched_prev.clone()
                R_batched_prev_[negative_mask_prev] = -R_batched_prev_[negative_mask_prev]
                pos_prev, pos_idx_prev = transform_mesh(
                    cameras=robot_renderer.cameras, mesh=robot_mesh_prev.extend(1),
                    R=R_batched_prev_, T=T_batched_prev_, args=args
                ) # project the batched meshes in the clip

                rendered_mask_prev = render(glctx, pos_prev, pos_idx_prev[0], resolution)[0] # shape (H, W)

            # Add noise to the rendered mask by signed distance function
            rendered_mask = noisy_mask(rendered_mask)
            
            # Extract contour
            contour = extract_contour(rendered_mask * 255., contour_length=args.contour_length)
            if args.generate_transition:
                contour_prev = extract_contour(rendered_mask_prev * 255., contour_length=args.contour_length)
            # print(f"{len(contour) if contour is not None else 0}, {len(contour_prev) if contour_prev is not None else 0}")

            if contour is not None or (args.generate_transition and contour_prev is not None):
                to_break = True
            else:
                continue

            # Plot contour and rendered mask (as well as the previous frame contour and rendered mask) by cv2
            contour_img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            # Overlay the rendered mask and previous rendered mask
            if args.generate_transition:
                contour_img[rendered_mask_prev.cpu().numpy() > 0.5] = [0, 0, 100]  # Dark red for previous frame
                contour_img[rendered_mask.cpu().numpy() > 0.5] = [0, 100, 0]  # Dark green for current frame
            else:
                # render mask by white
                contour_img[rendered_mask.cpu().numpy() > 0.5] = [255, 255, 255]  # White for current frame
            for j in range(len(contour)):
                pts = contour[j]
                y, x = pts
                # downsampled contour are floats, so convert to int
                y, x = int(round(y)), int(round(x))
                cv2.circle(contour_img, (x,y), 1, (255,0,0), -1)  # Red in BGR

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
                torch.tensor([0.0, 0.0004, 0.007])
                .to(joint_angles.dtype)
                .to(model.device)
            )
            p_local4 = (
                torch.tensor([0.0, -0.0004, 0.007])
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
            
            # Plot p_img3 and p_img4 by cyan
            x3, y3 = p_img3
            x4, y4 = p_img4
            cv2.circle(contour_img, (int(x3.item()), int(y3.item())), 3, (255,255,0), -1)  # Cyan in BGR
            cv2.circle(contour_img, (int(x4.item()), int(y4.item())), 3, (255,255,0), -1)  # Cyan in BGR

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

            # Visualize the lines between ref_center and p_img1, p_img2
            cv2.line(contour_img, (int(ref_center[1]), int(ref_center[0])), (int(p_img1[0].item()), int(p_img1[1].item())), (0,255,0), 1)  # Green line
            cv2.line(contour_img, (int(ref_center[1]), int(ref_center[0])), (int(p_img2[0].item()), int(p_img2[1].item())), (0,255,0), 1)  # Green line
            
            # Step 3: Compute star-skeletonization signal and use it to determine keypoint visibility
            keypoints_mask = torch.zeros((proj_keypoints.shape[0],), dtype=torch.bool, device=proj_keypoints.device)

            signal = edge_signal(contour, ref_center)

            # Plot the reference center by yellow
            cv2.circle(contour_img, (int(ref_center[1]), int(ref_center[0])), 3, (0,255,255), -1)  # Yellow in BGR

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

            # Plot the local maxima on the contour image
            for idx in local_max_indices:
                yk, xk = contour[idx]
                cv2.circle(contour_img, (int(xk), int(yk)), 3, (0,255,255), -1)  # yellow in BGR
                cv2.line(contour_img, (int(ref_center[1]), int(ref_center[0])), (int(xk), int(yk)), (255,0,0), 1)  # Blue line

            # Check if projected keypoints are close to any local maxima
            if len(local_max_indices) > 0:
                for k in range(proj_keypoints.shape[0]):
                    xk, yk = proj_keypoints[k]
                    dists = np.linalg.norm(contour[local_max_indices] - np.array([yk.item(), xk.item()]), axis=1)
                    if np.min(dists) < 10:  # pixels
                        keypoints_mask[k] = True

            # Plot the projected keypoints on the contour image (Green for detectable, Red for undetectable)
            for k in range(proj_keypoints.shape[0]):
                xk, yk = proj_keypoints[k]
                color = (0, 255, 0) if keypoints_mask[k] else (0, 0, 255)  # Green if detectable, Red if undetectable
                cv2.circle(contour_img, (int(xk.item()), int(yk.item())), 3, color, -1)
                    
            # Choose the local maxima closest to each projected keypoint as the ground truth
            kpts_idx = -1 * torch.ones((proj_keypoints.shape[0],), dtype=torch.long)
            for k in range(proj_keypoints.shape[0]):
                if not keypoints_mask[k]:
                    continue
                xk, yk = proj_keypoints[k]
                dists = np.linalg.norm(contour[local_max_indices] - np.array([yk.item(), xk.item()]), axis=1)
                best_local_max_idx = local_max_indices[np.argmin(dists)]
                kpts_idx[k] = best_local_max_idx

            # # Choose the closest contour point as the ground truth keypoint index
            # kpts_idx = -1 * torch.ones((proj_keypoints.shape[0],), dtype=torch.long)
            # for k in range(proj_keypoints.shape[0]):
            #     xk, yk = proj_keypoints[k]
            #     dists = np.linalg.norm(contour - np.array([yk.item(), xk.item()]), axis=1)
            #     best_idx = np.argmin(dists)
            #     kpts_idx[k] = best_idx

            # # Choose the closest contour point as the ground truth keypoint index
            # # Use both the Euclidean distance and the ANGLE between rays
            # kpts_idx = -1 * torch.ones((proj_keypoints.shape[0],), dtype=torch.long)

            # # ---- precompute contour direction vectors ----
            # cont_vecs = contour - ref_center[None, :]
            # cont_norms = np.linalg.norm(cont_vecs, axis=1, keepdims=True)
            # cont_vecs = cont_vecs / (cont_norms + 1e-8)  # (N, 2)

            # for k in range(proj_keypoints.shape[0]):
            #     xk, yk = proj_keypoints[k]

            #     # ---- ray direction: center → keypoint ----
            #     ray_vec = np.array([
            #         yk.item() - ref_center[0],
            #         xk.item() - ref_center[1]
            #     ])
            #     ray_vec = ray_vec / (np.linalg.norm(ray_vec) + 1e-8)  # (2,)

            #     # ---- angular distance ----
            #     # dot(cont_vecs[j], ray_vec) for all j
            #     cos_angles = cont_vecs @ ray_vec              # (N,)
            #     cos_angles = np.clip(cos_angles, -1.0, 1.0)   # numerical safety
            #     angle_dists = np.arccos(cos_angles)           # (N,) in radians

            #     # ---- Euclidean distance ----
            #     euc_dists = np.linalg.norm(
            #         contour - np.array([yk.item(), xk.item()])[None, :],
            #         axis=1
            #     )  # (N,)

            #     # ---- choose best contour index ----
            #     # print(f"Keypoint {k}: min euc dist = {np.min(euc_dists):.2f}, min angle dist = {np.min(angle_dists)*180/np.pi:.2f} deg")
            #     best_idx = np.argmin(euc_dists + 0.7 * angle_dists)
            #     kpts_idx[k] = best_idx

            normals, curvature = compute_contour_properties(contour)
            if args.generate_transition:
                normals_prev, curvature_prev = compute_contour_properties(contour_prev)

            # Plot normals on the contour image
            for j in range(0, len(contour), 10):  # plot every 10th normal
                y, x = contour[j]
                nx, ny = normals[j]
                cv2.line(contour_img, (int(x), int(y)), (int(x + nx * 10), int(y + ny * 10)), (0, 255, 255), 1)  # yellow

            # # Determine the keypoints visibility based on distance threshold
            # keypoints_mask = torch.zeros((proj_keypoints.shape[0],), dtype=torch.bool, device=proj_keypoints.device)
            # for k in range(proj_keypoints.shape[0]):
            #     xk, yk = proj_keypoints[k]
            #     dist = np.linalg.norm(contour[kpts_idx[k].item()] - np.array([yk.item(), xk.item()]))
            #     # print(curvature[kpts_idx[k].item()])
            #     if dist < 5 and curvature[kpts_idx[k].item()] < -0.01:  # pixels
            #         keypoints_mask[k] = True

            # Plot reference 
            for k in range(proj_keypoints.shape[0]):
                yc, xc = contour[kpts_idx[k].item()]
                cv2.circle(contour_img, (int(xc), int(yc)), 4, (255,0,255), -1)  # Magenta in BGR
                # Label the curvature in green or red based on mask
                curvature_value = curvature[kpts_idx[k].item()]
                color = (0,255,0) if keypoints_mask[k] else (0,0,255)
                # print(f"Keypoint {k}: curvature = {curvature_value:.4f}, visibility = {keypoints_mask[k].item()}")
                cv2.putText(contour_img, f"{curvature_value:.2f}", (int(xc)+5, int(yc)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Plot projected keypoints and visibility
            for k in range(proj_keypoints.shape[0]):
                xk, yk = proj_keypoints[k]
                color = (0, 255, 0) if keypoints_mask[k] else (0, 0, 255)  # Green if detectable, Red if undetectable
                cv2.circle(contour_img, (int(xk.item()), int(yk.item())), 3, color, -1)

            # Save image
            cv2.imwrite(os.path.join(image_dir, f"{i:06d}.png"), contour_img)

            # Plot the raw rendered mask
            rendered_mask_img = (rendered_mask.cpu().numpy() * 255).astype(np.uint8)
            rendered_mask_img_color = cv2.cvtColor(rendered_mask_img, cv2.COLOR_GRAY2BGR)
            # print(f"Saving raw rendered mask to {os.path.join(image_raw_dir, f'{i:06d}.png')}")
            cv2.imwrite(os.path.join(image_raw_dir, f"{i:06d}.png"), rendered_mask_img_color)
                
            # Store the keypoitns label (a txt file with keypoint indices and visibility)
            label_path = os.path.join(label_dir, f"{i:06d}.txt")
            with open(label_path, "w") as f:
                for k in range(proj_keypoints.shape[0]):
                    f.write(f"{kpts_idx[k].item() if keypoints_mask[k] else -1}\n")

            # Store the feature (the contour points and distance to centroid) by txt file
            centroid = contour_centroid(contour)
            signal_c = edge_signal(contour, centroid)

            if args.generate_transition:
                centroid_prev = contour_centroid(contour_prev)
                signal_c_prev = edge_signal(contour_prev, centroid_prev)
            
            feature_path = os.path.join(feature_dir, f"{i:06d}.txt")
            with open(feature_path, "w") as f:
                for j in range(len(contour)):
                    y, x = contour[j]
                    dist = signal_c[j]
                    f.write(f"{y:.4f} {x:.4f} {dist:.4f} {normals[j,0]:.4f} {normals[j,1]:.4f} {curvature[j]:.4f} ")
                    if args.generate_transition:
                        y_prev, x_prev = contour_prev[j]
                        dist_prev = signal_c_prev[j]
                        f.write(f"{y_prev:.4f} {x_prev:.4f} {dist_prev:.4f} {normals_prev[j,0]:.4f} {normals_prev[j,1]:.4f} {curvature_prev[j]:.4f}\n")
                    else:
                        f.write("\n")

            if args.generate_transition:
                # Store the pose and joint angles of the previous frame and current frame
                pose_path = os.path.join(pose_dir, f"{i:06d}.txt")
                with open(pose_path, "w") as f:
                    for val in pose_vec_prev.cpu().numpy():
                        f.write(f"{val:.6f} ") # input
                    f.write("\n")
                    for val in pose_vec.cpu().numpy():
                        f.write(f"{val:.6f} ") # target
                    f.write("\n")

