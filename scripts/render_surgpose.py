import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffcali.models.CtRNet import CtRNet

import torch
import nvdiffrast.torch as dr
import numpy as np
import cv2
import tqdm
from PIL import Image
import yaml


def dh_transform(a, alpha, D, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.0,   sa,     ca,     D],
        [0.0,  0.0,    0.0,   1.0]
    ])


def fk_psm_frame4(q1, q2, q3, q4):
    """
    Forward kinematics of the PSM up to frame 4
    using joints q1–q4.

    Returns:
        T_0_4 : 4x4 homogeneous transform
    """
    l_RCC  = 0.4318   # m
    l_tool = 0.4162   # m

    T01 = dh_transform(
        a=0.0,
        alpha=np.pi/2,
        D=0.0,
        theta=q1 + np.pi/2
    )

    T12 = dh_transform(
        a=0.0,
        alpha=-np.pi/2,
        D=0.0,
        theta=q2 - np.pi/2
    )

    T23 = dh_transform(
        a=0.0,
        alpha=np.pi/2,
        D=q3 - l_RCC,   # prismatic joint
        theta=0.0
    )

    T34 = dh_transform(
        a=0.0,
        alpha=0.0,
        D=l_tool,
        theta=q4
    )

    T_0_4 = T01 @ T12 @ T23 @ T34
    return T_0_4


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
    print(col.shape, pos.shape, pos_idx.shape)
    col_idx = pos_idx
    
    # Render the mesh
    rast_out, _ = dr.rasterize(glctx, pos, pos_idx, resolution=resolution, grad_db=False)
    color   , _ = dr.interpolate(col, rast_out, col_idx)
    if antialiasing:
        color = dr.antialias(color, rast_out, pos, pos_idx)
    return color.squeeze(-1) # (B, H, W)


def parseArgs():     
    import argparse

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

    args.height = 986
    args.width = 1400
    args.fx, args.fy, args.px, args.py = 1811.910046453570, 1809.640734154330, 588.5594517681759, 477.3975900383616
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


data_dir = "./data/surgpose/000000/original_data/"
pose_filename = "api_cp_data.yaml"
joint_filename = "api_jp_data.yaml"
machine_name = 'PSM3'

if __name__ == "__main__":
    with open(os.path.join(data_dir, joint_filename), 'r') as f:
        joint_data = yaml.safe_load(f)
    raw_joint_angles_lst = []
    for i in range(len(joint_data)):
        joint_i = joint_data[str(i)][machine_name]
        # Extract wrist pitch, wrist yaw, jaw
        q1, q2, q3, q4, q5, q6 = joint_i
        raw_joint_angles_lst.append(np.array([q1, q2, q3, q4, q5, q6], dtype=np.float32))

    # Load pose data yaml file
    with open(os.path.join(data_dir, pose_filename), 'r') as f:
        pose_data = yaml.safe_load(f)
    pose_matrix_lst = []
    joint_angles_lst = []
    for i in range(len(pose_data)):
        pose_i = pose_data[str(i)]
        R_i = pose_i[machine_name]['R']
        T_i = pose_i[machine_name]['t']
        pose_matrix_i = np.eye(4, dtype=np.float32)
        pose_matrix_i[:3, :3] = np.array(R_i, dtype=np.float32).reshape(3, 3)
        pose_matrix_i[:3, 3] = np.array(T_i, dtype=np.float32).reshape(3)

        # Use forward kinematics to compute frame 4 pose
        q1, q2, q3, q4, _, _ = raw_joint_angles_lst[i]
        T_0_4 = fk_psm_frame4(q1, q2, q3, q4)

        T_W_4 = pose_matrix_i @ T_0_4

        pose_matrix_lst.append(T_W_4)

        # Extract wrist pitch, wrist yaw, jaw
        wrist_pitch = raw_joint_angles_lst[i][4]
        wrist_yaw = raw_joint_angles_lst[i][5]
        jaw = 0.
        joint_angles_lst.append(np.array([wrist_pitch, wrist_yaw, jaw, jaw], dtype=np.float32))

    N = len(pose_matrix_lst)
    pose_matrix_seq = torch.tensor(np.stack(pose_matrix_lst, axis=0), dtype=torch.float32).cuda()  # (N, 4, 4)
    joint_angles_seq = torch.tensor(np.stack(joint_angles_lst, axis=0), dtype=torch.float32).cuda()  # (N, 3)

    # # Use a fixed joint angle sequence for debugging
    # joint_angles_seq = torch.tensor(np.tile(
    #     np.array([[0.0, 0.0, 0.5, 0.5]], dtype=np.float32), (N, 1)
    # ), dtype=torch.float32).cuda()  # (N, 3)

    # Load model and setup renderer
    args = parseArgs()

    model = CtRNet(args).cuda()
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
 
    with torch.no_grad():
        rendered_masks = []

        pbar = tqdm.tqdm(range(N), desc="Generating synthetic data...")
        for i in pbar:
            joint_angles  = joint_angles_seq[i]

            model.get_joint_angles(joint_angles)
            robot_mesh = robot_renderer.get_robot_mesh(joint_angles)

            R_batched = pose_matrix_seq[i, :3, :3].unsqueeze(0)
            R_batched = R_batched.transpose(1, 2)
            T_batched = pose_matrix_seq[i, :3, 3].unsqueeze(0)
            negative_mask = T_batched[:, -1] < 0  #flip where negative_mask is True
            T_batched_ = T_batched.clone()
            T_batched_[negative_mask] = -T_batched_[negative_mask]
            R_batched_ = R_batched.clone()
            R_batched_[negative_mask] = -R_batched_[negative_mask]
            # R_batched_ = R_batched
            # T_batched_ = T_batched
            pos, pos_idx = transform_mesh(
                cameras=robot_renderer.cameras, mesh=robot_mesh.extend(1),
                R=R_batched_, T=T_batched_, args=args
            ) # project the batched meshes in the clip 
            
            rendered_mask = render(glctx, pos, pos_idx[0], resolution)[0] # shape (H, W)

            # Just project the base point to the image plane for debug
            rendered_mask_np = rendered_mask.cpu().numpy()
            point_cam = R_batched_[0] @ torch.tensor([[0.0], [0.0], [0.0]], dtype=torch.float32).cuda() + T_batched_[0]  # (3, 1)
            x = (point_cam[0, 0] / point_cam[2, 0]) * args.fx + args.px
            y = (point_cam[1, 0] / point_cam[2, 0]) * args.fy + args.py
            x = int(np.round(x.item()))
            y = int(np.round(y.item()))
            print(f"Frame {i}: projected base point at pixel ({x}, {y})")
            cv2.circle(rendered_mask_np, (x, y), 5, (128,), -1)

            rendered_masks.append(rendered_mask_np)

    # Save rendered masks as mp4
    rendered_masks_np = np.stack(rendered_masks, axis=0)  # (N, H, W)
    rendered_masks_np = (rendered_masks_np > 0.5).astype(np.uint8) * 255

    gif_path = os.path.join(data_dir, "surgpose_synthetic_mask.gif")

    frames = [Image.fromarray(rendered_masks_np[i]) for i in range(rendered_masks_np.shape[0])]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )