import sys
import os
from skimage.morphology import skeletonize, medial_axis, convex_hull_image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffcali.models.CtRNet import CtRNet
from diffcali.utils.angle_transform_utils import (
    mix_angle_to_axis_angle,
    axis_angle_to_mix_angle,
    unscented_mix_angle_to_axis_angle,
    # unscented_mix_angle_to_local_quaternion,
    enforce_quaternion_consistency,
    enforce_axis_angle_consistency,
    find_local_quaternion_basis,
)

import math
import torch
import kornia
import nvdiffrast.torch as dr
from pytorch3d.renderer import MeshRasterizer
import numpy as np
import cv2
import argparse
import time
import warnings
import time

DF = 2

import torch.nn.functional as F

import torch
import torch.nn.functional as F


def parseArgs():     
    parser = argparse.ArgumentParser()
    data_dir = "data/synthetic/000000/PSM1/0"
    parser.add_argument("--data_dir", type=str, default=data_dir)  # reference mask
    parser.add_argument("--mesh_dir", type=str, default="urdfs/dVRK/meshes")
    parser.add_argument(
        "--joint_file", type=str, default=os.path.join(data_dir, "joint_0000.npy")
    )  # joint angles
    parser.add_argument(
        "--jaw_file", type=str, default=os.path.join(data_dir, "jaw_0000.npy")
    )  # jaw angles
    parser.add_argument("--arm", type=str, default="psm2")
    parser.add_argument("--sample_number", type=int, default=30)
    
    args = parser.parse_args()

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


def main1():
    with torch.no_grad():
        warnings.filterwarnings("ignore", category=UserWarning)
        # torch.manual_seed(42)
        # np.random.seed(42)

        args = parseArgs()
        
        joints = np.load(args.joint_file)
        jaw = np.load(args.jaw_file)

        """Or just for a single image processing"""

        # a.1) Build model
        model = CtRNet(args)
        mesh_files = [
            f"{args.mesh_dir}/shaft_multi_cylinder.ply",
            f"{args.mesh_dir}/logo_low_res_1.ply",
            f"{args.mesh_dir}/jawright_lowres.ply",
            f"{args.mesh_dir}/jawleft_lowres.ply",
        ]

        robot_renderer = model.setup_robot_renderer(mesh_files)
        robot_renderer.set_mesh_visibility([True, True, True, True])

        # a.2) Joint angles (same for all items, or replicate if needed)
        joint_angles_np = np.array(
            [joints[4], joints[5], jaw[0] / 2, jaw[0] / 2], dtype=np.float32
        )
        joint_angles = torch.tensor(
            joint_angles_np, device=model.device, requires_grad=False, dtype=torch.float32
        )
        model.get_joint_angles(joint_angles)

        robot_renderer.get_robot_mesh(joint_angles + 1) # warmup
        
        verts, faces, colors = robot_renderer.batch_get_robot_verts_and_faces(joint_angles.unsqueeze(0), ret_col=True) # warmup
        
        # Warm-up for more accurate timing
        start_time = time.time()
        robot_mesh = robot_renderer.get_robot_mesh(joint_angles)
        end_time = time.time()
        print(f"Mesh computing time: {(end_time - start_time) * 1000 :.4f} ms")
        

        joint_angles = joint_angles.unsqueeze(0)  # (1, 4)
        start_time = time.time()
        verts, faces, colors = robot_renderer.batch_get_robot_verts_and_faces(joint_angles, ret_col=True)
        end_time = time.time()
        print(f"Batch mesh computing time: {(end_time - start_time) * 1000 :.4f} ms")
        print(f"Verts shape: {verts.shape}, Faces shape: {faces.shape}, Colors shape: {colors.shape}")

        # a.3) Generate all initial cTr in some way (N total). For demo, let's do random.
        N = args.sample_number
        cTr_inits = []
        for i in range(N):
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
            if not torch.any(torch.isnan(cTr)):
                cTr_inits.append(cTr)
        cTr_batch = torch.cat(cTr_inits, dim=0) # All samples in a single batch
        B = cTr_batch.shape[0] 
        print(f"All ctr candiates: {cTr_batch.shape}")

        # a.4) Render silhouette shaders
        # warm-up for more accurate timing
        pred_masks = model.render_robot_mask_batch(
            cTr_batch, robot_mesh, robot_renderer
        ) # shape is [B, H, W]

        start_time = time.time()
        pred_masks = model.render_robot_mask_batch(
            cTr_batch, robot_mesh, robot_renderer
        ) # shape is [B, H, W]
        end_time = time.time()

        print(f"Predicted masks: {pred_masks.shape}")
        print(f"Batch Rendering Time (PyTorch3D): {(end_time - start_time) * 1000 :.4f} ms")

        # b.1) Configure NvDiffRast renderer
        glctx = dr.RasterizeCudaContext() # CUDA context (OpenGL is not available in my WSL)
        resolution = (args.height // DF, args.width // DF)
        original_resolution = (args.height, args.width)

        # b.2) Prepare data for rendering (using utils in PyTorch3D)

        R_batched = kornia.geometry.conversions.angle_axis_to_rotation_matrix(
            cTr_batch[:, :3]
        ) 
        R_batched = R_batched.transpose(1, 2)
        T_batched = cTr_batch[:, 3:] 
        negative_mask = T_batched[:, -1] < 0  #flip where negative_mask is True
        T_batched_ = T_batched.clone()
        T_batched_[negative_mask] = -T_batched_[negative_mask]
        R_batched_ = R_batched.clone()
        R_batched_[negative_mask] = -R_batched_[negative_mask]
        pos, pos_idx = transform_mesh(
            cameras=robot_renderer.cameras, mesh=robot_mesh.extend(B),
            R=R_batched_, T=T_batched_, args=args
        ) # project the batched meshes in the clip space
        
        # Check if all instance in pos_idx are the same
        for i in range(1, len(pos_idx)):
            assert torch.all(pos_idx[0] == pos_idx[i]), "Different instance indices in the batch"
        
        # b.3) Render the silhouette images

        import torch.nn.functional as F 
        pred_masks_nv = render(glctx, pos, pos_idx[0], resolution, col=None) # (B, H, W, 3)
        # color = pred_masks_nv.permute(0, 3, 1, 2)      # → (B,3,h,w)
        # color = F.interpolate(color, size=original_resolution, mode='bilinear')
        # pred_masks_nv = color

        # print(pred_masks_nv.sum(dim=(1))[0].max()) # print sum of each channel to see if they are different

        start_time = time.time()
        pred_masks_nv = render(glctx, pos, pos_idx[0], resolution, col=None) # (B, H, W)
        pred_masks_nv = F.interpolate(
            pred_masks_nv.unsqueeze(1), size=original_resolution, mode='bilinear'
        ).squeeze(1)
        # color = pred_masks_nv.permute(0, 3, 1, 2)      # → (B,3,h,w)
        # color = F.interpolate(color, size=original_resolution, mode='bilinear')
        # pred_masks_nv = color
        end_time = time.time()

        # # Project the origin (the translation vector) to the image plane
        # fx, fy = args.fx, args.fy
        # px, py = args.px, args.py

        # # View-space coordinates of the origin in camera frame (R @ 0 + T)
        # origin_camera = T_batched_  # (B, 3)

        # # Project to pixel coordinates
        # x = fx * (origin_camera[:, 0] / origin_camera[:, 2]) + px
        # y = fy * (origin_camera[:, 1] / origin_camera[:, 2]) + py

        # origin_proj = torch.stack([x, y], dim=-1)  # shape [B, 2]
        # origin_proj_int = origin_proj.round().to(torch.int32)

        fx, fy = args.fx, args.fy
        px, py = args.px, args.py

        axis_len = 0.02  # 2cm

        print(f"Predicted masks (NvDiffRast): {pred_masks_nv.shape}")
        print(f"Batch Rendering Time (NvDiffRast): {(end_time - start_time) * 1000 :.4f} ms")

        # Convert PyTorch3D masks
        pred_masks_np = (pred_masks.cpu().numpy() * 255).astype(np.uint8)
        pred_masks_np = np.stack([pred_masks_np]*3, axis=-1)

        # Convert NvDiffRast masks
        pred_masks_nv = (pred_masks_nv.cpu().numpy() * 255).astype(np.uint8)
        pred_masks_nv = np.stack([pred_masks_nv]*3, axis=-1)

        for i in range(min(10, args.sample_number)):

            overlay = cv2.addWeighted(
                pred_masks_np[i], 0.5, pred_masks_nv[i], 0.5, 0
            )

            # =============================
            # Get transform
            # =============================
            R = R_batched[i].T.cpu().numpy()       # (3,3)
            T = T_batched[i].cpu().numpy().reshape(3)

            # =============================
            # Frame in CAMERA coordinates
            # =============================
            origin_cam = T

            x_cam = T + R[:, 0] * axis_len
            y_cam = T + R[:, 1] * axis_len
            z_cam = T + R[:, 2] * axis_len

            pts_cam = np.stack([origin_cam, x_cam, y_cam, z_cam], axis=0)

            # =============================
            # If using PyTorch3D renderer
            # uncomment this if needed
            # =============================
            # pts_cam *= -1

            # =============================
            # Project
            # =============================
            X = pts_cam[:, 0]
            Y = pts_cam[:, 1]
            Z = pts_cam[:, 2] + 1e-9

            u = fx * (X / Z) + px
            v = fy * (Y / Z) + py

            pts_2d = np.stack([u, v], axis=-1).astype(int)

            origin_2d = tuple(pts_2d[0])
            x_2d = tuple(pts_2d[1])
            y_2d = tuple(pts_2d[2])
            z_2d = tuple(pts_2d[3])

            cv2.line(overlay, origin_2d, x_2d, (0, 0, 255), 3)   # X
            cv2.line(overlay, origin_2d, y_2d, (0, 255, 0), 3)   # Y
            cv2.line(overlay, origin_2d, z_2d, (255, 0, 0), 3)   # Z

            cv2.imshow(f"Sample {i}", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main2():
    with torch.no_grad():
        warnings.filterwarnings("ignore", category=UserWarning)
        # torch.manual_seed(42)
        # np.random.seed(42)

        args = parseArgs()
        
        joints = np.load(args.joint_file)
        jaw = np.load(args.jaw_file)

        """Or just for a single image processing"""

        # a.1) Build model
        model = CtRNet(args)
        model.args.use_nvdiffrast = True
        model.resolution = (args.height // DF, args.width // DF)
        original_resolution = (args.height, args.width)
        mesh_files = [
            f"{args.mesh_dir}/low_res_shaft_multi_cylinder.ply",
            f"{args.mesh_dir}/low_res_logo_low_res_1.ply",
            f"{args.mesh_dir}/low_res_jawright_lowres.ply",
            f"{args.mesh_dir}/low_res_jawleft_lowres.ply",
        ]

        robot_renderer = model.setup_robot_renderer(mesh_files)
        robot_renderer.set_mesh_visibility([True, True, True, True])

        # a.2) Joint angles (same for all items, or replicate if needed)
        joint_angles_np = np.array(
            [joints[4], joints[5], jaw[0] / 2, jaw[0] / 2], dtype=np.float32
        )
        joint_angles = torch.tensor(
            joint_angles_np, device=model.device, requires_grad=False, dtype=torch.float32
        )

        joint_angles2 = torch.tensor(
            joint_angles_np + 0.1, device=model.device, requires_grad=False, dtype=torch.float32
        )
        joint_angles[2:] = 0.

        # a.3) Generate all initial cTr in some way (N total). For demo, let's do random.
        N = args.sample_number * 2
        cTr_inits = []
        for i in range(N):
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
            if not torch.any(torch.isnan(cTr)):
                cTr_inits.append(cTr)
        cTr_batch = torch.cat(cTr_inits, dim=0) # All samples in a single batch
        B = cTr_batch.shape[0] 
        print(f"All ctr candiates: {cTr_batch.shape}")

        # b.1) Configure NvDiffRast renderer
        glctx = dr.RasterizeCudaContext() # CUDA context (OpenGL is not available in my WSL)

        # Test consistency of bi-manual rendering
        joint_angles_bi_manual = torch.cat(
            [
                joint_angles.repeat(args.sample_number, 1).unsqueeze(1), # (N, 1, 4)
                joint_angles2.repeat(args.sample_number, 1).unsqueeze(1)  # (N, 1, 4)
            ],
            dim=1
        ) # (N, 2, 4)
        print(f"Joint angles bi-manual shape: {joint_angles_bi_manual.shape}")
        verts, faces, R_list, t_list = robot_renderer.batch_get_robot_verts_and_faces(
            joint_angles_bi_manual, ret_lndFK=True, ret_col=False, bi_manual=True
        ) 

        print(f"R_list shape: {R_list.shape}, t_list shape: {t_list.shape}")

        # Render the bi-manual meshes (cTr_batch.shape = [n_samples, 2, 6], joint_angles_bi_manual.shape = [B, 2, 4])
        B = joint_angles_bi_manual.shape[0]
        cTr_batch_bi_manual = cTr_batch.view(B, 2, 6)
        pred_masks_bi_manual = model.render_robot_mask_batch_nvdiffrast(
            cTr_batch_bi_manual, verts, faces, robot_renderer, bi_manual=True
        ) # shape is [B, H, W]

        print(f"Predicted bi-manual masks (NvDiffRast): {pred_masks_bi_manual.shape}")

        start_time = time.time()
        pred_masks_bi_manual = model.render_robot_mask_batch_nvdiffrast(
            cTr_batch_bi_manual, verts, faces, robot_renderer, bi_manual=True
        ) # shape is [B, H, W]
        pred_masks_bi_manual = F.interpolate(
            pred_masks_bi_manual.unsqueeze(1), size=original_resolution, mode='bilinear'
        ).squeeze(1)
        end_time = time.time()

        bi_manual_time = end_time - start_time

        print(f"Batch Bi-manual Rendering Time (NvDiffRast): {bi_manual_time * 1000 :.4f} ms")

        # Render each arm separately and combine the results
        cTr_batch_left = cTr_batch_bi_manual[:, 0, :] # (N, 6)
        cTr_batch_right = cTr_batch_bi_manual[:, 1, :] # (N, 6)
        joint_angles_left = joint_angles.repeat(args.sample_number, 1) # (N, 4)
        joint_angles_right = joint_angles2.repeat(args.sample_number, 1) # (N, 4)

        verts_left, faces_left, _, _ = robot_renderer.batch_get_robot_verts_and_faces(
            joint_angles_left, ret_lndFK=True, ret_col=False, bi_manual=False  
        )
        pred_masks_left = model.render_robot_mask_batch_nvdiffrast(
            cTr_batch_left, verts_left, faces_left, robot_renderer, bi_manual=False
        ) # shape is [B, H, W]
        verts_right, faces_right, _, _ = robot_renderer.batch_get_robot_verts_and_faces(
            joint_angles_right, ret_lndFK=True, ret_col=False, bi_manual=False  
        )
        pred_masks_right = model.render_robot_mask_batch_nvdiffrast(
            cTr_batch_right, verts_right, faces_right, robot_renderer, bi_manual=False
        ) # shape is [B, H, W]

        start_time = time.time()
        verts_left, faces_left, _, _ = robot_renderer.batch_get_robot_verts_and_faces(
            joint_angles_left, ret_lndFK=True, ret_col=False, bi_manual=False  
        )
        pred_masks_left = model.render_robot_mask_batch_nvdiffrast(
            cTr_batch_left, verts_left, faces_left, robot_renderer, bi_manual=False
        ) # shape is [B, H, W]
        verts_right, faces_right, _, _ = robot_renderer.batch_get_robot_verts_and_faces(
            joint_angles_right, ret_lndFK=True, ret_col=False, bi_manual=False  
        )
        pred_masks_right = model.render_robot_mask_batch_nvdiffrast(
            cTr_batch_right, verts_right, faces_right, robot_renderer, bi_manual=False
        ) # shape is [B, H, W]
        pred_masks_left = F.interpolate(
            pred_masks_left.unsqueeze(1), size=original_resolution, mode='bilinear'
        ).squeeze(1)
        pred_masks_right = F.interpolate(
            pred_masks_right.unsqueeze(1), size=original_resolution, mode='bilinear'
        ).squeeze(1)
        end_time = time.time()
        separate_time = end_time - start_time

        print(f"Batch Separate Rendering Time (NvDiffRast): {separate_time * 1000 :.4f} ms")

        # Oberlay the two masks and display to check consistency
        pred_masks_combined = torch.clamp(pred_masks_left + pred_masks_right, 0, 1)
        pred_masks_bi_manual_np = (pred_masks_bi_manual.cpu().numpy() * 255).astype(np.uint8)  # (B, H, W)
        pred_masks_combined_np = (pred_masks_combined.cpu().numpy() * 255).astype(np.uint8)  # (B, H, W)
        # Make 3-channel images for visualization (blue for bi-manual, red for separate)
        pred_masks_bi_manual_np = np.stack([pred_masks_bi_manual_np]*3, axis=-1)  # (B, H, W, 3)
        pred_masks_bi_manual_np[..., 1:] = 0  # Keep only blue channel
        pred_masks_combined_np = np.stack([pred_masks_combined_np]*3, axis=-1)  # (B, H, W, 3)
        pred_masks_combined_np[..., :2] = 0  # Keep only red channel
        for i in range(min(10, args.sample_number)):
            # Compare the two rendering results by overlaying them
            overlay_bi_manual = cv2.addWeighted(
                pred_masks_bi_manual_np[i], 0.5, pred_masks_combined_np[i], 0.5, 0
            )
            cv2.imshow(f"Overlay Bi-manual vs Separate Sample {i}", overlay_bi_manual)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main1()
    # main2()