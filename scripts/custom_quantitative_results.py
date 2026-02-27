import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch as th
import pandas as pd
import numpy as np
import nvdiffrast.torch as dr

import argparse

from diffcali.models.CtRNet import CtRNet
from scripts.single_arm_quantitative_results import (
    evaluate_surgpose_trajectory,
    evaluate_synthetic_trajectory,
    parseArgs,
)
import tqdm
import glob
import cv2
import kornia


def parseCtRNetArgs():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.use_gpu = True
    args.trained_on_multi_gpus = False

    args.arm = "psm2"

    # Camera intrinsics
    # [ 1.02588223e+03,  0.0,  1.67919017e+02,
    #    0.0, 1.02588223e+03, 2.34152707e+02,
    #    0., 0., 1. ]

    # Setting for our custom data
    args.height = 480
    args.width = 640
    args.fx, args.fy, args.px, args.py = 1025.88223, 1025.88223, 167.919017, 234.152707

    # clip space parameters
    args.znear = 1e-3
    args.zfar = 1e9

    args.scale = 1.0

    # scale the camera parameters
    args.width = int(args.width * args.scale)
    args.height = int(args.height * args.scale)
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale

    return args


RESULT_DIR = "./pf_tracking_results"
VIDEO_LABELS = ["bag1", "bag2", "bag3", "bag4", "bag5", "bag6"]

def load_single(label, method="cma_es"):
    assert method in ["cma_es", "pf"], "Unsupported method, please choose from 'cma_es' or 'pf'."
    path = os.path.join(RESULT_DIR, f"{method}_{label}_tracking_results.pt")

    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return None

    data = torch.load(path, map_location="cpu")

    cTr = data["cTr"]                # (num_frames, 6)
    joint = data["joint_angles"]     # (num_frames, num_joints)
    time = data["time"]              # (num_frames-1,)

    cTr = cTr[1:,:] # Exclude the first frame which is initialization
    joint = joint[1:,:] # Exclude the first frame which is initialization
    if method == "pf":
        time = time[1:]  # Exclude the first frame which is initialization

    print(f"\nLoaded {label} using {method}:")
    print(f"  cTr shape: {cTr.shape}")
    print(f"  Joint shape: {joint.shape}")
    print(f"  Time shape: {time.shape}")
    print(f"  Avg time per frame: {time.mean().item():.4f} s")

    return {
        "label": label,
        "cTr": cTr,
        "joint": joint,
        "time": time
    }


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


def evaluate_trajectory(cTr_seq, joint_seq, time_seq, label, model, robot_renderer, glctx, args):
    cTr_seq = cTr_seq.cuda()
    joint_seq = joint_seq.cuda()

    # Render the trajectory and compute IoU with the ground truth
    gt_mask_dir = f"./data/custom/gt_masks/{label}/PSM3/"
    mask_ious = []

    frame_start = 1
    frame_end = len([name for name in os.listdir(gt_mask_dir) if os.path.isdir(os.path.join(gt_mask_dir, name)) and name.isdigit()])
    frame_end = min(frame_end, cTr_seq.shape[0]) # Ensure we don't go out of bounds of the trajectory data

    pbar = tqdm.tqdm(range(frame_start, frame_end), desc=f"Evaluating {label}")
    for frame_idx in pbar:
        frame_dir = os.path.join(gt_mask_dir, f"{frame_idx}")

        # Find the mask
        mask_lst = glob.glob(os.path.join(frame_dir, "*.png"))
        if len(mask_lst) == 0:
            print(f"No mask found in {frame_dir}")
            continue
        if len(mask_lst) > 1:
            print(f"Multiple masks found in {frame_dir}")
            continue

        mask_path = mask_lst[0]
        # frame = cv2.imread(frame_path)
        XXXX = mask_path.split("/")[-1].split(".")[0][1:]

        # Read ref_img_file of name 0XXXX.jpg
        ref_mask_path = os.path.join(frame_dir, "0" + XXXX + ".png")
        ref_img = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)

        if ref_img is None:
            print(f"No ref_img found in {frame_dir}")
            continue
        ref_mask = (ref_img / 255.0).astype(np.float32)
        ref_mask = th.tensor(ref_mask, requires_grad=False, dtype=th.float32).cuda()

        # Render the mask from cTr_seq and joint_seq
        cTr = cTr_seq[frame_idx-1].clone()
        joint_angles  = joint_seq[frame_idx-1]

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
        
        resolution = (args.height, args.width)
        rendered_mask = render(glctx, pos, pos_idx[0], resolution)[0] # shape (H, W)
        rendered_mask = (rendered_mask > 0.5).float()

        # Compute IoU with the ground truth mask
        inter = torch.logical_and(rendered_mask > 0,
                                  ref_mask > 0).sum().float()
        union = torch.logical_or(rendered_mask > 0,
                                 ref_mask > 0).sum().float()
        assert union > 0, "Union is zero, check the masks!"
        iou = (inter / union).item()
        mask_ious.append(iou)

        # Display the overlay
        overlay = cv2.addWeighted((rendered_mask.cpu().numpy() * 255).astype(np.uint8), 0.5, (ref_mask.cpu().numpy() * 255).astype(np.uint8), 0.5, 0)
        cv2.imshow(f"Rendered Mask vs GT Mask - {label}", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    mean_mask_error = 1.0 - np.mean(mask_ious) if mask_ious else None

    # runtime: average after first 10 frames
    time_seq = np.array(time_seq)
    if len(time_seq) > 10:
        avg_time = np.mean(time_seq[10:])
    else:
        avg_time = np.mean(time_seq)

    return mean_mask_error, avg_time



if __name__ == "__main__":
    # Load results for all videos and both methods
    all_results = {}
    for label in VIDEO_LABELS:
        cma_es_result = load_single(label, method="cma_es")
        pf_result = load_single(label, method="pf")
        all_results[label] = {
            "cma_es": cma_es_result,
            "pf": pf_result
        }

    args = parseCtRNetArgs()

    args.use_nvdiffrast = True

    model = CtRNet(args)
    mesh_files = [
        "urdfs/dVRK/meshes/shaft_multi_cylinder.ply",
        "urdfs/dVRK/meshes/logo_low_res_1.ply",
        "urdfs/dVRK/meshes/jawright_lowres.ply",
        "urdfs/dVRK/meshes/jawleft_lowres.ply",
    ]

    robot_renderer = model.setup_robot_renderer(mesh_files)
    robot_renderer.set_mesh_visibility([True, True, True, True])

    glctx = dr.RasterizeCudaContext() # CUDA context (OpenGL is not available in my WSL)
    resolution = (args.height, args.width)

    # Evaluate each trajectory and average across videos for each method
    cma_es_errors = []
    pf_errors = []
    cma_es_times = []
    pf_times = []
    for label, results in all_results.items():
        if results["cma_es"] is not None:
            cma_es_error, cma_es_time = evaluate_trajectory(
                results["cma_es"]["cTr"], results["cma_es"]["joint"], results["cma_es"]["time"],
                label, model, robot_renderer, glctx, args
            )
            cma_es_errors.append(cma_es_error)
            cma_es_times.append(cma_es_time)
            print(f"{label} CMA-ES: Mean Mask Error={cma_es_error:.4f}, Avg Time={cma_es_time:.4f} s")

        if results["pf"] is not None:
            pf_error, pf_time = evaluate_trajectory(
                results["pf"]["cTr"], results["pf"]["joint"], results["pf"]["time"],
                label, model, robot_renderer, glctx, args
            )
            pf_errors.append(pf_error)
            pf_times.append(pf_time)
            print(f"{label} Particle Filter: Mean Mask Error={pf_error:.4f}, Avg Time={pf_time:.4f} s")

    # print("\nOverall Results:")
    # if cma_es_errors:
    #     print(f"CMA-ES Average Mean Mask Error: {np.mean(cma_es_errors):.4f}")
    #     print(f"CMA-ES Average FPS (excluding initialization): {1.0 / np.mean([t for t in all_results[label]['cma_es']['time'][10:]]):.2f} FPS")
    # else:
    #     print("No CMA-ES results to average.")

    # if pf_errors:
    #     print(f"Particle Filter Average Mean Mask Error: {np.mean(pf_errors):.4f}")
    #     print(f"Particle Filter Average FPS (excluding initialization): {1.0 / np.mean([t for t in all_results[label]['pf']['time'][10:]]):.2f} FPS")
    # else:
    #     print("No Particle Filter results to average.")
    
    print("\nOverall Results:")
    if cma_es_errors:
        print(f"CMA-ES Average Mean Mask Error: {np.mean(cma_es_errors):.4f} ± {np.std(cma_es_errors):.4f}")
        print(f"CMA-ES Average FPS (excluding initialization): {np.mean(1 / np.array(cma_es_times)): .4f} ± {np.std(1 / np.array(cma_es_times)): .4f} FPS")
    else:
        print("No CMA-ES results to average.")

    if pf_errors:
        print(f"Particle Filter Average Mean Mask Error: {np.mean(pf_errors):.4f} ± {np.std(pf_errors):.4f}")
        print(f"Particle Filter Average FPS (excluding initialization): {np.mean(1 / np.array(pf_times)): .4f} ± {np.std(1 / np.array(pf_times)): .4f} FPS")
    else:
        print("No Particle Filter results to average.")

    # Print trajectory lengths
    print("\nTrajectory Lengths:")
    traj_lengths = []
    for label, results in all_results.items():
        lengths = []
        if results["cma_es"] is not None:
            lengths.append(results["cma_es"]["cTr"].shape[0])
        if results["pf"] is not None:
            lengths.append(results["pf"]["cTr"].shape[0])
        if lengths:
            traj_lengths.extend(lengths)
            print(f"  {label}: {lengths}")
    
    if traj_lengths:
        print(f"  Min: {min(traj_lengths)}, Max: {max(traj_lengths)}, Mean: {np.mean(traj_lengths):.0f}")
    



