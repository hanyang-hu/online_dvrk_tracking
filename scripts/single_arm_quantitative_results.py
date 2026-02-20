import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import nvdiffrast.torch as dr
import argparse
import kornia
import glob
import cv2
import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from diffcali.models.CtRNet import CtRNet
from diffcali.eval_dvrk.LND_fk import lndFK, batch_lndFK
from diffcali.utils.projection_utils import *
from diffcali.utils.ui_utils import *
from diffcali.utils.angle_transform_utils import mix_angle_to_axis_angle, axis_angle_to_mix_angle, mix_angle_to_rotmat

POSE_DIR = "./pose_results"


def parse_single_arm_filename(filename):
    """
    Format:
    {data_label}.{searcher}.{online_iters}.{joint_str}.
    {pts_loss_str}.{kpts_det_str}.{app_loss_str}.
    {filter_str}.{renderer_str}.pth
    """

    name = os.path.basename(filename).replace(".pth", "")
    parts = name.split(".")

    if len(parts) != 9:
        return None  # skip malformed

    return dict(
        data_label=parts[0],
        searcher=parts[1],
        online_iters=int(parts[2]),
        joint_label=parts[3],
        pts_loss=parts[4],
        kpts_det=parts[5],
        app_loss=parts[6],
        filter=parts[7],
        renderer=parts[8],
    )


def collect_results(model, robot_renderer, glctx, args, evaluate_surgpose):

    surgpose_rows = []
    synthetic_rows = []

    pbar = tqdm.tqdm(os.listdir(POSE_DIR), desc=f"Collecting results from {POSE_DIR}")
    for file in os.listdir(POSE_DIR):

        if not file.endswith(".pth"):
            continue

        full_path = os.path.join(POSE_DIR, file)
        meta = parse_single_arm_filename(file)
        if meta is None:
            continue

        data = torch.load(full_path)
        data_label = meta["data_label"]

        # reconstruct data_dir from label:
        # surgpose_000000_PSM1
        parts = data_label.split("_")
        base, seq, arm = parts
        data_dir = f"./data/{base}/{seq}/{arm}/"

        # ---------------------------------------------------
        # SURGPOSE
        # ---------------------------------------------------
        # print(base, evaluate_surgpose)
        if base == "surgpose" and evaluate_surgpose:

            mask_err, kpt_err, avg_time = evaluate_surgpose_trajectory(
                data_dir,
                data["cTr"],
                data["joint_angles"],
                time_seq=data["time"],
                model=model,
                robot_renderer=robot_renderer,
                glctx=glctx,
                args=args,
            )

            row = {
                **meta,
                "data_dir": data_dir,
                "mask_proj_error": mask_err,
                "keypoint_error": kpt_err,
                "avg_runtime": avg_time,
                "avg_runtime_per_iter":
                    avg_time / meta["online_iters"]
                    if meta["online_iters"] > 0 else None,
            }

            surgpose_rows.append(row)

        # ---------------------------------------------------
        # SYNTHETIC
        # ---------------------------------------------------
        elif base == "synthetic":

            rot_err, trans_err, joint_theta1_err, joint_theta2_err, joint_theta3_err, avg_time = \
                evaluate_synthetic_trajectory(
                    data_dir,
                    data["cTr"],
                    data["joint_angles"],
                    time_seq=data["time"],
                )

            row = {
                **meta,
                "data_dir": data_dir,
                "rotation_error": rot_err,
                "translation_error": trans_err,
                "joint_theta1_error": joint_theta1_err,
                "joint_theta2_error": joint_theta2_err,
                "joint_theta3_error": joint_theta3_err,
                "avg_runtime": avg_time,
                "avg_runtime_per_iter":
                    avg_time / meta["online_iters"]
                    if meta["online_iters"] > 0 else None,
            }

            synthetic_rows.append(row)

        pbar.update(1)

    # =======================================================
    # Build DataFrames
    # =======================================================
    surgpose_df = pd.DataFrame(surgpose_rows)
    synthetic_df = pd.DataFrame(synthetic_rows)

    # Sort for clarity
    if not surgpose_df.empty:
        surgpose_df = surgpose_df.sort_values(
            by=["data_label", "searcher", "online_iters"]
        )

    if not synthetic_df.empty:
        synthetic_df = synthetic_df.sort_values(
            by=["data_label", "searcher", "online_iters"]
        )

     # -------------------------------------------------------
    # SORT (raw results)
    # -------------------------------------------------------
    if not surgpose_df.empty:
        surgpose_df = surgpose_df.sort_values(
            by=["data_label", "searcher", "online_iters"]
        )

    if not synthetic_df.empty:
        synthetic_df = synthetic_df.sort_values(
            by=["data_label", "searcher", "online_iters"]
        )

    # Save raw tables
    if evaluate_surgpose:
        surgpose_df.to_csv("./single_arm_surgpose_raw.csv", index=False)
    synthetic_df.to_csv("./single_arm_synthetic_raw.csv", index=False)

    # =======================================================
    # AGGREGATE ACROSS DATA_LABEL
    # =======================================================

    group_cols = [
        "searcher",
        "online_iters",
        "joint_label",
        "pts_loss",
        "kpts_det",
        "app_loss",
        "filter",
        "renderer",
    ]

    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # CSV_DIR = os.path.join(BASE_DIR, "csv_results")

    # if not os.path.exists(CSV_DIR):
    #     os.makedirs(CSV_DIR)

    # ---------------- SURGPOSE aggregation ----------------
    if not surgpose_df.empty:
        surgpose_avg = (
            surgpose_df
            .groupby(group_cols, as_index=False)
            .agg({
                "mask_proj_error": "mean",
                "keypoint_error": "mean",
                "avg_runtime": "mean",
                "avg_runtime_per_iter": "mean",
            })
        )

        surgpose_avg = surgpose_avg.sort_values(
            by=["searcher", "online_iters"]
        )

        if evaluate_surgpose:
            surgpose_avg.to_csv(
                # os.path.join(CSV_DIR, "single_arm_surgpose_avg.csv"),
                "./single_arm_surgpose_avg.csv",
                index=False
            )
            print(f"Saved aggregated SurgPose results to single_arm_surgpose_avg.csv'")
    else:
        surgpose_avg = pd.DataFrame()

    # ---------------- SYNTHETIC aggregation ----------------
    if not synthetic_df.empty:
        synthetic_avg = (
            synthetic_df
            .groupby(group_cols, as_index=False)
            .agg({
                "rotation_error": "mean",
                "translation_error": "mean",
                "joint_theta1_error": "mean",
                "joint_theta2_error": "mean",
                "joint_theta3_error": "mean",
                "avg_runtime": "mean",
                "avg_runtime_per_iter": "mean",
            })
        )

        synthetic_avg = synthetic_avg.sort_values(
            by=["searcher", "online_iters"]
        )

        synthetic_avg.to_csv(
            # os.path.join(CSV_DIR, "single_arm_synthetic_avg.csv"),
            "./single_arm_synthetic_avg.csv",
            index=False
        )
        print(f"Saved aggregated synthetic results to ./single_arm_synthetic_avg.csv")
    else:
        synthetic_avg = pd.DataFrame()

    # =======================================================
    # DISPLAY
    # =======================================================
    # print("\n================ SURGPOSE RAW =================")
    # print(surgpose_df)

    print("\n================ SURGPOSE AVG =================")
    print(surgpose_avg)

    # print("\n================ SYNTHETIC RAW =================")
    # print(synthetic_df)

    print("\n================ SYNTHETIC AVG =================")
    print(synthetic_avg)

    return surgpose_df, synthetic_df


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


@torch.no_grad()
def evaluate_surgpose_trajectory(data_dir, cTr_seq, joint_seq, time_seq, model, robot_renderer, glctx, args):
    """Evaluate keypoint alignment and mask alignment"""
    kpts_errors = []
    mask_ious = []

    frame_start = 1
    frame_end = len([name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)) and name.isdigit()])

    pbar = tqdm.tqdm(range(frame_start, frame_end), desc=f"Evaluating {data_dir}")
    for i in pbar:
        frame_dir = os.path.join(data_dir, f"{i}")

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

        # Find the keypoint
        kpts_path = os.path.join(frame_dir, "keypoints_" + XXXX + ".npy")
        if not os.path.exists(kpts_path):
            print(f"No keypoint file found in {frame_dir}")
            continue
        ref_keypoints = np.load(kpts_path)
        ref_keypoints = torch.tensor(ref_keypoints).squeeze().float().cuda()

        # Render the mask from cTr_seq and joint_seq
        cTr = cTr_seq[i-1].clone()
        joint_angles  = joint_seq[i-1]

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

        # Project the keypoints
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

        # Compare the mask alignment and keypoint alignment
        inter = torch.logical_and(rendered_mask > 0,
                                  ref_mask > 0).sum().float()
        union = torch.logical_or(rendered_mask > 0,
                                 ref_mask > 0).sum().float()
        assert union > 0, "Union is zero, check the masks!"
        iou = (inter / union).item()
        mask_ious.append(iou)

        d1 = torch.norm(proj_keypoints - ref_keypoints, dim=-1).mean()

        d2 = torch.norm(
            proj_keypoints[[1, 0]] - ref_keypoints,
            dim=-1
        ).mean()

        # # visualize the rendererd mask and the reference mask for debugging
        # rendered_mask_vis = (rendered_mask.cpu().numpy() * 255).astype(np.uint8)
        # ref_mask_vis = (ref_mask.cpu().numpy() * 255).astype(np.uint8)
        # vis = np.concatenate([rendered_mask_vis, ref_mask_vis], axis=1)
        # cv2.imshow("Rendered Mask (Left) vs Reference Mask (Right)", vis)
        # cv2.waitKey(1)

        kpts_errors.append(min(d1.item(), d2.item()))

    mean_mask_error = 1.0 - np.mean(mask_ious) if mask_ious else None
    mean_kpts_error = np.mean(kpts_errors) if kpts_errors else None

    # runtime: average after first 10 frames
    time_seq = np.array(time_seq)
    if len(time_seq) > 10:
        avg_time = np.mean(time_seq[10:])
    else:
        avg_time = np.mean(time_seq)

    return mean_mask_error, mean_kpts_error, avg_time


def rotation_diff(mix_angle1, mix_angle2):
    """
    Compute norm of log(R_1^T R2) by kornia
    """
    # axis_angle1 = mix_angle_to_axis_angle(mix_angle1.unsqueeze(0)).squeeze()
    # axis_angle2 = mix_angle_to_axis_angle(mix_angle2.unsqueeze(0)).squeeze()
    # R1 = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle1.unsqueeze(0)).squeeze()
    # R2 = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle2.unsqueeze(0)).squeeze()
    R1 = mix_angle_to_rotmat(mix_angle1.unsqueeze(0)).squeeze()
    R2 = mix_angle_to_rotmat(mix_angle2.unsqueeze(0)).squeeze()
    # print(R1.shape, R2.shape)
    R_diff = torch.matmul(R1.transpose(0, 1), R2)
    log_R_diff = kornia.geometry.conversions.rotation_matrix_to_axis_angle(R_diff.unsqueeze(0)).squeeze()
    angle_diff = torch.norm(log_R_diff)
    return angle_diff

def translation_diff(t1, t2):
    """
    Compute L2 distance between t1 and t2
    """
    return torch.norm(t1 - t2, dim=-1)

def angle_diff(a, b):
    diff = a - b
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return torch.abs(diff)

@torch.no_grad()
def evaluate_synthetic_trajectory(data_dir, cTr_seq, joint_seq, time_seq):
    frame_start = 1
    frame_end = len([name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)) and name.isdigit()])

    ref_cTr_lst = []
    ref_joint_lst = []

    # pbar = tqdm.tqdm(range(frame_start, frame_end), desc=f"Evaluating {data_dir}")
    for i in range(frame_start, frame_end):
        frame_dir = os.path.join(data_dir, f"{i}")
        optim_ctr_path = os.path.join(frame_dir, "optimized_ctr.npy")
        optim_joint_path = os.path.join(frame_dir, "optimized_joint_angles.npy")
        if not os.path.exists(optim_ctr_path):
            raise FileNotFoundError(f"No optimized_ctr.npy found in {frame_dir}")
        else:
            optim_ctr_np = np.load(optim_ctr_path)
            optim_ctr = th.tensor(
                optim_ctr_np, requires_grad=False, dtype=th.float32
            ).cuda()
        if not os.path.exists(optim_joint_path):
            raise FileNotFoundError(f"No optimized_joint_angles.npy found in {frame_dir}")
        else:
            optim_joint_angles_np = np.load(optim_joint_path)
            optim_joint_angles = th.tensor(
                optim_joint_angles_np, requires_grad=False, dtype=th.float32
            ).cuda()

        ref_cTr_lst.append(optim_ctr)
        ref_joint_lst.append(optim_joint_angles)

    # Plot the prediction results of the 10 dimensions over time
    ref_cTr_seq = torch.stack(ref_cTr_lst, dim=0) # shape (T, 6)
    ref_joint_seq = torch.stack(ref_joint_lst, dim=0) # shape (T, 4)

    pred_rot = axis_angle_to_mix_angle(cTr_seq[:, :3])
    gt_rot = axis_angle_to_mix_angle(ref_cTr_seq[:, :3])

    pred_trans = cTr_seq[:, 3:]
    gt_trans = ref_cTr_seq[:, 3:]

    pred_joint = joint_seq[:, :3]
    gt_joint = ref_joint_seq[:, :3]

    rot_errors = []
    trans_errors = []
    joint_theta1_errors = []
    joint_theta2_errors = []
    joint_theta3_errors = []

    T = cTr_seq.shape[0]

    for i in range(T):

        # =======================
        # Branch A (normal)
        # =======================
        rA = pred_rot[i]
        tA = pred_trans[i]
        jA = joint_seq[i]

        rot_err_A = rotation_diff(rA, gt_rot[i])
        # joint_err_A = joint_diff(jA, ref_joint_seq[i])

        # chosen_rot_err = rot_err_A
        # chosen_joint = jA

        # =======================
        # Branch B (ambiguous)
        # =======================
        rB = rA.clone()
        rB[1] = rB[1] + np.pi

        jB = jA.clone()
        jB[0] = -jB[0]
        jB[1] = -jB[1]

        rot_err_B = rotation_diff(rB, gt_rot[i])

        # # ===================================================
        # # Choose branch CONSISTENTLY using rotation error
        # # ===================================================
        # if rot_err_A <= rot_err_B:
        #     chosen_rot_err = rot_err_A
        #     chosen_joint = jA
        # else:
        #     chosen_rot_err = rot_err_B
        #     chosen_joint = jB

        # # Choose branch using joint angle error (use L1 norm)
        # if torch.norm(jA[:2] - gt_joint[i][:2], p=1) <= torch.norm(jB[:2] - gt_joint[i][:2], p=1):
        #     chosen_rot_err = rot_err_A
        #     chosen_joint = jA
        # else:
        #     chosen_rot_err = rot_err_B
        #     chosen_joint = jB

        # Choose branch by checking whether or not need to +/- pi on beta to align with gt_rot
        if angle_diff(rA[1], gt_rot[i][1]) <= angle_diff(rB[1], gt_rot[i][1]):
            chosen_rot_err = rot_err_A
            chosen_joint = jA
        else:
            chosen_rot_err = rot_err_B
            chosen_joint = jB

        # Translation is identical in both branches
        chosen_trans_err = translation_diff(tA, gt_trans[i])

        # Compute joint error based on chosen branch for each joint
        joint_theta1_err = torch.abs(chosen_joint[0] - gt_joint[i][0])
        joint_theta2_err = torch.abs(chosen_joint[1] - gt_joint[i][1])
        joint_theta3_err = torch.abs(chosen_joint[2] - gt_joint[i][2])
        joint_theta1_errors.append(joint_theta1_err.item())
        joint_theta2_errors.append(joint_theta2_err.item())
        joint_theta3_errors.append(joint_theta3_err.item())

        rot_errors.append(chosen_rot_err.item())
        # joint_errors.append(chosen_joint_err.item())
        trans_errors.append(chosen_trans_err.item())

    mean_rot_error = float(np.mean(rot_errors))
    mean_trans_error = float(np.mean(trans_errors))
    # mean_joint_error = float(np.mean(joint_errors))
    mean_joint_theta1_error = float(np.mean(joint_theta1_errors))
    mean_joint_theta2_error = float(np.mean(joint_theta2_errors))
    mean_joint_theta3_error = float(np.mean(joint_theta3_errors))

    # Runtime (exclude first 10 frames)
    time_seq = np.array(time_seq)
    if len(time_seq) > 10:
        avg_time = np.mean(time_seq[10:])
    else:
        avg_time = np.mean(time_seq)

    # if mean_rot_error <= 0.:
    #     print(f"Mean rotation error is {mean_rot_error:.4f} radians, which is less than 0.5 radians.")
    # else: 
    #     print(f"Mean rotation error is {mean_rot_error:.4f} radians, which is greater than 0.5 radians. Consider checking the trajectory plot for potential global phase issues.")
    #     # Create figure with 10 subplots (5 rows x 2 columns)
    #     fig, axs = plt.subplots(5, 2, figsize=(10, 10), sharex=True)

    #     # Flatten axes array for easy iteration
    #     axs = axs.flatten()

    #     dim_name = ["Alpha", "Beta", "Gamma", "X", "Y", "Z", "Wrist Pitch", "Wrist Yaw", "Jaw 1", "Jaw 2"]

    #     pred_cTr_seq = torch.zeros_like(cTr_seq)
    #     pred_joint_seq = torch.zeros_like(joint_seq)
    #     for i in range(pred_cTr_seq.shape[0]):
    #         # =======================
    #         # Branch A (normal)
    #         # =======================
    #         rA = pred_rot[i]
    #         tA = pred_trans[i]
    #         jA = joint_seq[i]

    #         # rot_err_A = rotation_diff(rA, gt_rot[i])
    #         # joint_err_A = joint_diff(jA, ref_joint_seq[i])

    #         # =======================
    #         # Branch B (ambiguous)
    #         # =======================
    #         rB = rA.clone()
    #         rB[1] = rB[1] + np.pi

    #         jB = jA.clone()
    #         jB[0] = -jB[0]
    #         jB[1] = -jB[1]

    #         # rot_err_B = rotation_diff(rB, gt_rot[i])

    #         # ===================================================
    #         # Choose branch CONSISTENTLY using rotation error
    #         # ===================================================
    #         # if True or torch.norm(jA[:2] - gt_joint[i][:2], p=1) <= torch.norm(jB[:2] - gt_joint[i][:2], p=1):
    #         if angle_diff(rA[1], gt_rot[i][1]) <= angle_diff(rB[1], gt_rot[i][1]):
    #             chosen_rot = rA
    #             chosen_joint = jA
    #         else:
    #             chosen_rot = rB
    #             chosen_joint = jB

    #         # pred_cTr_seq[i, :3] = axis_angle_to_mix_angle(chosen_rot.unsqueeze(0)).squeeze()
    #         pred_cTr_seq[i, :3] = chosen_rot
    #         pred_joint_seq[i] = chosen_joint

    #     # Plot each dimension in its own subplot
    #     for j in range(3):
    #         ax = axs[j]

    #         # print(cTr_seq.shape, ref_cTr_seq.shape)

    #         # pred = np.unwrap(cTr_seq[:, j].cpu().numpy(), axis=0)
    #         pred = pred_cTr_seq[:, j].cpu().numpy()
    #         pred = np.unwrap(pred, axis=0)
    #         ref = gt_rot[:, j].cpu().numpy()
    #         ref  = np.unwrap(ref, axis=0)

    #         # # Align global phase
    #         # k = np.round(np.mean((ref - pred) / (2*np.pi)))
    #         # offset = 2 * np.pi * k
    #         # pred_aligned = pred + offset
    #         # Align phase for each time step individually (to handle potential phase jumps)
    #         pred_aligned = np.zeros_like(pred)
    #         for t in range(len(pred)):
    #             k = np.round((ref[t] - pred[t]) / (2*np.pi))
    #             offset = 2 * np.pi * k
    #             pred_aligned[t] = pred[t] + offset

    #         ax.plot(pred_aligned, label='Predicted', linewidth=1.5)
    #         ax.plot(ref, label='Reference', linestyle='--', linewidth=1.5)

    #         ax.set_title(dim_name[j])
    #         ax.grid(True, alpha=0.4)

    #     for j in range(3, 6):
    #         ax = axs[j]

    #         pred = cTr_seq[:, j].cpu().numpy()
    #         ref  = ref_cTr_seq[:, j].cpu().numpy()

    #         ax.plot(pred, label='Predicted', linewidth=1.5)
    #         ax.plot(ref, label='Reference', linestyle='--', linewidth=1.5)

    #         ax.set_title(dim_name[j])
    #         ax.grid(True, alpha=0.4)

    #     for j in range(6, 9):
    #         ax = axs[j]
    #         # ax.plot(joint_seq[:, j-6].cpu().numpy(), label='Predicted', linewidth=1.5)
    #         ax.plot(pred_joint_seq[:, j-6].cpu().numpy(), label='Predicted', linewidth=1.5)
    #         ax.plot(gt_joint[:, j-6].cpu().numpy(), label='Reference', linestyle='--', linewidth=1.5)
    #         ax.set_title(dim_name[j])
    #         ax.grid(True, alpha=0.4)

    #     # Add common labels
    #     fig.text(0.04, 0.5, 'cTr Values', va='center', rotation='vertical', fontsize=14)

    #     # Add a single legend below all subplots
    #     fig.legend(
    #         ['Predicted', 'Reference'],  # Labels
    #         loc='lower center',          # Position
    #         bbox_to_anchor=(0.5, 0.02), # Fine-tune position (x, y)
    #         ncol=2,                     # Number of columns in legend
    #         frameon=True,               # Add a frame
    #         fontsize=12                 # Adjust font size
    #     )

    #     # Adjust layout
    #     plt.tight_layout(rect=[0.03, 0.03, 1, 0.98])  # Make space for suptitle and labels

    #     # Save the figure
    #     if not os.path.exists("./pose_reconstruction"):
    #         os.makedirs("./pose_reconstruction")

    #     plot_save_path = f"./pose_reconstruction/{data_dir.replace('/', '_')}_tracking_results_{mean_rot_error:.4f}.png"
    #     fig.savefig(plot_save_path)

    return mean_rot_error, mean_trans_error, \
        mean_joint_theta1_error, mean_joint_theta2_error, mean_joint_theta3_error, \
        avg_time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--read_from_csv",  action="store_true")
    parser.add_argument("--evaluate_surgpose", action="store_true")
    args_eval = parser.parse_args()

    print(args_eval.evaluate_surgpose)

    if args_eval.read_from_csv:
        # Load raw CSVs
        surgpose_df = pd.read_csv("./single_arm_surgpose_raw.csv")
        synthetic_df = pd.read_csv("./single_arm_synthetic_raw.csv")

        # -------------------- SINGLE-ARM SURGPOSE --------------------
        if not surgpose_df.empty:
            # Normal: 000000 → 000007
            normal_mask = surgpose_df['data_label'].isin(
                [f"surgpose_{i:06d}_PSM1" for i in range(0, 8)] + [f"surgpose_{i:06d}_PSM3" for i in range(0, 8)]
            )
            # Fast: 000030 → 000033
            fast_mask = surgpose_df['data_label'].isin(
                [f"surgpose_{i:06d}_PSM1" for i in range(30, 34)] + [f"surgpose_{i:06d}_PSM3" for i in range(30, 34)]
            )

            surgpose_normal_df = surgpose_df[normal_mask].copy()
            surgpose_fast_df   = surgpose_df[fast_mask].copy()

            group_cols = [
                "searcher",
                "online_iters",
                "joint_label",
                "pts_loss",
                "kpts_det",
                "app_loss",
                "filter",
                "renderer",
            ]

            surgpose_normal_avg = surgpose_normal_df.groupby(group_cols, as_index=False).agg({
                "mask_proj_error": "mean",
                "keypoint_error": "mean",
                "avg_runtime": "mean",
                "avg_runtime_per_iter": "mean",
            })

            surgpose_fast_avg = surgpose_fast_df.groupby(group_cols, as_index=False).agg({
                "mask_proj_error": "mean",
                "keypoint_error": "mean",
                "avg_runtime": "mean",
                "avg_runtime_per_iter": "mean",
            })

            surgpose_normal_avg.to_csv("./single_arm_surgpose_normal_avg.csv", index=False)
            surgpose_fast_avg.to_csv("./single_arm_surgpose_fast_avg.csv", index=False)

            print("\n====== SINGLE-ARM SURGPOSE NORMAL AVG ======")
            print(surgpose_normal_avg)
            print("\n====== SINGLE-ARM SURGPOSE FAST AVG ======")
            print(surgpose_fast_avg)

        # -------------------- SINGLE-ARM SYNTHETIC --------------------
        if not synthetic_df.empty:
            group_cols = [
                "searcher",
                "online_iters",
                "joint_label",
                "pts_loss",
                "kpts_det",
                "app_loss",
                "filter",
                "renderer",
            ]

            synthetic_avg = synthetic_df.groupby(group_cols, as_index=False).agg({
                "rotation_error": "mean",
                "translation_error": "mean",
                "joint_theta1_error": "mean",
                "joint_theta2_error": "mean",
                "joint_theta3_error": "mean",
                "avg_runtime": "mean",
                "avg_runtime_per_iter": "mean",
            })

            synthetic_avg.to_csv("./single_arm_synthetic_avg.csv", index=False)
            print("\n====== SINGLE-ARM SYNTHETIC AVG ======")
            print(synthetic_avg)

    else:
        # Original code: re-evaluate .pth files
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

        surgpose_df, synthetic_df = collect_results(
            model,
            robot_renderer,
            glctx,
            args,
            evaluate_surgpose=args_eval.evaluate_surgpose,
        )



