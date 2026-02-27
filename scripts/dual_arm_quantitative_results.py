import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
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

POSE_DIR = "./pose_results"


# ==========================================================
# Filename parser
# ==========================================================

def parse_dual_arm_filename(filename):

    name = os.path.basename(filename).replace(".pth", "")

    if not name.startswith("BI_MANUAL_"):
        return None

    name = name.replace("BI_MANUAL_", "")
    parts = name.split(".")

    if len(parts) != 10:
        return None

    return dict(
        data_label=parts[0],
        searcher=parts[1],
        online_iters=int(parts[2]),
        joint_label=parts[3],
        pts_loss=parts[4],
        kpts_det=parts[5],
        app_loss=parts[6],
        filter=parts[7],
        loss_option=parts[8],
        separation=parts[9],
    )


# ==========================================================
# Dual-arm evaluators
# ==========================================================

@torch.no_grad()
def evaluate_dual_surgpose(
    data_root,
    cTr_seq,
    joint_seq,
    time_seq,
    model,
    robot_renderer,
    glctx,
    args,
    # evaluate_surgpose
):

    left_dir  = os.path.join(data_root, "PSM3")
    right_dir = os.path.join(data_root, "PSM1")

    pred_cTr_L = cTr_seq[:, 0, :].cuda()
    pred_cTr_R = cTr_seq[:, 1, :].cuda()

    pred_joint_L = joint_seq[:, 0, :].cuda()
    pred_joint_R = joint_seq[:, 1, :].cuda()

    # print(left_dir, pred_cTr_L.shape, pred_joint_L.shape)
    # print(right_dir, pred_cTr_R.shape, pred_joint_R.shape)

    mask_L, kpt_L, _ = evaluate_surgpose_trajectory(
        left_dir, pred_cTr_L, pred_joint_L,
        time_seq, model, robot_renderer, glctx, args,
        # evaluate_surgpose=evaluate_surgpose,
    )

    mask_R, kpt_R, _ = evaluate_surgpose_trajectory(
        right_dir, pred_cTr_R, pred_joint_R,
        time_seq, model, robot_renderer, glctx, args,
        # evaluate_surgpose=evaluate_surgpose,
    )

    mask_err = (mask_L + mask_R) / 2.0
    kpt_err  = (kpt_L + kpt_R) / 2.0

    time_seq = np.array(time_seq)
    avg_time = np.mean(time_seq[10:]) if len(time_seq) > 10 else np.mean(time_seq)

    return mask_err, kpt_err, avg_time


@torch.no_grad()
def evaluate_dual_synthetic(data_root, cTr_seq, joint_seq, time_seq):

    left_dir  = os.path.join(data_root, "PSM3")
    right_dir = os.path.join(data_root, "PSM1")

    pred_cTr_L = cTr_seq[:, 0, :].cuda()
    pred_cTr_R = cTr_seq[:, 1, :].cuda()

    pred_joint_L = joint_seq[:, 0, :].cuda()
    pred_joint_R = joint_seq[:, 1, :].cuda()

    rot_L, trans_L, joint_theta1_L, joint_theta2_L, joint_theta3_L, _ = evaluate_synthetic_trajectory(
        left_dir, pred_cTr_L, pred_joint_L, time_seq
    )

    rot_R, trans_R, joint_theta1_R, joint_theta2_R, joint_theta3_R, _ = evaluate_synthetic_trajectory(
        right_dir, pred_cTr_R, pred_joint_R, time_seq
    )

    rot_err   = (rot_L   + rot_R)   / 2.0
    trans_err = (trans_L + trans_R) / 2.0
    joint_theta1_err = (joint_theta1_L + joint_theta1_R) / 2.0
    joint_theta2_err = (joint_theta2_L + joint_theta2_R) / 2.0
    joint_theta3_err = (joint_theta3_L + joint_theta3_R) / 2.0

    time_seq = np.array(time_seq)
    avg_time = np.mean(time_seq[10:]) if len(time_seq) > 10 else np.mean(time_seq)

    return rot_err, trans_err, joint_theta1_err, joint_theta2_err, joint_theta3_err, avg_time


# ==========================================================
# Main collector
# ==========================================================

def collect_results(model, robot_renderer, glctx, args, evaluate_surgpose):

    surgpose_rows = []
    synthetic_rows = []

    pbar = tqdm.tqdm(os.listdir(POSE_DIR), desc=f"Collecting results from {POSE_DIR}")
    for file in os.listdir(POSE_DIR):

        if not file.endswith(".pth"):
            continue

        meta = parse_dual_arm_filename(file)
        if meta is None:
            continue

        full_path = os.path.join(POSE_DIR, file)
        data = torch.load(full_path, map_location="cpu")

        parts = meta["data_label"].split("_")
        base, seq = parts[0], parts[1]
        data_root = f"./data/{base}/{seq}"

        cTr_seq   = data["cTr"]
        joint_seq = data["joint_angles"]
        time_seq  = data["time"]

        # ================= SURGPOSE =================
        if base == "surgpose" and evaluate_surgpose and seq in ["000030", "000031", "000032", "000033"]:

            mask_err, kpt_err, avg_time = evaluate_dual_surgpose(
                data_root,
                cTr_seq,
                joint_seq,
                time_seq,
                model,
                robot_renderer,
                glctx,
                args,
                # evaluate_surgpose=evaluate_surgpose,
            )

            surgpose_rows.append({
                **meta,
                "mask_proj_error": mask_err,
                "keypoint_error": kpt_err,
                "avg_runtime": avg_time,
                "avg_runtime_per_iter":
                    avg_time / meta["online_iters"]
                    if meta["online_iters"] > 0 else None,
            })

        # ================= SYNTHETIC =================
        elif False and base == "synthetic":

            rot_err, trans_err, joint_theta1_err, joint_theta2_err, joint_theta3_err, avg_time = evaluate_dual_synthetic(
                data_root,
                cTr_seq,
                joint_seq,
                time_seq
            )

            synthetic_rows.append({
                **meta,
                "rotation_error": rot_err,
                "translation_error": trans_err,
                "joint_theta1_error": joint_theta1_err,
                "joint_theta2_error": joint_theta2_err,
                "joint_theta3_error": joint_theta3_err,
                "avg_runtime": avg_time,
                "avg_runtime_per_iter":
                    avg_time / meta["online_iters"]
                    if meta["online_iters"] > 0 else None,
            })

        pbar.update(1)

    surgpose_df = pd.DataFrame(surgpose_rows)
    synthetic_df = pd.DataFrame(synthetic_rows)

    # ================= RAW =================
    if evaluate_surgpose:
        surgpose_df.to_csv("dual_arm_surgpose_raw.csv", index=False)
    synthetic_df.to_csv("dual_arm_synthetic_raw.csv", index=False)

    # ================= AGGREGATION =================
    group_cols = [
        "searcher",
        "online_iters",
        "joint_label",
        "pts_loss",
        "kpts_det",
        "app_loss",
        "filter",
        "loss_option",
        "separation",
    ]

    if not surgpose_df.empty:
        # surgpose_avg = surgpose_df.groupby(group_cols, as_index=False).mean()
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
        if evaluate_surgpose:
            surgpose_avg.to_csv("dual_arm_surgpose_avg.csv", index=False)
            print("\n====== DUAL ARM SURGPOSE AVG ======")
            print(surgpose_avg)

    if not synthetic_df.empty:
        synthetic_avg = (
            synthetic_df
            .groupby(group_cols, as_index=False)
            .agg({
                "rotation_error": "mean",
                "translation_error": "mean",
                # "joint_error": "mean",
                "joint_theta1_error": "mean",
                "joint_theta2_error": "mean",
                "joint_theta3_error": "mean",
                "avg_runtime": "mean",
                "avg_runtime_per_iter": "mean",
            })
        )
        synthetic_avg.to_csv("dual_arm_synthetic_avg.csv", index=False)
        print("\n====== DUAL ARM SYNTHETIC AVG ======")
        print(synthetic_avg)

    return surgpose_df, synthetic_df


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_from_csv",  action="store_true")
    parser.add_argument("--evaluate_surgpose", action="store_true")
    args_eval = parser.parse_args()

    if args_eval.read_from_csv:
        surgpose_df = pd.read_csv("dual_arm_surgpose_raw.csv")
        # synthetic_df = pd.read_csv("dual_arm_synthetic_raw.csv")
        
        # Read raw CSVs
        surgpose_df = pd.read_csv("dual_arm_surgpose_raw.csv")
        # synthetic_df = pd.read_csv("dual_arm_synthetic_raw.csv")

        # ----------------------
        # SurgPose aggregation (by data_label)
        # ----------------------
        if not surgpose_df.empty:
            normal_mask = surgpose_df['data_label'].isin(
                [f"surgpose_{i:06d}" for i in range(0, 8)]
            )
            fast_mask = surgpose_df['data_label'].isin(
                [f"surgpose_{i:06d}" for i in range(30, 34)]
            )

            normal_df = surgpose_df[normal_mask].copy()
            fast_df   = surgpose_df[fast_mask].copy()

            group_cols = [
                "searcher",
                "online_iters",
                "joint_label",
                "pts_loss",
                "kpts_det",
                "app_loss",
                "filter",
                "loss_option",
                "separation",
            ]

            surgpose_normal_avg = normal_df.groupby(group_cols, as_index=False).agg({
                "mask_proj_error": "mean",
                "keypoint_error": "mean",
                "avg_runtime": "mean",
                "avg_runtime_per_iter": "mean",
            })
            surgpose_fast_avg = fast_df.groupby(group_cols, as_index=False).agg({
                "mask_proj_error": "mean",
                "keypoint_error": "mean",
                "avg_runtime": "mean",
                "avg_runtime_per_iter": "mean",
            })

            surgpose_normal_avg.to_csv("dual_arm_surgpose_normal_avg.csv", index=False)
            surgpose_fast_avg.to_csv("dual_arm_surgpose_fast_avg.csv", index=False)

            print("\n====== DUAL ARM SURGPOSE NORMAL AVG ======")
            print(surgpose_normal_avg)
            print("\n====== DUAL ARM SURGPOSE FAST AVG ======")
            print(surgpose_fast_avg)

        # # ----------------------
        # # Synthetic aggregation
        # # ----------------------
        # if not synthetic_df.empty:
        #     group_cols = [
        #         "searcher",
        #         "online_iters",
        #         "joint_label",
        #         "pts_loss",
        #         "kpts_det",
        #         "app_loss",
        #         "filter",
        #         "loss_option",
        #         "separation",
        #     ]
        #     synthetic_avg = synthetic_df.groupby(group_cols, as_index=False).agg({
        #         "rotation_error": "mean",
        #         "translation_error": "mean",
        #         "joint_theta1_error": "mean",
        #         "joint_theta2_error": "mean",
        #         "joint_theta3_error": "mean",
        #         "avg_runtime": "mean",
        #         "avg_runtime_per_iter": "mean",
        #     })
        #     synthetic_avg.to_csv("dual_arm_synthetic_avg.csv", index=False)
        #     print("\n====== DUAL ARM SYNTHETIC AVG ======")
        #     print(synthetic_avg)

    else:
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

        glctx = dr.RasterizeCudaContext()
        resolution = (args.height, args.width)
        print(f"Initialized rasterizer with resolution {resolution}")

        collect_results(model, robot_renderer, glctx, args, evaluate_surgpose=args_eval.evaluate_surgpose)
