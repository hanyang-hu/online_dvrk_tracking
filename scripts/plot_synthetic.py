import argparse
import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ------------------ Path bootstrap ------------------
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

LOCAL_MODULE_DIRS = [
    REPO_ROOT,
]

for p in LOCAL_MODULE_DIRS:
    if p not in sys.path:
        sys.path.insert(0, p)


from diffcali.utils.angle_transform_utils import (
    mix_angle_to_axis_angle,
    axis_angle_to_mix_angle,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 9-dim pose reconstruction results on synthetic data")

    parser.add_argument("--results_dir", type=str, default="./pose_results/", help="Directory containing the results files")
    parser.add_argument("--data_dir", type=str, default="./data/synthetic/", help="Directory containing the data files")
    parser.add_argument("--bag_idx", type=str, default="000000/PSM3", help="Name of the bag to process (e.g., bag1, bag2, etc.)")
    parser.add_argument("--output_dir", type=str, default="./plots/synthetic/", help="Directory to save the generated plots")

    args = parser.parse_args()

    # Load pose results (example: synthetic_000015_PSM3.CMA-ES.5.wo_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.nvdiffrast.pth)
    CMA_ES_w_joint_file = f"synthetic_{args.bag_idx.replace('/', '_')}.CMA-ES.3.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.nvdiffrast.pth"
    CMA_ES_wo_joint_file = f"synthetic_{args.bag_idx.replace('/', '_')}.CMA-ES.3.wo_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.nvdiffrast.pth"
    GD_w_joint_file = f"synthetic_{args.bag_idx.replace('/', '_')}.Gradient.10.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.nvdiffrast.pth"
    GD_wo_joint_file = f"synthetic_{args.bag_idx.replace('/', '_')}.Gradient.10.wo_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.nvdiffrast.pth"
    CMA_ES_w_joint = torch.load(os.path.join(args.results_dir, CMA_ES_w_joint_file))
    CMA_ES_wo_joint = torch.load(os.path.join(args.results_dir, CMA_ES_wo_joint_file))
    GD_w_joint = torch.load(os.path.join(args.results_dir, GD_w_joint_file))
    GD_wo_joint = torch.load(os.path.join(args.results_dir, GD_wo_joint_file))

    # Load ground truth pose
    data_dir = os.path.join(args.data_dir, args.bag_idx)
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
            optim_ctr = torch.tensor(
                optim_ctr_np, requires_grad=False, dtype=torch.float32
            ).cuda()
        if not os.path.exists(optim_joint_path):
            raise FileNotFoundError(f"No optimized_joint_angles.npy found in {frame_dir}")
        else:
            optim_joint_angles_np = np.load(optim_joint_path)
            optim_joint_angles = torch.tensor(
                optim_joint_angles_np, requires_grad=False, dtype=torch.float32
            ).cuda()

        ref_cTr_lst.append(optim_ctr)
        ref_joint_lst.append(optim_joint_angles)

    ref_cTr_seq = torch.stack(ref_cTr_lst, dim=0) # shape (T, 6)
    ref_joint_seq = torch.stack(ref_joint_lst, dim=0) # shape (T, 4)

    print("Ground truth cTr shape:", ref_cTr_seq.shape)
    print("Ground truth joint angles shape:", ref_joint_seq.shape)

    print("CMA-ES with joint angles - cTr shape:", CMA_ES_w_joint["cTr"].shape, "joint angles shape:", CMA_ES_w_joint["joint_angles"].shape)
    print("CMA-ES without joint angles - cTr shape:", CMA_ES_wo_joint["cTr"].shape, "joint angles shape:", CMA_ES_wo_joint["joint_angles"].shape)
    print("Gradient with joint angles - cTr shape:", GD_w_joint["cTr"].shape, "joint angles shape:", GD_w_joint["joint_angles"].shape)
    print("Gradient without joint angles - cTr shape:", GD_wo_joint["cTr"].shape, "joint angles shape:", GD_wo_joint["joint_angles"].shape)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert the rotation from axis-angle to Euler angles for better visualization (shape (T, 3))
    ref_cTr_seq[:, :3] = axis_angle_to_mix_angle(ref_cTr_seq[:, :3])
    CMA_ES_w_joint["cTr"][:, :3] = axis_angle_to_mix_angle(CMA_ES_w_joint["cTr"][:, :3])
    CMA_ES_wo_joint["cTr"][:, :3] = axis_angle_to_mix_angle(CMA_ES_wo_joint["cTr"][:, :3])
    GD_w_joint["cTr"][:, :3] = axis_angle_to_mix_angle(GD_w_joint["cTr"][:, :3])
    GD_wo_joint["cTr"][:, :3] = axis_angle_to_mix_angle(GD_wo_joint["cTr"][:, :3])

    # Convert 10-DOF pose to 9-DOF by dropping the last joint angle (gripper angle)
    ref_joint_seq = ref_joint_seq[:, :3]
    # ref_joint_seq[:, -1] *= 2
    CMA_ES_w_joint["joint_angles"] = CMA_ES_w_joint["joint_angles"][:, :3]
    # CMA_ES_w_joint["joint_angles"][:, -1] *= 2
    CMA_ES_wo_joint["joint_angles"] = CMA_ES_wo_joint["joint_angles"][:, :3]
    # CMA_ES_wo_joint["joint_angles"][:, -1] *= 2
    GD_w_joint["joint_angles"] = GD_w_joint["joint_angles"][:, :3]
    # GD_w_joint["joint_angles"][:, -1] *= 2
    GD_wo_joint["joint_angles"] = GD_wo_joint["joint_angles"][:, :3]
    # GD_wo_joint["joint_angles"][:, -1] *= 2

    dim_name = ["$\\alpha$ (rad)", "$\\beta$ (rad)", "$\\gamma$ (rad)", "x (m)", "y (m)", "z (m)", "Wrist Pitch (rad)", "Wrist Yaw (rad)", "Jaw (rad)"]

        # ------------------ Display Control ------------------
    SHOW = {
        "gt": True,
        "cma_w": True,
        "cma_wo": True,
        "gd_w": False,
        "gd_wo": True,
    }

        # ------------------ Prepare Data ------------------

    ref_cTr_seq = ref_cTr_seq.detach().cpu()
    ref_joint_seq = ref_joint_seq.detach().cpu()

    CMA_ES_w_joint_cTr = CMA_ES_w_joint["cTr"].detach().cpu()
    CMA_ES_wo_joint_cTr = CMA_ES_wo_joint["cTr"].detach().cpu()
    GD_w_joint_cTr = GD_w_joint["cTr"].detach().cpu()
    GD_wo_joint_cTr = GD_wo_joint["cTr"].detach().cpu()

    CMA_ES_w_joint_joint = CMA_ES_w_joint["joint_angles"].detach().cpu()
    CMA_ES_wo_joint_joint = CMA_ES_wo_joint["joint_angles"].detach().cpu()
    GD_w_joint_joint = GD_w_joint["joint_angles"].detach().cpu()
    GD_wo_joint_joint = GD_wo_joint["joint_angles"].detach().cpu()

    T = ref_cTr_seq.shape[0]
    time_axis = np.arange(T)

    # ------------------ Publication Style ------------------
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "lines.linewidth": 1.6,
    })

    # Wider figure (ideal for IROS two-column)
    fig, axes = plt.subplots(3, 3, figsize=(20, 6), sharex=True)
    axes = axes.reshape(3, 3)

    def unwrap_and_align(pred_tensor, ref_tensor):
        pred = np.unwrap(pred_tensor.numpy(), axis=0)
        ref = np.unwrap(ref_tensor.numpy(), axis=0)

        k = np.round(np.mean((ref - pred) / (2 * np.pi)))
        pred += 2 * np.pi * k
        return pred, ref

    def plot_dimension(ax, ref, cma_w, cma_wo, gd_w, gd_wo, name, is_angle=False):
        curves = {}

        if is_angle:
            if SHOW["cma_w"]:
                curves["Ours (w/ joint, 3 iter/frame)"], ref_proc = unwrap_and_align(cma_w, ref)
            if SHOW["cma_wo"]:
                curves["Ours (w/o joint, 3 iter/frame)"], _ = unwrap_and_align(cma_wo, ref)
            if SHOW["gd_w"]:
                curves["GD (w/ joint, 10 iter/frame)"], _ = unwrap_and_align(gd_w, ref)
            if SHOW["gd_wo"]:
                curves["GD (w/o joint, 10 iter/frame)"], _ = unwrap_and_align(gd_wo, ref)
            ref_np = np.unwrap(ref.numpy(), axis=0)
        else:
            ref_np = ref.numpy()
            if SHOW["cma_w"]:
                curves["Ours (w/ joint, 3 iter/frame)"] = cma_w.numpy()
            if SHOW["cma_wo"]:
                curves["Ours (w/o joint, 3 iter/frame)"] = cma_wo.numpy()
            if SHOW["gd_w"]:
                curves["GD (w/ joint, 10 iter/frame)"] = gd_w.numpy()
            if SHOW["gd_wo"]:
                curves["GD (w/o joint, 10 iter/frame)"] = gd_wo.numpy()

        if SHOW["gt"]:
            ax.plot(
                time_axis,
                ref_np,
                color="black",
                linewidth=2.8,
                label="Ground Truth",
                zorder=0
            )

        for label, values in curves.items():
            if label == "CMA-ES w/joint":
                ax.plot(
                    time_axis,
                    values,
                    label=label,
                    color="red",
                    linewidth=1.8,
                    zorder=5
                )
            else:
                ax.plot(
                    time_axis,
                    values,
                    label=label,
                    zorder=5
                )

        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    # ------------------ Column 1: Rotation ------------------
    for i in range(3):
        plot_dimension(
            axes[i, 0],
            ref_cTr_seq[:, i],
            CMA_ES_w_joint_cTr[:, i],
            CMA_ES_wo_joint_cTr[:, i],
            GD_w_joint_cTr[:, i],
            GD_wo_joint_cTr[:, i],
            dim_name[i],
            is_angle=True
        )

    # ------------------ Column 2: Translation ------------------
    for i in range(3):
        plot_dimension(
            axes[i, 1],
            ref_cTr_seq[:, i + 3],
            CMA_ES_w_joint_cTr[:, i + 3],
            CMA_ES_wo_joint_cTr[:, i + 3],
            GD_w_joint_cTr[:, i + 3],
            GD_wo_joint_cTr[:, i + 3],
            dim_name[i + 3],
            is_angle=False
        )

    # ------------------ Column 3: Joint Angles ------------------
    for i in range(3):
        plot_dimension(
            axes[i, 2],
            ref_joint_seq[:, i],
            CMA_ES_w_joint_joint[:, i],
            CMA_ES_wo_joint_joint[:, i],
            GD_w_joint_joint[:, i],
            GD_wo_joint_joint[:, i],
            dim_name[i + 6],
            is_angle=True
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Frame")

    # --------- Center-bottom Legend (Only Active Curves) ----------
    handles, labels = [], []
    for ax in axes.flatten():
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
        if len(labels) > 0:
            break

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    save_path = os.path.join(args.output_dir, f"{args.bag_idx.replace('/', '_')}_9dof_plot.png")
    plt.savefig(save_path, format="png", bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {save_path}")


"""
Plot for 0-15
python scripts/plot_synthetic.py --bag_idx 000000/PSM1
python scripts/plot_synthetic.py --bag_idx 000000/PSM3
python scripts/plot_synthetic.py --bag_idx 000001/PSM1
python scripts/plot_synthetic.py --bag_idx 000001/PSM3
python scripts/plot_synthetic.py --bag_idx 000002/PSM1
python scripts/plot_synthetic.py --bag_idx 000002/PSM3
python scripts/plot_synthetic.py --bag_idx 000003/PSM1
python scripts/plot_synthetic.py --bag_idx 000003/PSM3
python scripts/plot_synthetic.py --bag_idx 000004/PSM1
python scripts/plot_synthetic.py --bag_idx 000004/PSM3  
python scripts/plot_synthetic.py --bag_idx 000005/PSM1
python scripts/plot_synthetic.py --bag_idx 000005/PSM3
python scripts/plot_synthetic.py --bag_idx 000006/PSM1
python scripts/plot_synthetic.py --bag_idx 000006/PSM3
python scripts/plot_synthetic.py --bag_idx 000007/PSM1
python scripts/plot_synthetic.py --bag_idx 000007/PSM3
python scripts/plot_synthetic.py --bag_idx 000008/PSM1
python scripts/plot_synthetic.py --bag_idx 000008/PSM3
python scripts/plot_synthetic.py --bag_idx 000009/PSM1
python scripts/plot_synthetic.py --bag_idx 000009/PSM3
python scripts/plot_synthetic.py --bag_idx 000010/PSM1
python scripts/plot_synthetic.py --bag_idx 000010/PSM3
python scripts/plot_synthetic.py --bag_idx 000011/PSM1  
python scripts/plot_synthetic.py --bag_idx 000011/PSM3
python scripts/plot_synthetic.py --bag_idx 000012/PSM1
python scripts/plot_synthetic.py --bag_idx 000012/PSM3
python scripts/plot_synthetic.py --bag_idx 000013/PSM1
python scripts/plot_synthetic.py --bag_idx 000013/PSM3
python scripts/plot_synthetic.py --bag_idx 000014/PSM1
python scripts/plot_synthetic.py --bag_idx 000014/PSM3
python scripts/plot_synthetic.py --bag_idx 000015/PSM1
python scripts/plot_synthetic.py --bag_idx 000015/PSM3
"""