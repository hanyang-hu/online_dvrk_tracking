import argparse
import torch
import os
import sys
import cv2
import glob

# ------------------ Path bootstrap ------------------
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

LOCAL_MODULE_DIRS = [
    REPO_ROOT,
]

for p in LOCAL_MODULE_DIRS:
    if p not in sys.path:
        sys.path.insert(0, p)


from diffcali.models.CtRNet import CtRNet
from diffcali.utils.ui_utils import *
from diffcali.utils.skeleton_visualizer import SkeletonVisualizer


def parseCtRNetArgs():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.use_gpu = True
    args.trained_on_multi_gpus = False

    # args.height = 480
    # args.width = 640
    # args.fx, args.fy, args.px, args.py = 1025.88223, 1025.88223, 167.919017, 234.152707

    # Setting for SurgPose data
    args.height = 986 // 2
    args.width = 1400 // 2
    args.fx, args.fy, args.px, args.py = 1811.910046453570 / 2, 1809.640734154330 / 2, 588.5594517681759 / 2, 477.3975900383616 / 2

    args.scale = 1.0

    # scale the camera parameters
    args.width = int(args.width * args.scale)
    args.height = int(args.height * args.scale)
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale

    return args


def plot_bi_manual_skeleton(cTr_traj, joint_angle_traj, img_list, mask_lst, skeleton_visualizer_args, output_dir, color=(0, 255, 0)):
    os.makedirs(output_dir, exist_ok=True)

    # Plot left and right arms separately
    cTr_left, cTr_right = cTr_traj[:, 0, :], cTr_traj[:, 1, :]
    joint_angles_left, joint_angles_right = joint_angle_traj[:, 0, :], joint_angle_traj[:, 1, :]

    # skeleton_visualizer_args are (model, ctrnet_args, None, intr, p_local1, p_local2)
    skeleton_visualizer_left = SkeletonVisualizer(*skeleton_visualizer_args)
    skeleton_visualizer_right = SkeletonVisualizer(*skeleton_visualizer_args)
    for i in range(len(img_list)):
        img = img_list[i]
        mask_psm3 = mask_lst["PSM3"][i]
        mask_psm1 = mask_lst["PSM1"][i]

        # Binary union mask (0 or 1)
        mask = (np.maximum(mask_psm3, mask_psm1) > 0)

        alpha = 0.3
        blended = img.copy()

        # Blend ONLY where mask is True
        blended[mask] = (
            img[mask] * (1 - alpha) +
            np.array(color, dtype=np.uint8) * alpha
        ).astype(np.uint8)

        # Plot left arm skeleton
        blended = skeleton_visualizer_left.plot_skeleton_overlay(
            blended,
            cTr_left[i],
            joint_angles_left[i],
        )

        # Plot right arm skeleton
        blended = skeleton_visualizer_right.plot_skeleton_overlay(
            blended,
            cTr_right[i],
            joint_angles_right[i],
        )

        output_path = os.path.join(output_dir, f"{output_dir.split('/')[-1]}_{i+1:04d}.png")
        cv2.imwrite(output_path, blended)
        # print(f"Saved overlay image to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting script for dVRK tracking results")
    parser.add_argument("--results_dir", type=str, default="./pose_results/", help="Directory containing the results files")
    parser.add_argument("--data_dir", type=str, default="./data/surgpose/", help="Directory containing the data files")
    parser.add_argument("--video_dir", type=str, default="./data/online_videos/", help="Directory to save the generated videos")
    parser.add_argument("--bag_idx", type=str, default="000000", help="Name of the bag to process (e.g., bag1, bag2, etc.)")
    parser.add_argument("--output_dir", type=str, default="./plots/bi_manual/", help="Directory to save the generated plots")
    args = parser.parse_args()

    # Load pose results from the specified directory
    traj_file_wo_joint = f"BI_MANUAL_surgpose_{args.bag_idx}.CMA-ES.3.wo_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.joint.hardsep.pth"
    traj_wo_joint = torch.load(os.path.join(args.results_dir, traj_file_wo_joint))
    traj_file_w_joint = f"BI_MANUAL_surgpose_{args.bag_idx}.CMA-ES.3.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.joint.hardsep.pth"
    traj_w_joint = torch.load(os.path.join(args.results_dir, traj_file_w_joint))
    traj_file_GD_wo_joint = f"BI_MANUAL_surgpose_{args.bag_idx}.Gradient.10.wo_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.joint.hardsep.pth"
    traj_GD_wo_joint = torch.load(os.path.join(args.results_dir, traj_file_GD_wo_joint))
    traj_file_GF_w_joint = f"BI_MANUAL_surgpose_{args.bag_idx}.Gradient.10.w_joint_angles.w_pts_loss.w_tipnet.w_app_loss.Kalman.joint.hardsep.pth"
    traj_GD_w_joint = torch.load(os.path.join(args.results_dir, traj_file_GF_w_joint))

    # Load masks for the specified bag
    mask_lst = {"PSM3": [], "PSM1": []}
    for arm in ["PSM3", "PSM1"]:
        frame_start = 1
        mask_dir = os.path.join(args.data_dir, args.bag_idx, arm)
        frame_end = len([name for name in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, name)) and name.isdigit()])
        for frame_idx in range(frame_start, frame_end):
            frame_dir = os.path.join(mask_dir, f"{frame_idx}")

            # Find the mask
            masks = glob.glob(os.path.join(frame_dir, "*.png"))
            if len(masks) == 0:
                print(f"No mask found in {frame_dir}")
                continue
            if len(masks) > 1:
                print(f"Multiple masks found in {frame_dir}")
                continue

            mask_path = masks[0]
            # frame = cv2.imread(frame_path)
            XXXX = mask_path.split("/")[-1].split(".")[0][1:]

            # Read ref_img_file of name 0XXXX.jpg
            ref_mask_path = os.path.join(frame_dir, "0" + XXXX + ".png")
            ref_img = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)

            mask_lst[arm].append(ref_img)

    # Load data for the specified bag
    video_path = os.path.join(args.video_dir, args.bag_idx, "video.mp4")
    print(f"Loading video from {video_path}")
    cap = cv2.VideoCapture(video_path)
    img_list = []
    img_size = mask_lst["PSM3"][0].shape
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_size[1], img_size[0]))
        img_list.append(frame)

    # Discard the first and last frames
    img_list = img_list[1:-1]

    # # Save the overlay of the masks on the video frames as ./args.output_dir/1.png, ... ./args.output_dir/N.png
    # # Both masks use the same color (e.g., red) for simplicity, and the overlay is blended with the original image using an alpha value of 0.5
    # os.makedirs(args.output_dir, exist_ok=True)
    # for idx, (img, mask_psm3, mask_psm1) in enumerate(zip(img_list, mask_lst["PSM3"], mask_lst["PSM1"])):
    #     overlay = img.copy()
    #     overlay[mask_psm3 > 0] = (0, 0, 255) 
    #     overlay[mask_psm1 > 0] = (0, 0, 255) 
    #     alpha = 0.5
    #     blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    #     output_path = os.path.join(args.output_dir, f"{idx+1:04d}.png")
    #     cv2.imwrite(output_path, blended)
    #     # print(f"Saved overlay image to {output_path}")

    # Define skeleton visualizer
    ctrnet_args = parseCtRNetArgs()
    ctrnet_args.use_nvdiffrast = True

    # Load rendering model
    model = CtRNet(ctrnet_args)

    mesh_files = [
        "urdfs/dVRK/meshes/shaft_multi_cylinder.ply",
        "urdfs/dVRK/meshes/logo_low_res_1.ply",
        "urdfs/dVRK/meshes/jawright_lowres.ply",
        "urdfs/dVRK/meshes/jawleft_lowres.ply",
    ]
    robot_renderer = model.setup_robot_renderer(mesh_files, downscale_factor=1)
    robot_renderer.set_mesh_visibility([True, True, True, True])

    # Specify camera intrinsics and keypoints
    intr = torch.tensor(
        [
            [ctrnet_args.fx, 0, ctrnet_args.px], 
            [0, ctrnet_args.fy, ctrnet_args.py], 
            [0, 0, 1]
        ],
        device="cuda",
        dtype=torch.float32,
    )

    tip_length = 0.0096 # Set the tip length (distance from the last joint to the tip) to 9.6mm, which is more accurate according to the keypoint prompts
    p_local1 = (
        torch.tensor([0.0, 0.0004, tip_length]) 
        .to(torch.float32)
        .to(model.device)
    )
    p_local2 = (
        torch.tensor([0.0, -0.0004, tip_length])
        .to(torch.float32)
        .to(model.device)
    )

    skeleton_visualizer_args = (model, ctrnet_args, None, intr, p_local1, p_local2)

    # Plot skeleton overlay for each trajectory ans save to different folders
    # # CMA-ES without joint angles
    # plot_bi_manual_skeleton(traj_wo_joint["cTr"], traj_wo_joint["joint_angles"], img_list, mask_lst, skeleton_visualizer_args, output_dir=os.path.join(args.output_dir, "CMA-ES_wo_joint_angles"))
    # print("Plotted CMA-ES without joint angles")

    # CMA-ES with joint angles
    plot_bi_manual_skeleton(traj_w_joint["cTr"], traj_w_joint["joint_angles"], img_list, mask_lst, skeleton_visualizer_args, output_dir=os.path.join(args.output_dir, "CMA-ES_w_joint_angles"))
    print("Plotted CMA-ES with joint angles")

    # # Gradient Descent without joint angles
    # plot_bi_manual_skeleton(traj_GD_wo_joint["cTr"], traj_GD_wo_joint["joint_angles"], img_list, mask_lst, skeleton_visualizer_args, output_dir=os.path.join(args.output_dir, "GD_wo_joint_angles"))
    # print("Plotted Gradient Descent without joint angles")

    # # Gradient Descent with joint angles
    # plot_bi_manual_skeleton(traj_GD_w_joint["cTr"], traj_GD_w_joint["joint_angles"], img_list, mask_lst, skeleton_visualizer_args, output_dir=os.path.join(args.output_dir, "GD_w_joint_angles"))
    # print("Plotted Gradient Descent with joint angles")
