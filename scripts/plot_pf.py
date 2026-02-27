import argparse
import torch
import os
import sys
import cv2
import glob
import imutils

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

    # Setting for our custom data
    args.height = 480
    args.width = 640
    args.fx, args.fy, args.px, args.py = 1025.88223, 1025.88223, 167.919017, 234.152707

    args.scale = 1.0

    # scale the camera parameters
    args.width = int(args.width * args.scale)
    args.height = int(args.height * args.scale)
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale

    return args


def segmentColorAndGetKeyPoints(
        img,
        hsv_min=(40, 80, 80),
        hsv_max=(80, 255, 255),
        draw_contours=False):

    hsv = cv2.cvtColor((img).astype(np.uint8), cv2.COLOR_RGB2HSV)

    # Strengthen green separation
    hsv = hsv.astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)   # boost saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)   # increase brightness
    hsv = hsv.astype(np.uint8)

    lower = np.array(hsv_min)
    upper = np.array(hsv_max)

    mask = cv2.inRange(hsv, lower, upper)

    # Explicitly suppress dark pixels (strong black filtering)
    value_mask = hsv[:, :, 2] > 100
    mask = mask & value_mask.astype(np.uint8) * 255

    kernel_close = np.ones((5, 5), np.uint8)
    kernel_open  = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    centroids = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 30:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        centroids.append(np.array([cX, cY]))

        if draw_contours:
            cv2.drawContours(img, [c], -1, (0, 0, 255), 2)
            cv2.circle(img, (int(cX), int(cY)), 4, (0, 255, 0), -1)

    return np.array(centroids), img


def plot_skeleton(cTr_traj, joint_angle_traj, img_list, mask_lst, skeleton_visualizer_args, output_dir, color=(0, 255, 0), paint_mask=False, paint_marker=False):
    os.makedirs(output_dir, exist_ok=True)

    cTr_traj = cTr_traj.cuda()
    joint_angle_traj = joint_angle_traj.cuda()

    # skeleton_visualizer_args are (model, ctrnet_args, None, intr, p_local1, p_local2)
    skeleton_visualizer = SkeletonVisualizer(*skeleton_visualizer_args, use_filter=False)
    for i in range(min(len(img_list), len(cTr_traj))):
        img = img_list[i]
        mask = mask_lst[i] > 0
        blended = img.copy()

        if paint_mask:
            alpha = 0.3
            blended = img.copy()

            # Blend ONLY where mask is True
            blended[mask] = (
                img[mask] * (1 - alpha) +
                np.array(color, dtype=np.uint8) * alpha
            ).astype(np.uint8)

        if paint_marker:
            _, blended = segmentColorAndGetKeyPoints(blended, draw_contours=True)

        # Plot left arm skeleton
        blended = skeleton_visualizer.plot_skeleton_overlay(
            blended,
            cTr_traj[i],
            joint_angle_traj[i],
        )

        output_path = os.path.join(output_dir, f"{output_dir.split('/')[-1]}_{i+1:04d}.png")
        cv2.imwrite(output_path, blended)
        # print(f"Saved overlay image to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting script for dVRK tracking results")
    parser.add_argument("--results_dir", type=str, default="./pf_tracking_results/", help="Directory containing the results files")
    parser.add_argument("--data_dir", type=str, default="./data/custom/gt_masks/", help="Directory containing the data files")
    parser.add_argument("--video_dir", type=str, default="./data/custom/", help="Directory to save the generated videos")
    parser.add_argument("--bag_idx", type=str, default="bag1", help="Name of the bag to process (e.g., bag1, bag2, etc.)")
    parser.add_argument("--output_dir", type=str, default="./plots/pf/", help="Directory to save the generated plots")
    args = parser.parse_args()

    # Load pose results from the specified directory
    CMA_ES_file = f"cma_es_{args.bag_idx}_tracking_results.pt"
    CMA_ES_traj = torch.load(os.path.join(args.results_dir, CMA_ES_file))
    PF_file = f"pf_{args.bag_idx}_tracking_results.pt"
    PF_traj = torch.load(os.path.join(args.results_dir, PF_file))

    # Load masks for the specified bag
    mask_lst = []
    frame_start = 1
    mask_dir = os.path.join(args.data_dir, args.bag_idx, "PSM3")
    frame_end = len([name for name in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, name)) and name.isdigit()])
    # print(mask_dir, frame_start, frame_end)
    for frame_idx in range(frame_start, frame_end):
        frame_dir = os.path.join(mask_dir, f"{frame_idx}")
        # print(f"Loading mask from {frame_dir}")

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

        mask_lst.append(ref_img)

    # Load data for the specified bag
    video_path = os.path.join(args.video_dir, args.bag_idx, "left.mp4")
    print(f"Loading video from {video_path}")
    cap = cv2.VideoCapture(video_path)
    img_list = []
    img_size = mask_lst[0].shape
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_size[1], img_size[0]))
        img_list.append(frame)

    # Discard the first and last frames
    img_list = img_list[1:]

    print(f"Loaded {len(img_list)} video frames and {len(mask_lst)} masks")
    print(f"Number of pose estimates: CMA-ES traj length = {len(CMA_ES_traj['cTr'])}, Particle Filter traj length = {len(PF_traj['cTr'])}")
    img_lst = img_list[:len(CMA_ES_traj['cTr'])]
    mask_lst = mask_lst[:len(CMA_ES_traj['cTr'])]
    print(f"After trimming, {len(img_lst)} video frames and {len(mask_lst)} masks will be used for plotting, which matches the number of pose estimates.")

    # # Save the overlay of the masks on the video frames as ./args.output_dir/1.png, ... ./args.output_dir/N.png
    # # Both masks use the same color (e.g., red) for simplicity, and the overlay is blended with the original image using an alpha value of 0.5
    # os.makedirs(args.output_dir, exist_ok=True)
    # for idx, (img, mask) in enumerate(zip(img_list, mask_lst)):
    #     overlay = img.copy()
    #     overlay[mask > 0] = (0, 0, 255) 
    #     alpha = 0.5
    #     blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    #     output_path = os.path.join(args.output_dir, f"{idx+1:04d}.png")
    #     cv2.imwrite(output_path, blended)
    #     print(f"Saved overlay image to {output_path}")

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
    # Plot skeleton overlay for CMA-ES results
    plot_skeleton(
        CMA_ES_traj["cTr"],
        CMA_ES_traj["joint_angles"],
        img_list,
        mask_lst,
        skeleton_visualizer_args,
        os.path.join(args.output_dir, "cma_es"),
        color=(0, 255, 0),
        paint_mask=True,
        paint_marker=False,
    )

    # Plot skeleton overlay for Particle Filter results
    plot_skeleton(
        PF_traj["cTr"],
        PF_traj["joint_angles"],
        img_list,
        mask_lst,
        skeleton_visualizer_args,
        os.path.join(args.output_dir, "particle_filter"),
        color=(255, 0, 0),
        paint_mask=False,
        paint_marker=True,
    )