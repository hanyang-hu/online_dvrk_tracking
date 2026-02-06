import torch
import numpy as np
import cv2
import os
import glob
import time
import argparse

from model import ContourTipNet
from data_generation import extract_contour, edge_signal, contour_centroid
from model_2d import Tip2DNet
from visualize_1d import kpts_solve
from visualize_2d import get_local_maxima

from diffcali.models.CtRNet import CtRNet
from diffcali.eval_dvrk.LND_fk import lndFK
from diffcali.utils.projection_utils import *


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


@torch.no_grad()
def detect_1d(model, mask):
    # Compute contour from mask
    contour = extract_contour((mask > 0.5).to(torch.uint8) * 255, contour_length=200)  # (N,2) in (y,x)
    if contour is None:
        return np.zeros((0,2), dtype=np.int32)

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

    centroid = contour_centroid(contour)
    signal_c = edge_signal(contour, centroid)

    # Put features together
    features = np.zeros((N, 6), dtype=np.float32) 
    features[:, 0] = contour[:, 0]  # y
    features[:, 1] = contour[:, 1]  # x
    features[:, 2] = signal_c       # distance to centroid
    features[:, 3] = normals[:, 0]  # normal x
    features[:, 4] = normals[:, 1]  # normal y
    features[:, 5] = curvature      # curvature

    # Apply transformations to enforce scale invariance
    # Compute centroid
    centroid = features[:, :2].mean(axis=0, keepdims=True)  # [1, 2]
    
    # Center features
    centered_coordinates = features[:, :2] - centroid  # [N, 2]

    # Scale centered coordinates by distances
    scale = np.sqrt((centered_coordinates ** 2).sum(axis=1).mean())
    scaled_coordinates = centered_coordinates / scale
    
    # Scale distances to the range [0, 1]
    distances = features[:, 2]
    scaled_distances = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

    # Combine scaled features
    features = np.concatenate([scaled_coordinates, scaled_distances.reshape(-1, 1), features[:, 3:]], axis=-1)  # [N, F]

    # Convert to tensor
    features_tensor = torch.from_numpy(features).unsqueeze(0).cuda()  # [1, N, F]

    # Predict heatmap using ContourTipNet
    pred_heatmap = model(features_tensor)  # [1, N]
    pred_heatmap = pred_heatmap.squeeze(0).cpu().numpy()  # [N]

    # Based on this number, use NMS to select keypoints from the heatmap 
    keypoint_indices = kpts_solve(pred_heatmap, thresh=0.5, min_dist=5)

    if len(keypoint_indices) > 0:
        keypoint_indices = np.sort(keypoint_indices)
        keypoints = contour[keypoint_indices]  # [K, 2] in (y,x)
    else:
        keypoints = np.zeros((0,2), dtype=np.int32)

    # Make sure (x, y) order
    keypoints = keypoints[:, [1,0]]  # [K, 2] in (x,y)

    return keypoints


@torch.no_grad()
def detect_2d(model, mask):
    # Resize to 224x224
    mask_resized = cv2.resize((mask * 255).numpy(), (224, 224))
    mask_tensor = torch.from_numpy(mask_resized / 255.).unsqueeze(0).unsqueeze(0).float().cuda()  # [1, 1, H, W]

    # Predict heatmap using Tip2DNet
    pred_heatmap_raw = model.raw_predict(mask_tensor)  # [1, 1, H, W]
    pred_heatmap = torch.sigmoid(pred_heatmap_raw)
    pred_heatmap = pred_heatmap.squeeze(1).squeeze(0) 
    pred_heatmap = pred_heatmap.cpu().numpy()  # [H, W]

    # Find local maxima in predicted heatmap
    peaks = get_local_maxima(pred_heatmap, min_distance=3, min_area=1, threshold=0.5)
    peaks = sorted(peaks, key=lambda p: pred_heatmap_raw[0, 0, p[0], p[1]], reverse=True)  # Sort by raw heatmap value

    peaks = peaks[:2] # Take top 2 peaks

    # Scale keypoints back to original mask size
    keypoints = []
    h_orig, w_orig = mask.shape
    for (y, x) in peaks:
        x_orig = int(x / 224.0 * w_orig)
        y_orig = int(y / 224.0 * h_orig)
        keypoints.append((x_orig, y_orig))  # (x, y)

    keypoints = np.array(keypoints, dtype=np.int32)  # [K, 2] in (x,y)
    return keypoints


def kpts_dist(pred_kpts, ref_kpts, threshold=10.0):
    """
    Compute the sum of distance between predicted keypoints and reference keypoints.
    pred_kpts: [K, 2]
    ref_kpts:  [K, 2]
    """
    assert pred_kpts.shape[0] == 2 and ref_kpts.shape[0] == 2

    # Extract the two predicted keypoints and two reference keypoints
    p1, p2 = pred_kpts[0], pred_kpts[1]
    r1, r2 = ref_kpts[0], ref_kpts[1]

    # Compute distances for both possible assignments
    dist1 = max(np.linalg.norm(p1 - r1) - threshold, 0) + max(np.linalg.norm(p2 - r2) - threshold, 0)
    dist2 = max(np.linalg.norm(p1 - r2) - threshold, 0) + max(np.linalg.norm(p2 - r1) - threshold, 0)
    avg_dist = min(dist1, dist2)

    return avg_dist


if __name__ == '__main__':
    args = parseArgs()

    trajectory_lst = [
        'synthetic_data/syn_rw1',
        'synthetic_data/syn_rw5',
        'synthetic_data/syn_rw6',
        'synthetic_data/syn_rw7',
        'synthetic_data/syn_rw8',
        'synthetic_data/syn_rw9',
        'synthetic_data/syn_rw14',
        'synthetic_data/syn_rw15',
    ]
    root_dir = '../data/'
    device = "cuda"

    # Load ContourTipNet model
    model_1d = ContourTipNet(feature_dim=6, max_len=200).cuda()
    model_1d.load_state_dict(torch.load("./models/ctn_model.pth", map_location=device))
    model_1d.eval()

    # Load Tip2DNet model
    model_2d = Tip2DNet(mask_size=224, use_attention=False).cuda()
    model_2d.load_state_dict(torch.load("./models/cnn_model.pth", map_location=device))
    model_2d.eval()

    # Camera intrinsics and kpts local coordinates
    # Project keypoints
    intr = torch.tensor(
        [
            [args.fx, 0, args.px], 
            [0, args.fy, args.py], 
            [0, 0, 1]
        ],
        device="cuda",
        dtype=torch.float32
    )

    p_local1 = (
        torch.tensor([0.0, 0.0004, 0.0096])
        .to(torch.float32)
        .cuda()
    )
    p_local2 = (
        torch.tensor([0.0, -0.0004, 0.0096])
        .to(torch.float32)
        .cuda()
    )

    ctr_model = CtRNet(args).cuda()

    for traj_name in trajectory_lst:
        data_dir = root_dir + traj_name

        frame_start = 0
        frame_end = len(os.listdir(data_dir)) - 1

        time_1d_lst = []
        time_2d_lst = []

        acc_1d_cnt = 0
        acc_2d_cnt = 0
        total_cnt = 0

        rmse_1d_lst = []
        rmse_2d_lst = []

        for i in range(frame_start, frame_end):
            # Load masks
            frame_dir = os.path.join(data_dir, f"{i}")

            mask_lst = glob.glob(os.path.join(frame_dir, "*.png"))
            if len(mask_lst) == 0:
                raise ValueError(f"No mask found in {frame_dir}")
            if len(mask_lst) > 1:
                raise ValueError(f"Multiple masks found in {frame_dir}")

            mask_path = mask_lst[0]
            XXXX = mask_path.split("/")[-1].split(".")[0][1:]
            ref_mask_path = os.path.join(frame_dir, "0" + XXXX + ".png")
            mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)

            num_labels, labels_im = cv2.connectedComponents(mask)
            largest_label = 1 + np.argmax([np.sum(labels_im == i) for i in range(1, num_labels)])
            mask[labels_im != largest_label] = 0

            mask = torch.from_numpy((mask).astype(np.float32)) / 255.0

            # Load ground truth pose and compute ground truth keypoints with forward kinematics
            cTr = np.load(os.path.join(frame_dir, "optimized_ctr.npy"))  # [6]
            cTr = torch.from_numpy(cTr).to(torch.float32).cuda()
            joint_angles = np.load(os.path.join(frame_dir, "optimized_joint_angles.npy"))  # [4]
            joint_angles = torch.from_numpy(joint_angles).to(torch.float32).cuda()

            pose_matrix = ctr_model.cTr_to_pose_matrix(cTr.unsqueeze(0)).squeeze().to(torch.float32)  # [4,4]

            R_list, t_list = lndFK(joint_angles)

            p1_img, _ = get_img_coords(p_local1, R_list[2], t_list[2], pose_matrix, intr, ret_cam_coords=True)  # [2]
            p2_img, _ = get_img_coords(p_local2, R_list[3], t_list[3], pose_matrix, intr, ret_cam_coords=True)  # [2]

            gt_keypoints = torch.stack([p1_img, p2_img], dim=0).cpu().numpy()  # [2, 2] in (x,y)

            # Predict keypoints using ContourTipNet
            start_time = time.time()
            torch.cuda.synchronize()
            keypoints_1d = detect_1d(model_1d, mask)  # [K, 2] in (y,x)
            torch.cuda.synchronize()
            end_time = time.time()
            time_1d_lst.append(end_time - start_time)

            # Predict keypoints using Tip2DNet
            start_time = time.time()
            torch.cuda.synchronize()
            keypoints_2d = detect_2d(model_2d, mask)
            torch.cuda.synchronize()
            end_time = time.time()
            time_2d_lst.append(end_time - start_time)

            # print(keypoints_2d, gt_keypoints)

            # Compute RMSE if two keypoints are detected
            if keypoints_1d.shape[0] >= 2:
                rmse_1d = kpts_dist(keypoints_1d, gt_keypoints, threshold=0.0) / 2.0
                rmse_1d_lst.append(rmse_1d)
            if keypoints_2d.shape[0] >= 2:
                rmse_2d = kpts_dist(keypoints_2d, gt_keypoints, threshold=0.0) / 2.0
                rmse_2d_lst.append(rmse_2d)

            # Count successful detections (within 10 pixels) if two keypoints are detected
            if keypoints_1d.shape[0] >= 2 and kpts_dist(keypoints_1d, gt_keypoints) == 0:
                acc_1d_cnt += 1
            if keypoints_2d.shape[0] >= 2 and kpts_dist(keypoints_2d, gt_keypoints) == 0:
                acc_2d_cnt += 1
            total_cnt += 1

            # if (keypoints_1d.shape[0] >= 2 and kpts_dist(keypoints_1d, gt_keypoints) == 0) and not (keypoints_2d.shape[0] >= 2 and kpts_dist(keypoints_2d, gt_keypoints) == 0):
            #     print(f"Frame {i} in {traj_name}: 1D success, 2D fail")
            #     print(f"    1D keypoints: {keypoints_1d}, 2D keypoints: {keypoints_2d}, GT keypoints: {gt_keypoints}")

        print(f"Trajectory: {traj_name}")
        print(f"    1D Keypoint Detection Accuracy: {acc_1d_cnt}/{total_cnt} = {acc_1d_cnt/total_cnt*100:.2f}%")
        print(f"    Average 1D RMSE: {np.mean(rmse_1d_lst):.4f} pixels")
        print(f"    Average 1D detection time: {np.mean(time_1d_lst):.4f} seconds")
        print(f"    2D Keypoint Detection Accuracy: {acc_2d_cnt}/{total_cnt} = {acc_2d_cnt/total_cnt*100:.2f}%")
        print(f"    Average 2D RMSE: {np.mean(rmse_2d_lst):.4f} pixels")
        print(f"    Average 2D detection time: {np.mean(time_2d_lst):.4f} seconds\n")
