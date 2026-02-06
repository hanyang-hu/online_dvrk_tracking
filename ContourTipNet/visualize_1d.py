import torch
import numpy
import cv2
import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from model import ContourTipNet
from data_generation import extract_contour, edge_signal, contour_centroid


def kpts_solve_emd(pred):
    """
    Closed-form 1D EMD minimization to select up to 2 keypoints.
    pred: 1D histogram (array of positive numbers, mass not normalized)
    Returns:
        kpts_indices: list of 0, 1, or 2 indices (bin positions) that minimize 1D EMD
    """
    # Determine number of keypoints M to select by thresholding
    M = 2 if pred.sum() >= 1.95 else 0
    # print(pred.shape)
    pred[pred < 1e-4] = 0.0  # zero out very small values

    cumsum_pred = np.cumsum(pred)  # cumulative target mass

    if M == 0:
        return []
    elif M == 1:
        # find median of cumulative mass
        median_mass = M / 2
        k1_index = np.searchsorted(cumsum_pred, median_mass)
        return [k1_index,]
    else:
        # M == 2
        # Find bin indices for 1/4 and 3/4 of cumulative mass
        quarter_mass = M / 4
        three_quarter_mass = 3 * M / 4
        k1_index = np.searchsorted(cumsum_pred, quarter_mass)
        k2_index = np.searchsorted(cumsum_pred, three_quarter_mass)

        # return [k1_index, k2_index]

        # Use closed-form solution as an inital guess, then do local search
        # Only consider per-bin values after Gaussian smoothing, not EMD
        pred_smooth = gaussian_filter1d(pred, sigma=3.0)
        pred_smooth = gaussian_filter1d(pred, sigma=3.0)
        best_value_k1 = -float('inf')
        best_value_k2 = -float('inf')
        best_k1 = k1_index
        best_k2 = k2_index

        for delta1 in range(-7, 7):
            cand_k1 = k1_index + delta1
            if cand_k1 < 0 or cand_k1 >= len(pred):
                continue
            # Compute cost
            if best_value_k1 < pred_smooth[cand_k1]:
                best_value_k1 = pred_smooth[cand_k1]
                best_k1 = cand_k1

        for delta2 in range(-7, 7):
            cand_k2 = k2_index + delta2
            if cand_k2 < 0 or cand_k2 >= len(pred):
                continue
            # Compute cost
            if best_value_k2 < pred_smooth[cand_k2]:
                best_value_k2 = pred_smooth[cand_k2]
                best_k2 = cand_k2

        return [best_k1, best_k2]


def kpts_solve(pred, thresh=0.5, min_dist=5, max_kpts=2):
    """
    Predict keypoint indices using thresholding + 1D connected components.

    pred: 1D numpy array (heat / confidence)
    thresh: absolute threshold to form components
    min_dist: minimum separation between keypoints (in indices)
    max_kpts: maximum number of keypoints to return
    """
    pred = pred.copy()
    pred[pred < thresh] = 0.0

    if pred.sum() == 0:
        return []

    active = pred > 0
    N = len(pred)

    # --- extract one peak per connected component ---
    peaks = []  # (index, score)
    i = 0
    while i < N:
        if not active[i]:
            i += 1
            continue

        start = i
        while i < N and active[i]:
            i += 1
        end = i  # [start, end)

        seg = pred[start:end]
        k = start + np.argmax(seg)
        peaks.append((k, pred[k]))

    # --- enforce minimum distance (greedy NMS) ---
    peaks.sort(key=lambda x: x[1], reverse=True)

    selected = []
    for k, v in peaks:
        if all(abs(k - ks) >= min_dist for ks in selected):
            selected.append(k)
        if len(selected) >= max_kpts:
            break

    return sorted(selected)


import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

def make_heatmap_overlay_image_argb(contour_img, labels_heatmap, pred_heatmap):
    N = len(pred_heatmap)
    fig, ax = plt.subplots(figsize=(contour_img.shape[0]/100, contour_img.shape[0]/100), dpi=100)

    bar_width = max(1.0 / N * 2, 0.5)
    ax.bar(range(N), labels_heatmap, alpha=0.5, label="Selected", width=bar_width)
    ax.bar(range(N), pred_heatmap, alpha=0.5, label="Prediction", width=bar_width)

    ax.set_xlabel("Contour Point Index")
    ax.set_ylabel("Heatmap Value")
    ax.legend(fontsize=8)
    ax.set_title("Contour Heatmap Overlay")
    # plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    plt.tight_layout()

    fig.canvas.draw()
    signal_img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    h, w = fig.canvas.get_width_height()
    signal_img = signal_img.reshape((h, w, 4))
    signal_img = signal_img[:, :, [1,2,3]]  # ARGB -> RGB
    plt.close(fig)

    # Resize heatmap to match contour image
    heatmap_img_resized = cv2.resize(signal_img, (contour_img.shape[0], contour_img.shape[0]))

    overlay_img = np.concatenate([contour_img, heatmap_img_resized], axis=1)
    overlay_img_pil = Image.fromarray(overlay_img)
    return overlay_img_pil



if __name__ == "__main__":
    import argparse
    import tqdm
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="ContourTipNet visualization script")
    parser.add_argument('--task_name', type=str, default="consecutive_prediction/rw15", help='Name of the task/data folder')
    args = parser.parse_args()
    task_name = args.task_name

    data_dir = "../data/" + task_name
    
    frame_start = 0
    frame_end = len(os.listdir(data_dir)) - 1

    # Predict tip location using ContourTipNet
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load trained model
        model = ContourTipNet(feature_dim=6, max_len=200).cuda()
        model.load_state_dict(torch.load("./models/ctn_model.pth", map_location=device))
        model.eval()

        overlay_frames = []

        pbar = tqdm.tqdm(total=frame_end - frame_start, desc="Processing frames")
        for i in range(frame_start, frame_end):
            pbar.update(1)

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

            # Load mask
            num_labels, labels_im = cv2.connectedComponents(mask)
            largest_label = 1 + np.argmax([np.sum(labels_im == i) for i in range(1, num_labels)])
            mask[labels_im != largest_label] = 0

            mask = torch.from_numpy((mask).astype(np.float32)) / 255.0
            
            # Obtain coutour features
            # Convert mask to binary images for contour extraction
            contour = extract_contour((mask > 0.5).to(torch.uint8) * 255, contour_length=200)  # (N,2) in (y,x)

            if contour is None:
                print(f"Frame {i}: No contour found, skipping...")
                # Draw image that fails the contour extraction
                # cv2.imwrite(f"GIFs/CTN_{task_name.replace('/', '_')}_frame_{i:04d}.png", cv2.cvtColor(mask.numpy().astype(np.uint8)*255, cv2.COLOR_GRAY2BGR))
                continue

            # Plot contour and rendered mask by cv2
            contour_img = cv2.cvtColor((mask.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for j in range(len(contour)):
                pts = contour[j]
                y, x = pts
                # downsampled contour are floats, so convert to int
                y, x = int(round(y)), int(round(x))
                cv2.circle(contour_img, (x,y), 1, (255,0,0), -1)  # Red in BGR

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

            # Plot normals on the contour image
            for j in range(0, len(contour), 10):  # plot every 10th normal
                y, x = contour[j]
                nx, ny = normals[j]
                cv2.line(contour_img, (int(x), int(y)), (int(x + nx * 10), int(y + ny * 10)), (0, 255, 255), 1)  # Cyan in BGR

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
            keypoint_indices = kpts_solve(pred_heatmap)
            estimated_keypoint_num = len(keypoint_indices)
            if len(keypoint_indices) > 0:
                keypoint_indices = np.sort(keypoint_indices)
                keypoints = contour[keypoint_indices]  # [K, 2] in (y,x)

                # Plot the keypoints on the contour image
                for kp in keypoints:
                    y, x = kp
                    y, x = int(round(y)), int(round(x))
                    cv2.circle(contour_img, (x,y), 5, (0,255,0), -1)  # Green in BGR

            # Add text info to contour image
            cv2.putText(contour_img, f"Est. Keypoint Num: {estimated_keypoint_num}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Red in BGR

            one_hot_kpts_heatmap = np.zeros_like(pred_heatmap)
            for idx in keypoint_indices:
                one_hot_kpts_heatmap[idx] = 1.0

            # overlay_img = make_heatmap_overlay_image_argb(
            #     contour_img,
            #     labels_heatmap=one_hot_kpts_heatmap,
            #     pred_heatmap=pred_heatmap
            # )
            overlay_img = Image.fromarray(contour_img)
            
            # Concatenate contour image and heatmap image
            overlay_frames.append(overlay_img)
        pbar.close()


    # ---------- Save GIF ----------
    output_gif = f"GIFs/CTN_{task_name.replace('/', '_')}.gif"
    overlay_frames[0].save(
        output_gif,
        save_all=True,
        append_images=overlay_frames[1:],
        duration=20,
        loop=0
    )

    print(f"[OK] Saved overlay GIF to {output_gif}")



"""
python visualize_1d.py --task_name consecutive_prediction/rw1
python visualize_1d.py --task_name consecutive_prediction/rw5
python visualize_1d.py --task_name consecutive_prediction/rw6
python visualize_1d.py --task_name consecutive_prediction/rw7
python visualize_1d.py --task_name consecutive_prediction/rw8
python visualize_1d.py --task_name consecutive_prediction/rw9
python visualize_1d.py --task_name consecutive_prediction/rw14
python visualize_1d.py --task_name consecutive_prediction/rw15


python visualize_1d.py --task_name synthetic_data/syn_rw1
python visualize_1d.py --task_name synthetic_data/syn_rw5
python visualize_1d.py --task_name synthetic_data/syn_rw6
python visualize_1d.py --task_name synthetic_data/syn_rw7
python visualize_1d.py --task_name synthetic_data/syn_rw8
python visualize_1d.py --task_name synthetic_data/syn_rw9
python visualize_1d.py --task_name synthetic_data/syn_rw14
python visualize_1d.py --task_name synthetic_data/syn_rw15
"""