import torch
import numpy
import cv2
import os
import glob
import numpy as np
from PIL import Image

from model_2d import Tip2DNet

from scipy import ndimage


def get_local_maxima(heatmap, min_distance=3, min_area=5, threshold=0.5):
    """
    Use non-maximum suppression to find local maxima in the heatmap.
    Args:
        heatmap (np.ndarray): 2D array of shape (H, W)
        min_distance (int): Minimum number of pixels separating peaks
        min_area (int): Minimum area of connected component to be considered a peak
        threshold (float): Minimum value to consider a peak
    Returns:
        peaks (list of tuples): List of (y, x) coordinates of local maxima
    """ 
    H, W = heatmap.shape
    heatmap = heatmap.copy()

    # 1. Threshold
    heatmap[heatmap < threshold] = 0.0

    # 2. Find connected components in the thresholded heatmap
    structure = np.ones((3, 3), dtype=np.int32)
    labeled, num = ndimage.label(heatmap > 0, structure=structure)

    # 3. Filter out small components
    for label in range(1, num + 1):
        coords = np.where(labeled == label)
        if len(coords[0]) < min_area:
            heatmap[coords] = 0.0
            labeled[coords] = 0

    peaks = []

    for label in range(1, num + 1):
        coords = np.where(labeled == label)

        if len(coords[0]) == 0:
            continue

        # location of maximum within component
        values = heatmap[coords]
        max_idx = np.argmax(values)
        y = coords[0][max_idx]
        x = coords[1][max_idx]

        peaks.append((y, x))

    # 4. Enforce minimum distance between peaks
    final_peaks = []
    for y, x in sorted(peaks, key=lambda p: heatmap[p], reverse=True):
        if all((y - py) ** 2 + (x - px) ** 2 >= min_distance ** 2
               for py, px in final_peaks):
            final_peaks.append((y, x))

    return final_peaks


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
    threshold = 0.3
    peaks = get_local_maxima(pred_heatmap, min_distance=3, min_area=1, threshold=threshold)
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


if __name__ == "__main__":
    import argparse
    import tqdm
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="2D Keypoint Detection visualization script")
    parser.add_argument('--task_name', type=str, default="consecutive_prediction/rw15", help='Name of the task/data folder')
    args = parser.parse_args()
    task_name = args.task_name

    data_dir = "../data/" + task_name
    
    frame_start = 0
    frame_end = len(os.listdir(data_dir)) - 1

    # Predict tip location using ContourTipNet
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = Tip2DNet(mask_size=224, use_attention=False).cuda()
        model_name = "./models/cnn_model.pth"
        # model_name = "./models/resnet18_fpn_model.pth"
        model.load_state_dict(torch.load(model_name, map_location=device))
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
            
            mask_resized = cv2.resize(mask, (224, 224))
            mask_tensor = torch.from_numpy(mask_resized / 255.).unsqueeze(0).unsqueeze(0).float().to(device)  # [1, 1, H, W]
            # Predict heatmap
            pred_heatmap_raw = model.raw_predict(mask_tensor)  # [1, 1, H, W]
            # pred_heatmap_sigmoid = torch.sigmoid(pred_heatmap)
            # pred_heatmap = torch.clamp(pred_heatmap, min=0.0)
            pred_heatmap = torch.sigmoid(pred_heatmap_raw)
            pred_heatmap = pred_heatmap.squeeze(1).squeeze(0) 
            pred_heatmap = pred_heatmap.cpu().numpy()  # [H, W]
            # pred_heatmap_sigmoid = pred_heatmap_sigmoid.squeeze(1).squeeze(0) 
            # pred_heatmap_sigmoid = pred_heatmap_sigmoid.cpu().numpy()  # [H, W]
            
            # # Find local maxima in predicted heatmap
            # threshold = 0.3
            # peaks = get_local_maxima(pred_heatmap, min_distance=3, min_area=5, threshold=threshold)
            # peaks = sorted(peaks, key=lambda p: pred_heatmap_raw[0, 0, p[0], p[1]], reverse=True)  # Sort by heatmap value
            # peaks = peaks[:2]
            # # print(pred_heatmap_raw.max(), pred_heatmap_raw.min())

            peaks = detect_2d(model, torch.from_numpy((mask).astype(np.float32)) / 255.0)

            # Convert peaks to (224, 224) scale
            peaks = [(int(y / mask.shape[0] * 224), int(x / mask.shape[1] * 224)) for (x, y) in peaks]

            # pred_heatmap[pred_heatmap < threshold] = 0.0

            # Plot keypoints on mask by opencv
            mask_color = cv2.cvtColor(mask_resized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for (y, x) in peaks:
                cv2.circle(mask_color, (x, y), 4, (0, 0, 255), -1)

            # Plot the heatmap and concatenate beside the mask
            pred_heatmap_raw = (pred_heatmap_raw - pred_heatmap_raw.min()) / (pred_heatmap_raw.max() - pred_heatmap_raw.min() + 1e-8)
            heatmap_color = cv2.applyColorMap((pred_heatmap_raw.squeeze(1).squeeze(0).cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_color = cv2.addWeighted(heatmap_color, 0.7, heatmap_color, 0, 0)
            combined = np.concatenate((mask_color, heatmap_color), axis=1)

            # Concatenate contour image and heatmap image
            overlay_frames.append(Image.fromarray(combined))
        pbar.close()


    # ---------- Save GIF ----------
    output_gif = f"GIFs/CNN_{task_name.replace('/', '_')}.gif"
    overlay_frames[0].save(
        output_gif,
        save_all=True,
        append_images=overlay_frames[1:],
        duration=20,
        loop=0
    )

    print(f"[OK] Saved overlay GIF to {output_gif}")



"""
python visualize_2d.py --task_name consecutive_prediction/rw1
python visualize_2d.py --task_name consecutive_prediction/rw5
python visualize_2d.py --task_name consecutive_prediction/rw6
python visualize_2d.py --task_name consecutive_prediction/rw7
python visualize_2d.py --task_name consecutive_prediction/rw8
python visualize_2d.py --task_name consecutive_prediction/rw9
python visualize_2d.py --task_name consecutive_prediction/rw14
python visualize_2d.py --task_name consecutive_prediction/rw15


python visualize_2d.py --task_name synthetic_data/syn_rw1
python visualize_2d.py --task_name synthetic_data/syn_rw5
python visualize_2d.py --task_name synthetic_data/syn_rw6
python visualize_2d.py --task_name synthetic_data/syn_rw7
python visualize_2d.py --task_name synthetic_data/syn_rw8
python visualize_2d.py --task_name synthetic_data/syn_rw14
python visualize_2d.py --task_name synthetic_data/syn_rw15
"""