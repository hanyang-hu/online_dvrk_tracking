import os
import cv2
import glob
import numpy as np
from PIL import Image
import time

import torch
import torch.nn.functional as F

from skimage.morphology import skeletonize, thin


import torch
import numpy as np
import cv2


def extract_contour(mask: torch.Tensor):
    """
    mask: (H,W) torch.float {0,1}, single component
    Returns: contour points as np.ndarray (N,2) in (y,x)
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)

    # External contours only, no holes
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.zeros((0,2), dtype=np.int32)

    # Pick the longest contour
    main_contour = max(contours, key=lambda c: len(c))
    main_contour = main_contour.squeeze(1)  # shape (N,2), columns = (x,y)
    main_contour = main_contour[:, [1,0]]   # convert to (y,x)

    # epsilon = 0.01 * cv2.arcLength(main_contour, True)
    # main_contour = cv2.approxPolyDP(main_contour, epsilon, True) # approximate polygonal curve
    # main_contour = main_contour.squeeze(1)  # shape (N,2), columns = (y,x)

    # # mask: float32 {0,1}
    # contours = measure.find_contours(mask_np, level=0.5)

    # # Pick the longest contour
    # main_contour = max(contours, key=lambda c: len(c))
    # # Convert from (row, col) -> (y, x)
    # main_contour = main_contour[:, [0,1]]

    return main_contour


def contour_centroid(contour: np.ndarray):
    yc = contour[:,0].mean()
    xc = contour[:,1].mean()
    return np.array([yc, xc])

def edge_signal(contour: np.ndarray, centroid: np.ndarray):
    vectors = contour - centroid
    distances = np.linalg.norm(vectors, axis=1)
    return distances


if __name__ == "__main__":
    import argparse
    import tqdm
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Skeleton extraction script")
    parser.add_argument('--task_name', type=str, default="consecutive_prediction/rw15", help='Name of the task/data folder')
    args = parser.parse_args()
    task_name = args.task_name

    data_dir = "./data/" + task_name
    
    frame_start = 0
    frame_end = len(os.listdir(data_dir)) - 1

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

        # Load mask
        mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
        # mask = torch.from_numpy(mask).float() / 255.0

        # ---------- Extract main component ----------
        # Select the largest connected component as the mask
        num_labels, labels_im = cv2.connectedComponents(mask)
        largest_label = 1 + np.argmax([np.sum(labels_im == i) for i in range(1, num_labels)])
        mask[labels_im != largest_label] = 0

        clean_mask = mask = torch.from_numpy((mask).astype(np.float32)) 

        # ---------- Extract contour ----------
        contour = extract_contour(clean_mask)

        # ---------- Compute centroid and 1-D signal ----------
        centroid = contour_centroid(contour)
        signal = edge_signal(contour, centroid)

        # Smooth signal by DFT
        signal_fft = np.fft.fft(signal)
        freq_cutoff_ratio = 0.01
        N = len(signal)
        freq_cutoff = int(N * freq_cutoff_ratio)
        signal_fft[freq_cutoff:(N - freq_cutoff)] = 0
        signal_smooth = np.fft.ifft(signal_fft).real
        signal = signal_smooth

        # ---------- Find local maxima of the signal (considering circularity) ----------
        signal_extended = np.concatenate([signal[-5:], signal, signal[:5]])
        local_max_indices = (np.diff(np.sign(np.diff(signal_extended))) < 0).nonzero()[0] + 1  # +1 due to diff reducing length by 1
        local_max_indices = local_max_indices[(local_max_indices >= 5) & (local_max_indices < len(signal_extended) - 5)] - 5  # shift back
        local_max_values = signal[local_max_indices]

        # Create overlay of contour on mask
        vis = cv2.cvtColor((mask.cpu().numpy()).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # Draw contour in lines
        for j in range(len(contour)):
            y0, x0 = contour[j]
            y1, x1 = contour[(j+1)%len(contour)]
            cv2.line(vis, (x0,y0), (x1,y1), (0,0,255), 1)  # Blue in BGR
        cv2.circle(vis, (int(centroid[1]), int(centroid[0])), 3, (0,255,0), -1) 

        # Label local maxima by large red dots and connect to the centroid
        for idx in local_max_indices:
            y, x = contour[idx]
            cv2.circle(vis, (x,y), 4, (0,0,255), -1)  # Red in BGR
            cv2.line(vis, (int(centroid[1]), int(centroid[0])), (x,y), (255,0,0), 1)  # Blue line

        # ---------- Plot the signal and concatenate beside overlay ----------
        fig, ax = plt.subplots(figsize=(vis.shape[0]/100, vis.shape[0]/100), dpi=100)
        ax.plot(signal, linewidth=2, color='green')
        ax.scatter(local_max_indices, local_max_values, color='red', s=20)
        ax.set_title(f"Frame {i} Edge Signal", fontsize=8)
        ax.set_xlabel("Contour Point Index")
        ax.set_ylabel("Distance to Centroid")
        
        fig.tight_layout(pad=0)

        # Render to ARGB buffer
        fig.canvas.draw()
        signal_img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        h, w = fig.canvas.get_width_height()
        signal_img = signal_img.reshape((h, w, 4))

        # Convert ARGB -> RGB
        signal_img = signal_img[:, :, [1,2,3]]
        plt.close(fig)
        
        # Resize to target size
        signal_img = cv2.resize(signal_img, (vis.shape[0], vis.shape[0]))

        vis = np.concatenate([vis, signal_img], axis=1)

        overlay_frames.append(Image.fromarray(vis))


    # ---------- Save GIF ----------
    output_gif = f"skeleton_{task_name.replace('/', '_')}.gif"
    overlay_frames[0].save(
        output_gif,
        save_all=True,
        append_images=overlay_frames[1:],
        duration=100,
        loop=0
    )

    print(f"[OK] Saved overlay GIF to {output_gif}")



"""
python scripts/skeleton_test.py --task_name consecutive_prediction/rw1
python scripts/skeleton_test.py --task_name consecutive_prediction/rw5
python scripts/skeleton_test.py --task_name consecutive_prediction/rw6
python scripts/skeleton_test.py --task_name consecutive_prediction/rw7
python scripts/skeleton_test.py --task_name consecutive_prediction/rw8
python scripts/skeleton_test.py --task_name consecutive_prediction/rw9
python scripts/skeleton_test.py --task_name consecutive_prediction/rw14
python scripts/skeleton_test.py --task_name consecutive_prediction/rw15


python scripts/skeleton_test.py --task_name synthetic_data/syn_rw1
python scripts/skeleton_test.py --task_name synthetic_data/syn_rw5
python scripts/skeleton_test.py --task_name synthetic_data/syn_rw6
python scripts/skeleton_test.py --task_name synthetic_data/syn_rw7
python scripts/skeleton_test.py --task_name synthetic_data/syn_rw8
python scripts/skeleton_test.py --task_name synthetic_data/syn_rw14
python scripts/skeleton_test.py --task_name synthetic_data/syn_rw15
"""