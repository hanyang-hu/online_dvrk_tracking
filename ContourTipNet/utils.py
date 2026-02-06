import torch
import numpy as np
import cv2

from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
        

def downsample_contour_pchip(main_contour, fixed_length=300):
    """
    Arc-length–parametrized PCHIP downsampling for an open contour.
    """
    pts = np.asarray(main_contour, dtype=np.float64)
    N = len(pts)

    if N < fixed_length:
        return None

    # ---- arc-length parameter ----
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])

    if s[-1] == 0:
        return None

    # ---- uniform resampling ----
    s_new = np.linspace(0, s[-1], fixed_length)

    # ---- PCHIP interpolation ----
    fx = PchipInterpolator(s, pts[:, 0], extrapolate=False)
    fy = PchipInterpolator(s, pts[:, 1], extrapolate=False)

    x_new = fx(s_new)
    y_new = fy(s_new)

    # Safety (should not trigger)
    if np.any(np.isnan(x_new)) or np.any(np.isnan(y_new)):
        return None

    return np.stack((x_new, y_new), axis=1)


def edge_signal(contour: np.ndarray, centroid: np.ndarray):
    vectors = contour - centroid
    distances = np.linalg.norm(vectors, axis=1)
    return distances
    

def extract_contour_features(mask: torch.Tensor, contour_length=300, border_cut_thres=1):
    """
    Extract contour features from a binary mask.
    Args:
        mask: (H,W) torch.float {0,1}, single component
        contour_length: desired output contour length
        # border_cut_thres: threshold (in pixels) to cut border connections

    Returns: 
        contour_np (N, 2): downsampled contour points (y, x)
        scaled contour features (y, x, dist, nx, ny, curvature)
      (N = contour_length, returns None if failure)
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)
    mask_np = (mask_np > 0.5).astype(np.uint8) * 255

    # External contours only, no holes
    contour_lst, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contour_lst) == 0:
        return None, None

    # Pick the longest contour
    main_contour = max(contour_lst, key=lambda c: len(c))
    main_contour = main_contour.squeeze(1)  # shape (N,2), columns = (x,y)
    main_contour = main_contour[:, [1,0]]   # convert to (y,x)

    # Filter out the point on the image border
    H, W = mask.shape
    N = len(main_contour)
    valid_indices = []
    for idx, (y, x) in enumerate(main_contour):
        if y <= 1 or y >= H - 1 or x <= 1 or x >= W - 1:
            continue
        valid_indices.append(idx)

    if not valid_indices:
        return None, None

    valid_indices = np.asarray(valid_indices, dtype=np.int32)

    # Differences with wrap-around
    diffs = (np.roll(valid_indices, -1) - valid_indices) % N

    # Adjacency means diff <= border_cut_thres
    breaks = np.where(diffs > border_cut_thres)[0]

    if len(breaks) > 1:
        return None, None  # multiple breaks — ambiguous

    if len(breaks) == 1:
        cut = breaks[0] + 1   # first index AFTER the deleted segment
    else:
        # No deletion or full contour — keep original order
        cut = 0

    # Rotate indices
    valid_indices = np.roll(valid_indices, -cut)

    # Apply re-ordering to contour
    main_contour = main_contour[valid_indices]

    # Use interpolation to obtain downsampled contour
    contour = downsample_contour_pchip(main_contour, fixed_length=contour_length)
    N = len(contour) # updated length after downsampling

    # Compute the unit normal of the contour at each contour point as a feature
    contour_xy = np.array([[x, y] for y, x in contour], dtype=np.float32)  # [N,2] in (x,y)

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

    # Apply transformations to enforce scale invariance
    # Compute centroid
    centroid = contour[:, :2].mean(axis=0, keepdims=True)  # [1, 2]

    # Put features together
    features = np.zeros((N, 6), dtype=np.float32) 
    features[:, 0] = contour[:, 0]                      # y
    features[:, 1] = contour[:, 1]                      # x
    features[:, 2] = edge_signal(contour, centroid)     # distance to centroid
    features[:, 3] = normals[:, 0]                      # normal x
    features[:, 4] = normals[:, 1]                      # normal y
    features[:, 5] = curvature                          # curvature
    
    # Center features
    centered_coordinates = features[:, :2] - centroid  # [N, 2]

    # Scale centered coordinates by distances
    scale = np.sqrt((centered_coordinates ** 2).sum(axis=1).mean())
    scaled_coordinates = centered_coordinates / scale
    
    # Scale distances to the range [0, 1]
    distances = features[:, 2]
    scaled_distances = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

    # Combine scaled features
    features = np.concatenate([scaled_coordinates, scaled_distances.reshape(-1, 1), features[:, 3:]], axis=-1)  # [N, 6]

    # Convert to tensor
    features_tensor = torch.from_numpy(features).unsqueeze(0).to(mask.device) # [1, N, 6]

    return contour, features_tensor


def kpts_solve(pred, detectability_thres=1.9, heatmap_thres=1e-3):
    """
    Closed-form 1D EMD minimization to select up to 2 keypoints.
    Args:
        pred: 1D histogram (array of positive numbers, mass not normalized)
        detectability_thres: minimum total mass to consider keypoint detection
    Returns:
        kpts_indices: list of at most 2 indices (bin positions) that minimize 1D EMD
    """
    # Determine number of keypoints M to select by thresholding
    M = 2 if pred.sum() >= detectability_thres else 0
    pred[pred < heatmap_thres] = 0.0  # zero out very small values to improve stability

    cumsum_pred = np.cumsum(pred)  # cumulative target mass

    if M == 0:
        return []
    # elif M == 1:
    #     # find median of cumulative mass
    #     median_mass = M / 2
    #     k1_index = np.searchsorted(cumsum_pred, median_mass)
    #     return [k1_index,]
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

        # if best_k1 == best_k2:
        #     # If both keypoints collapse to the same bin, treat as failure
        #     return None

        return [best_k1, best_k2]


@torch.no_grad()
def detect_keypoints(model, mask, contour_length=200, detectability_thres=1.8, distance_threshold=10.0):
    """
    Args:
        model: ContourTipNet
        mask: (H,W) torch.float binary mask
    returns: keypoint coordinates if detected, else None
    """

    contour, contour_features = extract_contour_features(
        mask=mask * 255., 
        contour_length=contour_length, 
        # border_cut_thres=border_cut_thres
    )

    if contour is None or contour_features is None:
        # print("Contour extraction failed.")
        return None

    heatmap = model(contour_features.to(torch.float32))  # [1, N]
    heatmap = heatmap.squeeze(0).cpu().numpy()  # [N]

    kpt_indices = kpts_solve(heatmap, detectability_thres=detectability_thres)

    if len(kpt_indices) < 2:
        # print("Keypoint detection failed.")
        return None

    kpt_coords = []
    for idx in kpt_indices:
        coord = contour[idx][::-1]  # (x, y)
        kpt_coords.append(coord)
    kpts_np = np.stack(kpt_coords, axis=0)  # (2, 2)

    # If two keypoints are too close, treat as failure
    if np.linalg.norm(kpts_np[0] - kpts_np[1]) < distance_threshold:
        return None

    return torch.from_numpy(kpts_np).float().to(mask.device)  # (2, 2)


