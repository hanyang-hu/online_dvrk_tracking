import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models.detection.backbone_utils import FeaturePyramidNetwork

from scipy.interpolate import PchipInterpolator
from scipy import ndimage


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        x: [B, N, d_model]
        """
        B, N, _ = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0)  # [1, N]
        pos = self.pos_embed(positions)  # [1, N, d_model]
        return x + pos
        

class ContourTransformer(nn.Module):
    def __init__(
        self,
        feature_dim,
        d_model=16,
        num_heads=2,
        num_layers=2,
        max_len=300,
        dropout=0.1,
        conv_kernel_size=5  # local conv kernel size
    ):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoding = LearnedPositionalEncoding(d_model=d_model, max_len=max_len)

        # Local 1D convolution along contour
        # Conv1d expects [B, C, N]
        self.local_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2
        )
        self.conv_norm = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,   # IMPORTANT: [B, N, C]
            activation="gelu",
            norm_first=True     # Pre-LN = more stable
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Per-point prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        """
        x: [B, N, F]
        return: [B, N] heatmap (unnormalized logits)
        """
        x = self.input_proj(x)       # [B, N, d_model]
        x = self.pos_encoding(x)     # inject positional info

        # Local convolution with residual
        # Conv1d expects [B, C, N], output back to [B, N, C]
        x_conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv_norm(x + x_conv)  # residual + norm

        # Transformer encoder for global context
        x = self.encoder(x)

        # Output heatmap per point
        heatmap = self.head(x).squeeze(-1)  # [B, N]
        return heatmap


class SmoothBias(nn.Module):
    def __init__(self, N, kernel_size):
        super().__init__()
        self.raw_bias = nn.Parameter(torch.zeros(N))

        # # fixed smoothing kernel
        # kernel = torch.ones(1, 1, kernel_size) / kernel_size
        # self.register_buffer("kernel", kernel)
        # self.padding = kernel_size // 2

    def forward(self):
        b = self.raw_bias[None, None, :]  # [1,1,N]
        # # Pad for 'same' convolution
        # b = torch.nn.functional.pad(b, (self.padding, self.padding), mode='replicate')
        # # Apply 1D convolution to smooth the bias
        # b = torch.nn.functional.conv1d(b, self.kernel)
        b = b.squeeze()

        # Ensure symmetric bias
        N = b.size(0)
        b = 0.5 * (b + b[torch.arange(N-1, -1, -1)])  # symmetric

        return b


class ContourTipNet(nn.Module):
    """Main model for contour tip keypoint detection."""
    def __init__(self, feature_dim=6, max_len=300):
        super().__init__()
        # print("Initializing ContourTipNet model...")

        self.model = ContourTransformer(
            feature_dim=feature_dim,
            d_model=16,
            num_heads=2,
            max_len=max_len,
        )

        self.heatmap_bias = SmoothBias(N=max_len, kernel_size=5)
    
    def forward(self, x):
        """
        x: [B, N, F]
        returns: [B, N] heatmap
        """
        # Ensure flip equivariance by processing both original and flipped inputs
        x_flipped = x[ :, torch.arange(x.size(1)-1, -1, -1), :]  # Flip along N dimension
        x_cat = torch.cat([x, x_flipped], dim=0)  # [2B, N, F]

        # Process concatenated inputs through the model
        heatmap_cat = self.model(x_cat)

        # Split outputs back
        B = x.size(0)
        heatmap = heatmap_cat[:B]  # [B, N]
        heatmap_flipped = heatmap_cat[B:]  # [B, N]
        heatmap_unflipped = heatmap_flipped[ :, torch.arange(heatmap_flipped.size(1)-1, -1, -1)]  # Un-flip

        # Average the heatmaps
        heatmap_final = 0.5 * (heatmap + heatmap_unflipped)

        # Ensure heatmap output is positive by applying softplus

        # Adding learnable bias before sigmoid
        heatmap_final = torch.sigmoid(heatmap_final + self.heatmap_bias().unsqueeze(0))

        return heatmap_final
        

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



@torch.no_grad()
def detect_keypoints(model, mask, contour_length=200):
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

    kpt_indices = kpts_solve(heatmap)

    if len(kpt_indices) < 2:
        # print("Keypoint detection failed.")
        return None

    kpt_coords = []
    for idx in kpt_indices:
        coord = contour[idx][::-1]  # (x, y)
        kpt_coords.append(coord)
    kpts_np = np.stack(kpt_coords, axis=0)  # (2, 2)

    return torch.from_numpy(kpts_np).float().to(mask.device)  # (2, 2)



class DeconvHead(nn.Module):
    def __init__(self, in_channels=128, out_channels=1):
        super().__init__()

        self.layers = nn.Sequential(
            # 56 → 112
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),

            # 112 → 224
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),

            # smoothing before output
            nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.layers(x)


class Tip2DNet(nn.Module):
    def __init__(self, mask_size=224, pretrained=False, fpn_dim=64, use_attention=False):
        super().__init__()
        assert mask_size == 224

        backbone = resnet18(pretrained=pretrained)
        backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

        # ResNet layers
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # C2: 56×56
        self.layer2 = backbone.layer2  # C3: 28×28
        self.layer3 = backbone.layer3  # C4: 14×14
        self.layer4 = backbone.layer4  # C5: 7×7

        # Whether to apply attention to C4: 14×14
        self.use_attention = use_attention
        if use_attention:
            # self.attention_layer = nn.TransformerEncoderLayer(
            #     d_model=512,
            #     nhead=8,
            #     dim_feedforward=2048,
            #     dropout=0.1,
            #     activation="relu",
            # )
            # self.transformer_encoder = nn.TransformerEncoder(
            #     self.attention_layer,
            #     num_layers=1
            # )
            # self.positional_encoding = LearnedPositionalEncoding(d_model=512, max_len=49)
            self.attention_layer = nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation="relu",
            )
            self.transformer_encoder = nn.TransformerEncoder(
                self.attention_layer,
                num_layers=1
            )
            self.positional_encoding = LearnedPositionalEncoding(d_model=256, max_len=196)

        # FPN using torchvision FPN
        in_channels_list = [64, 128, 256, 512]  # C2–C5
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=fpn_dim
        )

        # Fusion conv after concatenation of all FPN levels
        self.fpn_fuse = nn.Conv2d(fpn_dim * 4, fpn_dim, 3, padding=1)

        self.head = DeconvHead(in_channels=fpn_dim, out_channels=1)

    def forward(self, x):
        heatmap = self.raw_predict(x)
        return torch.sigmoid(heatmap)

    def raw_predict(self, x):
        """
        Forward pass without sigmoid activation.
        """
        x = self.stem(x)
        c2 = self.layer1(x)  # 56×56
        c3 = self.layer2(c2) # 28×28
        c4 = self.layer3(c3) # 14×14

        # Apply attention to C4 if enabled
        if self.use_attention:
            B, C, H, W = c4.shape
            c4_flat = c4.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
            c4_flat = self.positional_encoding(c4_flat)
            c4_flat = self.transformer_encoder(c4_flat)
            c4 = c4_flat.permute(0, 2, 1).view(B, C, H, W)

        c5 = self.layer4(c4) # 7×7

        # Pass features through FPN
        features = {"0": c2, "1": c3, "2": c4, "3": c5}
        fpn_out = self.fpn(features)

        # Upsample all levels to P2 resolution (56×56) and concatenate
        p2 = fpn_out["0"]
        p3 = F.interpolate(fpn_out["1"], size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p4 = F.interpolate(fpn_out["2"], size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p5 = F.interpolate(fpn_out["3"], size=p2.shape[-2:], mode="bilinear", align_corners=False)

        fpn_cat = torch.cat([p2, p3, p4, p5], dim=1)
        fpn_feat = self.fpn_fuse(fpn_cat)

        heatmap = self.head(fpn_feat)
        return heatmap


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


def detect_keypoints_2d(model, mask):
    # Resize to 224x224 by bilinear interpolation
    mask_resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    )  # [1, 1, 224, 224]

    # Predict heatmap using Tip2DNet
    pred_heatmap_raw = model.raw_predict(mask_resized)  # [1, 1, 224, 224]
    pred_heatmap = torch.sigmoid(pred_heatmap_raw)
    pred_heatmap = pred_heatmap.squeeze(1).squeeze(0) 
    pred_heatmap = pred_heatmap.cpu().numpy()  # [224, 224]

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
    if len(keypoints) < 2:
        return None

    return torch.from_numpy(keypoints).float().to(mask.device)  # (K, 2)