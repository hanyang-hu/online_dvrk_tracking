import os
import torch
import numpy as np
from torch.utils.data import Dataset
import tqdm
import cv2


class ContourTipDataset(Dataset):
    def __init__(self, feature_dir, label_dir, device="cuda", use_gaussian_label=False, sigma=1.5):
        self.device = device

        feature_files = sorted(os.listdir(feature_dir))

        features_all = []
        labels_all = []

        for fname in tqdm.tqdm(feature_files, desc="Loading Dataset", ncols=110):
            # stem = os.path.splitext(fname)[0]

            # ---------- Load ----------
            features = torch.from_numpy(
                np.loadtxt(os.path.join(feature_dir, fname), dtype=np.float32)
            )

            label_idx = torch.from_numpy(
                np.loadtxt(os.path.join(label_dir, fname), dtype=np.int64)
            )

            N = features.shape[0]

            # ---------- Labels ----------
            labels = torch.zeros(N)
            valid = label_idx[label_idx >= 0]
            labels[valid] = 1.0

            if use_gaussian_label and len(valid) > 0:
                # Convert to Gaussian heatmap labels
                x = torch.arange(N).unsqueeze(0)  # [1, N]
                mu = valid.unsqueeze(1)            # [K, 1]
                gaussians = torch.exp(-0.5 * ((x - mu) ** 2) / (sigma ** 2))  # [K, N]
                labels = torch.max(gaussians, dim=0).values  # [N]

            # ---------- Geometry ----------
            coords = features[:, :2]

            centroid = coords.mean(dim=0, keepdim=True)

            centered = coords - centroid

            scale = torch.sqrt((centered ** 2).sum(dim=1).mean())

            coords_scaled = centered / scale

            d = features[:, 2]

            d = (d - d.min()) / (d.max() - d.min() + 1e-8)

            scaled_features = torch.cat([
                coords_scaled,
                d.unsqueeze(-1),
                features[:, 3:6],
            ], dim=-1)

            features_all.append(scaled_features)
            labels_all.append(labels)

        self.features = features_all
        self.labels = labels_all

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.labels[idx]
        )


class KeypointDataset(Dataset):
    def __init__(self, mask_dir, feature_dir, label_dir, mask_shape=256, sigma=2, device="cuda"):
        """
        Generate heatmap targets for keypoint detection.
        Args:
            mask_dir (str): Directory containing contour mask files.
            feature_dir (str): Directory containing feature files.
            label_dir (str): Directory containing label files.
            mask_shape (tuple): Shape of the mask (in square pixels).
            sigma (float): Standard deviation for Gaussian heatmap.
            device (str): Device to load data onto.
        """
        self.device = device

        feature_files = sorted(os.listdir(feature_dir))

        masks_all = []
        targets_all = []
        distmaps_all = []

        for fname in tqdm.tqdm(feature_files, desc="Loading Dataset", ncols=110):
            # ---------- Load ----------
            features = torch.from_numpy(
                np.loadtxt(os.path.join(feature_dir, fname), dtype=np.float32)
            )

            label_idx = torch.from_numpy(
                np.loadtxt(os.path.join(label_dir, fname), dtype=np.int64)
            )

            masks = cv2.imread(
                os.path.join(mask_dir, fname.replace('.txt', '.png')),
                cv2.IMREAD_GRAYSCALE
            )
            h, w = masks.shape
            masks = cv2.resize(masks, (mask_shape, mask_shape)) # Resize to fixed size
            masks = torch.from_numpy(masks).float() / 255.0  # Normalize to [0, 1]

            N = features.shape[0]

            # Extract keypoint locations
            coords = features[:, :2].numpy()  # [N, 2]
            keypoint_indices = label_idx[label_idx >= 0].numpy()
            keypoint_coords = coords[keypoint_indices]  # [K, 2]

            # Interpolate to mask size
            keypoint_coords[:, 0] = keypoint_coords[:, 0] * (mask_shape / h)
            keypoint_coords[:, 1] = keypoint_coords[:, 1] * (mask_shape / w)

            # # Generate 2D heatmap target by Gaussian kernel
            # heatmap = torch.zeros((mask_shape, mask_shape), dtype=torch.float32)
            # for kp in keypoint_coords:
            #     x, y = int(kp[0]), int(kp[1]) 
            #     if 0 <= x < mask_shape and 0 <= y < mask_shape:
            #         heatmap[x, y] = 1.0

            # heatmap = cv2.GaussianBlur(
            #     heatmap.numpy(),
            #     ksize=(0, 0),
            #     sigmaX=sigma,
            #     sigmaY=sigma
            # )
            # heatmap = torch.from_numpy(heatmap)
                    
            heatmap = torch.zeros((mask_shape, mask_shape), dtype=torch.float32)
            for kp in keypoint_coords:
                x, y = int(kp[0]), int(kp[1]) 
                if 0 <= x < mask_shape and 0 <= y < mask_shape:
                    # Create a Gaussian heatmap
                    xx, yy = torch.meshgrid(torch.arange(mask_shape), torch.arange(mask_shape))
                    xx = xx.float()
                    yy = yy.float()
                    gaussian = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
                    heatmap = torch.max(heatmap, gaussian)

            masks_all.append(masks.unsqueeze(0))  # [1, H, W]
            targets_all.append(heatmap.unsqueeze(0)) # [1, H, W]

            # Generate 2D distance map target
            distmap = torch.full((mask_shape, mask_shape), 1e6, dtype=torch.float32)
            for kp in keypoint_coords:
                x, y = int(kp[0]), int(kp[1]) 
                if 0 <= x < mask_shape and 0 <= y < mask_shape:
                    xx, yy = torch.meshgrid(torch.arange(mask_shape), torch.arange(mask_shape))
                    xx = xx.float()
                    yy = yy.float()
                    distance = torch.abs(xx - x) + torch.abs(yy - y) # L1 distance
                    distmap = torch.min(distmap, distance)

            # print(f"Max distmap value for {fname}: {distmap.max().item()}")

            distmaps_all.append(distmap.unsqueeze(0))  # [1, H, W]

        self.masks = masks_all
        self.targets = targets_all
        self.distmaps = distmaps_all

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return (
            self.masks[idx],
            self.targets[idx],
            self.distmaps[idx]
        )
    

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # from scipy.ndimage import gaussian_filter1d

    # Simple test
    dataset = ContourTipDataset(
        feature_dir="./data/train/features",
        label_dir="./data/train/labels",
        # pose_dir="./data/train/poses"
        use_gaussian_label=True
    )
    print(f"Dataset size: {len(dataset)}")
    features, labels = dataset[0]
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # # Apply gaussian blur to the labels and average over samples
    # # Visualize and store as a PyTorch tensor
    # prior_dist = torch.zeros_like(labels)
    # prior_dist_smoothed = torch.zeros(features.shape[0])
    # for i in range(len(dataset)):
    #     features, labels = dataset[i]
    #     labels_blurred = gaussian_filter1d(labels.numpy(), sigma=3)
    #     prior_dist += labels
    #     prior_dist_smoothed += torch.from_numpy(labels_blurred)
    # prior_dist /= len(dataset)
    # prior_dist_smoothed /= len(dataset)

    # prior_dist = 0.5 * (prior_dist + prior_dist[torch.arange(prior_dist.size(0)-1, -1, -1)]) # make symmetric
    # prior_dist_smoothed = 0.5 * (prior_dist_smoothed + prior_dist_smoothed[torch.arange(prior_dist_smoothed.size(0)-1, -1, -1)])

    # # Ensure fat tails
    # uniform_dist, alpha = 1.0, 0.3
    # prior_dist_smoothed = (1 - alpha) * prior_dist_smoothed + alpha * uniform_dist

    # plt.figure(figsize=(10, 4))
    # plt.plot(prior_dist.numpy(), label="Original Labels", alpha=0.5)
    # plt.plot(prior_dist_smoothed.numpy(), label="Blurred Labels", linewidth=2)
    # plt.legend()
    # plt.title("Example Labels and Blurred Labels")
    # plt.savefig("label_heatmap.png")

    # # Save prior distribution
    # torch.save(prior_dist_smoothed, "./models/prior_dist.pt")

    # dataset = KeypointDataset(
    #     mask_dir="./data/test/images_raw",
    #     feature_dir="./data/test/features",
    #     label_dir="./data/test/labels",
    #     mask_shape=256,
    #     sigma=1.5,
    #     device="cuda"
    # )
    # print(f"Dataset size: {len(dataset)}")

    # # Plot the heatmap target that overlays on the mask (not subplot)
    # idx_lst = [1, 2, 10, 100, 150, 200]
    # for idx in idx_lst:
    #     mask, heatmap, _ = dataset[idx]
    #     print(heatmap.max(), heatmap.min())
    #     mask_np = (mask.squeeze(0).numpy() * 255).astype(np.uint8)
    #     heatmap_np = (heatmap.squeeze(0).numpy() * 255).astype(np.uint8)
    #     heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
    #     overlay = cv2.addWeighted(cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR), 0.1, heatmap_color, 0.9, 0)

    #     # cv2.imwrite(f"mask_{idx}.png", mask_np)
    #     # cv2.imwrite(f"heatmap_{idx}.png", heatmap_color)
    #     cv2.imwrite(f"overlay_{idx}.png", overlay)

