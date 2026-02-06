import torch
import numpy
import cv2
import numpy as np

from model_2d import Tip2DNet
from dataset import KeypointDataset


if __name__ == "__main__":
    test_dataset = KeypointDataset(
        mask_dir="./data/test/images_raw",
        feature_dir="./data/test/features",
        label_dir="./data/test/labels",
        mask_shape=224,
        sigma=1.5,
        device="cuda"
    )

    # Predict tip location using ContourTipNet
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = Tip2DNet(mask_size=224, use_attention=False).cuda()
        model_name = "./models/cnn_model.pth"
        # model_name = "./models/resnet18_fpn_model.pth"
        model.load_state_dict(torch.load(model_name, map_location=device))
        model.eval()

        test_idx = [0, 10, 20, 30, 40, 50]
        for idx in test_idx:
            features, target_heatmap, _ = test_dataset[idx]
            features = features.unsqueeze(0).to(device)  # [1, C, H, W]
            pred_heatmap = model(features)  # [1, 1, H, W]
            pred_heatmap = pred_heatmap.squeeze().cpu().numpy()  # [H, W]
            target_heatmap = target_heatmap.cpu().numpy()  # [1，H, W]

            # Save image, predicted heatmap and target heatmap as BGR images together (by concatenation)
            features_np = (features.squeeze().unsqueeze(-1).cpu().numpy()* 255).astype(np.uint8)  # [H, W]
            features_np = cv2.cvtColor(features_np, cv2.COLOR_GRAY2BGR)  # [H, W, 3]
            pred_heatmap_np = (pred_heatmap * 255).astype(np.uint8)  # [H, W]
            pred_heatmap_np = cv2.applyColorMap(pred_heatmap_np, cv2.COLORMAP_JET) # [H, W, 3]
            target_heatmap_np = (target_heatmap.squeeze() * 255).astype(np.uint8)  # [H, W]
            target_heatmap_np = cv2.applyColorMap(target_heatmap_np, cv2.COLORMAP_JET) # [H, W, 3]

            concat_img = np.concatenate([features_np, pred_heatmap_np, target_heatmap_np], axis=1)
            cv2.imwrite(f"./GIFs/test_{idx}.png", concat_img)