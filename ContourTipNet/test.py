import torch
import os

from model import ContourTipNet, EnergyEMDLoss
from dataset import ContourTipDataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = ContourTipDataset(
        feature_dir="./data/test/features",
        label_dir="./data/test/labels"
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ContourTipNet(feature_dim=6).to(device)
    model.load_state_dict(torch.load("./models/ctn_model.pth", map_location=device))

    model.eval()

    criterion = EnergyEMDLoss()
    
    with torch.no_grad():
        # Visualize the predictions using matplotlib 
        plt.figure(figsize=(10, 5))
        for i, (features, labels) in enumerate(dataloader):
            # Only visualize predictions as 1-D heatmap
            features = features.to(device)  # [B, N, F]
            labels = labels.to(device)      # [B, N]
            pred = model(features)          # [B, N]
            loss = criterion(pred, labels)
            print(f"Sample {i}: Test Loss = {loss.item():.6f}")
            # Estimate number of keypoints
            num_keypoints = labels.sum().item()
            print(f"  Number of keypoints (ground truth): {num_keypoints:.1f}")
            num_keypoints_pred = pred.sum().item()
            print(f"  Number of keypoints (predicted): {num_keypoints_pred:.1f}")
            plt.clf()
            plt.bar(range(len(labels[0].cpu().numpy())), labels[0].cpu().numpy(), alpha=0.5, label="Ground Truth", width=1.)
            plt.bar(range(len(pred[0].cpu().numpy())), pred[0].cpu().numpy(), alpha=0.5, label="Prediction", width=1.)
            plt.title(f"Sample {i} - Test Loss: {loss.item():.6f}")
            plt.xlabel("Contour Point Index")
            plt.ylabel("Heatmap Value")
            plt.legend()
            plt.pause(0.2)  # Pause to visualize
        plt.show()
            
