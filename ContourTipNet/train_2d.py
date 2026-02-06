import os
import torch
from torch.utils.data import DataLoader

from model_2d import Tip2DNet, Heatmap2DLoss, KeypointMatchingLoss
from dataset import KeypointDataset


if __name__ == "__main__":
    import tqdm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask_size = 224
    batch_size = 64
    lr = 5e-5
    min_lr = 5e-6
    num_epochs = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_freq = 20
    
    mse_weight = 100.
    kpts_weight = 0.0

    best_val_loss = float("inf")
    best_model = None

    train_dataset = KeypointDataset(
        mask_dir="./data/train/images_raw",
        feature_dir="./data/train/features",
        label_dir="./data/train/labels",
        mask_shape=mask_size,
        sigma=1.5,
        device="cuda"
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = KeypointDataset(
        mask_dir="./data/val/images_raw",
        feature_dir="./data/val/features",
        label_dir="./data/val/labels",
        mask_shape=mask_size,
        sigma=1.5,
        device="cuda"
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Tip2DNet(mask_size=mask_size, pretrained=False, use_attention=False).to(device)
    # pretrained_modelname = "resnet18_fpn_transformer_model.pth"
    # model.load_state_dict(torch.load("./models/" + pretrained_modelname, map_location=device))
    mse_criterion = Heatmap2DLoss()
    kpts_criterion = KeypointMatchingLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=min_lr
    )

    train_loss_history = []
    val_loss_history = []

    # If consecutive validations do not improve, stop training
    val_tol = 3
    val_counter = 0

    pbar = tqdm.tqdm(range(num_epochs), desc="Training", ncols=100)
    for epoch in pbar:
        model.train()
        total_mse_loss = 0.0
        total_kpts_loss = 0.0

        for masks, heatmaps, distmaps in train_loader:
            masks = masks.to(device)  # [B, 1, H, W]
            heatmaps = heatmaps.to(device)  # [B, 1, H, W]
            distmaps = distmaps.to(device)  # [B, 1, H, W]

            pred = model(masks)        # [B, 1, H, W]

            mse_loss= mse_criterion(pred, heatmaps)
            kpts_loss = kpts_criterion(pred, distmaps)

            loss = mse_weight * mse_loss + kpts_weight * kpts_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mse_loss += mse_loss.item()
            total_kpts_loss += kpts_loss.item()

        avg_mse_loss = total_mse_loss / len(train_loader)
        avg_kpts_loss = total_kpts_loss / len(train_loader)
        train_loss_history.append(mse_weight * avg_mse_loss + kpts_weight * avg_kpts_loss)
        scheduler.step()
        
        pbar.set_postfix({"MSE Loss": f"{avg_mse_loss:.6f}", "KPTS Loss": f"{avg_kpts_loss:.6f}"})

        if (epoch + 1) % val_freq == 0:
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for masks, heatmaps, distmaps in val_loader:
                    masks = masks.to(device)
                    heatmaps = heatmaps.to(device)
                    distmaps = distmaps.to(device)

                    pred = model(masks)

                    loss = mse_weight * mse_criterion(pred, heatmaps) + kpts_weight * kpts_criterion(pred, distmaps)

                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model.state_dict()

                os.makedirs("./models", exist_ok=True)
                torch.save(best_model, "./models/cnn_model_best_val.pth")

                val_counter = 0
            else:
                val_counter += 1
                if val_counter >= val_tol:
                    print(f"Validation loss did not improve for {val_tol} consecutive validations. Stopping training.")
                    break

            os.makedirs("./models", exist_ok=True)
            torch.save(model.state_dict(), "./models/cnn_model.pth")

    # Plot log loss curves and save
    import math

    train_loss_history = [math.log(l) for l in train_loss_history]
    val_loss_history = [math.log(l) for l in val_loss_history]

    plt.figure()
    plt.plot(train_loss_history, label="Train Loss")
    val_idx = [(i + 1) * val_freq - 1 for i in range(len(val_loss_history))]
    plt.plot(val_idx, val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.savefig("./training_loss_curve.png")