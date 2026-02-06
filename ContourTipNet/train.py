import os
import torch
from torch.utils.data import DataLoader

from model import ContourTipNet, EnergyEMDLoss, MSELoss
from dataset import ContourTipDataset


if __name__ == "__main__":
    import tqdm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature_dim = 6 
    batch_size = 64
    lr = 5e-4
    min_lr = 5e-5
    num_epochs = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_freq = 20

    use_gaussian_label = True

    best_val_loss = float("inf")
    best_model = None

    train_dataset = ContourTipDataset(
        feature_dir="./data/train/features",
        label_dir="./data/train/labels",
        use_gaussian_label=use_gaussian_label
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ContourTipDataset(
        feature_dir="./data/val/features",
        label_dir="./data/val/labels",
        use_gaussian_label=use_gaussian_label
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ContourTipNet(feature_dim=feature_dim, max_len=200).to(device)
    criterion = EnergyEMDLoss() # Use EMD loss regardless of label type
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
        total_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device)  # [B, N, F]
            labels = labels.to(device)      # [B, N]

            pred = model(features)        # [B, N]

            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_loss)
        scheduler.step()
        
        pbar.set_postfix({"Train Loss": f"{avg_loss:.6f}"})

        if (epoch + 1) % val_freq == 0:
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(device)
                    labels = labels.to(device)

                    pred = model(features)

                    loss = criterion(pred, labels)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model.state_dict()

                os.makedirs("./models", exist_ok=True)
                torch.save(best_model, "./models/ctn_model_best_val.pth")

                val_counter = 0
            else:
                val_counter += 1
                if val_counter >= val_tol:
                    print(f"Validation loss did not improve for {val_tol} consecutive validations. Stopping training.")
                    break

            os.makedirs("./models", exist_ok=True)
            torch.save(model.state_dict(), "./models/ctn_model.pth")

            # Plot the heatmap bias learned
            heatmap_bias = model.heatmap_bias().detach().cpu()
            plt.plot(heatmap_bias.numpy(), label="Learned Bias", linewidth=2)
            plt.title("Learned Heatmap Bias")
            plt.xlabel("Keypoint Index")
            plt.ylabel("Bias Value")
            plt.savefig("./heatmap_bias.png")
            plt.close()

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