import os
import torch
import pandas as pd

POSE_DIR = "./pose_results"


def parse_dual_arm_filename(filename):
    """
    Format:
    BI_MANUAL_{data_label}.{searcher}.{online_iters}.{joint_str}.
    {pts_loss_str}.{kpts_det_str}.{app_loss_str}.
    {filter_str}.{option_label}.{sep_label}.pth
    """

    name = os.path.basename(filename).replace(".pth", "")

    if not name.startswith("BI_MANUAL_"):
        return None

    name = name.replace("BI_MANUAL_", "")
    parts = name.split(".")

    if len(parts) != 10:
        return None

    return dict(
        data_label=parts[0],
        searcher=parts[1],
        online_iters=int(parts[2]),
        joint_label=parts[3],
        pts_loss=parts[4],
        kpts_det=parts[5],
        app_loss=parts[6],
        filter=parts[7],
        loss_option=parts[8],       # joint / sep
        separation=parts[9],        # softsep / hardsep
    )


def collect_results():
    rows = []

    for file in os.listdir(POSE_DIR):
        if not file.endswith(".pth"):
            continue

        full_path = os.path.join(POSE_DIR, file)
        meta = parse_dual_arm_filename(file)

        if meta is None:
            continue

        data = torch.load(full_path, map_location="cpu")

        total_runtime = sum(data["time"])
        online_iters = meta["online_iters"]

        row = {
            **meta,
            "total_runtime": total_runtime,
            "runtime_per_iter": total_runtime / online_iters,
        }

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = collect_results()
    df = df.sort_values(
        by=["data_label", "searcher", "loss_option", "separation"]
    )
    print(df)

    df.to_csv("dual_arm_summary.csv", index=False)
