import os
import sys
import yaml
import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

# ------------------ Path bootstrap ------------------
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

LOCAL_MODULE_DIRS = [
    REPO_ROOT,
    os.path.join(REPO_ROOT, "ParticleFilter"),
]

for p in LOCAL_MODULE_DIRS:
    if p not in sys.path:
        sys.path.insert(0, p)


# ----------------------------------------------------
# Utilities
# ----------------------------------------------------

def get_rgb(msg):
    height, width = msg.height, msg.width
    total = len(msg.data)
    pixels = height * width
    channels = total // pixels

    if total % pixels != 0:
        return None

    img = np.frombuffer(msg.data, dtype=np.uint8)
    img = img.reshape((height, width, channels))

    return img if channels == 3 else None


def swap_rb(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ----------------------------------------------------
# Main
# ----------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/super/")
    parser.add_argument("--bag", type=str, default="grasp1.bag")
    parser.add_argument("--frame_skip", type=int, default=1)
    args = parser.parse_args()

    traj_name = Path(args.bag).stem
    output_dir = Path("./data/online_videos") / traj_name
    output_dir.mkdir(parents=True, exist_ok=True)

    bagpath = Path(os.path.join(args.data_dir, args.bag))

    # ---- Load background ----
    background_path = Path(args.data_dir) / "background_2.jpg"
    if not background_path.exists():
        raise FileNotFoundError(f"Missing background.jpg at {background_path}")

    background = cv2.imread(str(background_path))
    if background is None:
        raise RuntimeError("Failed to read background.jpg")

    writer_left = None
    writer_right = None
    writer_mask = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    joint_dict = {}
    frame_idx = 0

    latest_left = None
    latest_right = None
    latest_mask = None
    latest_joint = None

    with AnyReader([bagpath]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            msg = reader.deserialize(rawdata, connection.msgtype)

            if topic == "/stereo/slave/left/image":
                latest_left = get_rgb(msg)

            elif topic == "/stereo/slave/right/image":
                latest_right = get_rgb(msg)

            elif topic == "/stereo/viewer/left/image":
                latest_mask = get_rgb(msg)

            elif topic == "/dvrk/PSM1/slave/state_joint_current":
                latest_joint = np.array(msg.position)

            if (
                latest_left is not None
                and latest_right is not None
                and latest_mask is not None
                and latest_joint is not None
            ):

                if frame_idx % args.frame_skip != 0:
                    frame_idx += 1
                    continue

                # ---- Fix color channels ----
                left_img = swap_rb(latest_left)
                right_img = swap_rb(latest_right)
                mask_img = swap_rb(latest_mask)

                height, width = left_img.shape[:2]

                # Resize background once to match resolution
                if writer_left is None:
                    bg_resized = cv2.resize(background, (width, height))

                    writer_left = cv2.VideoWriter(
                        str(output_dir / "video.mp4"),
                        fourcc,
                        30,
                        (width, height),
                    )
                    writer_right = cv2.VideoWriter(
                        str(output_dir / "video_right.mp4"),
                        fourcc,
                        30,
                        (width, height),
                    )
                    writer_mask = cv2.VideoWriter(
                        str(output_dir / "rendered.mp4"),
                        fourcc,
                        30,
                        (width, height),
                    )

                # ---- Extract red foreground from mask ----
                # mask_img is now BGR
                R, G, B = cv2.split(mask_img)
                red_foreground = ~((R > 150) & (G < 80) & (B < 80))

                # composed = bg_resized.copy()
                # composed[red_foreground] = mask_img[red_foreground]
                # Save as binary mask for simplicity
                composed = np.zeros_like(bg_resized)
                composed[red_foreground] = [255, 255, 255]

                # ---- Write videos ----
                writer_left.write(left_img)
                writer_right.write(right_img)
                writer_mask.write(composed)

                # ---- Save joint angles ----
                joint_dict[str(frame_idx)] = latest_joint.tolist()

                frame_idx += 1

                latest_left = None
                latest_right = None
                latest_mask = None
                latest_joint = None

    if writer_left is not None:
        writer_left.release()
        writer_right.release()
        writer_mask.release()

    with open(output_dir / "joint_angles.yaml", "w") as f:
        yaml.dump(joint_dict, f, sort_keys=True)

    print("Done.")
