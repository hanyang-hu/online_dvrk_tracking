import os
import sys
import numpy as np
import torch
import cv2
import yaml
import argparse

# ------------------ Path bootstrap ------------------
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

LOCAL_MODULE_DIRS = [
    REPO_ROOT,
    os.path.join(REPO_ROOT, "SurgicalSAM2"),
]

for p in LOCAL_MODULE_DIRS:
    if p not in sys.path:
        sys.path.insert(0, p)

from sam2.build_sam import build_sam2_camera_predictor
import shutil

# ------------------ Globals (interactive) ------------------
tooltips = []        # display space
prompt_points = []  # display space
prompt_labels = []  # 1 = FG, 0 = BG

base_frame = None
vis_frame = None
predictor = None
mask_ds = None

sx = sy = None
orig_w = orig_h = None


def get_ds_prompts():
    pts = np.array(
        [[int(x * sx), int(y * sy)] for x, y in prompt_points],
        dtype=np.float32,
    )
    lbs = np.array(prompt_labels, dtype=np.int64)
    return pts, lbs


def draw_overlay(mask_ds_local=None):
    vis_frame[:] = base_frame

    # Overlay SAM mask (resize back to display space)
    if mask_ds_local is not None:
        mask_full = cv2.resize(
            (mask_ds_local.astype(np.uint8) * 255),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        color = cv2.applyColorMap(mask_full, cv2.COLORMAP_JET)
        vis_frame[:] = cv2.addWeighted(vis_frame, 0.7, color, 0.3, 0)

    # Tool tips (blue)
    for i, (x, y) in enumerate(tooltips):
        cv2.circle(vis_frame, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(
            vis_frame, str(i),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )

    # Prompts
    for (x, y), l in zip(prompt_points, prompt_labels):
        c = (0, 255, 0) if l == 1 else (0, 0, 255)
        cv2.circle(vis_frame, (x, y), 5, c, -1)

    # UI text
    h = vis_frame.shape[0]
    instructions = [
        "Left click: Tool tip",
        "SHIFT + Left click: Foreground prompt",
        "CTRL + Left click: Background prompt",
        "ENTER: Save & exit | r: Reset | q / ESC: Quit",
    ]
    for i, text in enumerate(instructions):
        cv2.putText(
            vis_frame,
            text,
            (10, h - 90 + 25 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )


def mouse_callback(event, x, y, flags, param):
    global mask_ds

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # CTRL → background prompt
    if flags & cv2.EVENT_FLAG_CTRLKEY:
        prompt_points.append([x, y])
        prompt_labels.append(0)

    # SHIFT → foreground prompt
    elif flags & cv2.EVENT_FLAG_SHIFTKEY:
        prompt_points.append([x, y])
        prompt_labels.append(1)

    # plain click → tool tip
    else:
        tooltips.append([x, y])
        draw_overlay(mask_ds)
        return

    # Update SAM mask
    pts_ds, lbs = get_ds_prompts()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, _, mask_logits = predictor.add_new_points(
            frame_idx=0,
            obj_id=0,
            points=pts_ds,
            labels=lbs,
        )

    mask_ds = (mask_logits.squeeze() > 0).cpu().numpy()
    draw_overlay(mask_ds)


"""
Annotate videos from "000000" to "000007" and from "000030" to "000033" in the SurgPose dataset:
python scripts/video_annotator.py --idx 000000 --machine_label PSM3 --annotate_sequence
python scripts/video_annotator.py --idx 000001 --machine_label PSM3 --annotate_sequence
python scripts/video_annotator.py --idx 000002 --machine_label PSM3 --annotate_sequence
python scripts/video_annotator.py --idx 000003 --machine_label PSM3 --annotate_sequence
python scripts/video_annotator.py --idx 000004 --machine_label PSM3 --annotate_sequence
python scripts/video_annotator.py --idx 000005 --machine_label PSM3 --annotate_sequence
python scripts/video_annotator.py --idx 000006 --machine_label PSM3 --annotate_sequence
python scripts/video_annotator.py --idx 000007 --machine_label PSM3 --annotate_sequence
python scripts/video_annotator.py --idx 000030 --machine_label PSM3 --annotate_sequence
python scripts/video_annotator.py --idx 000031 --machine_label PSM3 --annotate_sequence
python scripts/video_annotator.py --idx 000032 --machine_label PSM3 --annotate_sequence
python scripts/video_annotator.py --idx 000033 --machine_label PSM3 --annotate_sequence

python scripts/video_annotator.py --idx 000000 --machine_label PSM1 --annotate_sequence
python scripts/video_annotator.py --idx 000001 --machine_label PSM1 --annotate_sequence
python scripts/video_annotator.py --idx 000002 --machine_label PSM1 --annotate_sequence
python scripts/video_annotator.py --idx 000003 --machine_label PSM1 --annotate_sequence
python scripts/video_annotator.py --idx 000004 --machine_label PSM1 --annotate_sequence
python scripts/video_annotator.py --idx 000005 --machine_label PSM1 --annotate_sequence
python scripts/video_annotator.py --idx 000006 --machine_label PSM1 --annotate_sequence 
python scripts/video_annotator.py --idx 000007 --machine_label PSM1 --annotate_sequence
python scripts/video_annotator.py --idx 000030 --machine_label PSM1 --annotate_sequence
python scripts/video_annotator.py --idx 000031 --machine_label PSM1 --annotate_sequence
python scripts/video_annotator.py --idx 000032 --machine_label PSM1 --annotate_sequence
python scripts/video_annotator.py --idx 000033 --machine_label PSM1 --annotate_sequence
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/surgpose_raw/")
    parser.add_argument("--target_path", type=str, default="data/surgpose/")
    parser.add_argument("--video_path", type=str, default="data/online_videos/")
    parser.add_argument("--idx", type=str, default="000000")
    parser.add_argument("--machine_label", type=str, default="PSM3")
    parser.add_argument("--sam2_checkpoint", type=str,
                        default="./SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth")
    parser.add_argument("--model_cfg", type=str,
                        default="./configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--downsample_factor", type=int, default=2, help="Downsample factor for tracker input.")
    parser.add_argument("--num_frames", type=int, default=1000, help="Number of frames to track (default: all frames)")
    parser.add_argument("--annotate_sequence", action="store_true", help="Whether to annotate the entire sequence (default: only first frame)")
    parser.add_argument("--cam_side", type=str, default="left", choices=["left", "right"], help="Camera side to annotate (default: left)")
    args = parser.parse_args()


    # Prepare target directory and copy video
    target_video_dir = os.path.join(args.video_path, args.idx)
    os.makedirs(target_video_dir, exist_ok=True)

    cam_side = args.cam_side
    source_video_path = os.path.join(
        args.data_path, args.idx, "regular", f"{cam_side}_video.mp4"
    )
    target_video_path = os.path.join(target_video_dir, "video.mp4")

    if not os.path.exists(target_video_path):
        os.system(f"cp {source_video_path} {target_video_path}")

    keypoint_path = os.path.join(
        target_video_dir, f"{args.machine_label}_keypoints.txt"
    )
    prompts_path = os.path.join(
        target_video_dir, f"{args.machine_label}_prompts.txt"
    )
    init_mask_path = os.path.join(
        target_video_dir, f"{args.machine_label}_ref_mask.png"
    )

    # Compute width and height for SAM input by reading the first frame
    cap = cv2.VideoCapture(target_video_path)
    ret, frame_full = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read video")
    orig_h, orig_w = frame_full.shape[:2]
    args.width = orig_w // args.downsample_factor
    args.height = orig_h // args.downsample_factor

    # Load video and prepare display frame
    cap = cv2.VideoCapture(target_video_path)
    ret, frame_full = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read video")

    orig_h, orig_w = frame_full.shape[:2]
    sx = args.width / orig_w
    sy = args.height / orig_h

    frame_ds = cv2.resize(frame_full, (args.width, args.height))
    base_frame = frame_full.copy()
    vis_frame = frame_full.copy()

    # Initialize SAM predictor with the first frame
    predictor = build_sam2_camera_predictor(
        args.model_cfg,
        args.sam2_checkpoint,
        vos_optimized=True,
    )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.load_first_frame(frame_ds)

    # Annotation loop
    WINDOW = f"Initialize annotations ({args.machine_label})"
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, mouse_callback)

    draw_overlay()

    print(f"[INFO] Annotate first frame ({args.machine_label})")
    print("[INFO] Left: tool tip | SHIFT: FG | CTRL: BG | ENTER: save")
    print("[INFO] PSM 3 is on the left, PSM 1 is on the right.")

    while True:
        cv2.imshow(WINDOW, vis_frame)
        key = cv2.waitKey(10) & 0xFF

        if key == 13:      # ENTER
            break
        elif key == ord("r"):
            tooltips.clear()
            prompt_points.clear()
            prompt_labels.clear()
            mask_ds = None
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.reset_state()
                predictor.load_first_frame(frame_ds)
            draw_overlay()
        elif key in [ord("q"), 27]:
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    # Save outputs
    tooltips_ds = np.array(
        [[int(x * sx), int(y * sy)] for x, y in tooltips],
        dtype=np.int32,
    )

    prompts_ds = np.array(
        [[int(x * sx), int(y * sy), l]
         for (x, y), l in zip(prompt_points, prompt_labels)],
        dtype=np.int32,
    )

    np.savetxt(keypoint_path, tooltips_ds, fmt="%d")
    np.savetxt(prompts_path, prompts_ds, fmt="%d")

    if mask_ds is not None:
        cv2.imwrite(init_mask_path, mask_ds.astype(np.uint8) * 255)

    print(f"[INFO] Saved keypoints → {keypoint_path}")
    print(f"[INFO] Saved prompts   → {prompts_path}")
    print(f"[INFO] Saved mask      → {init_mask_path}")

    if not args.annotate_sequence:
        print("[INFO] Sequence annotation skipped. Exiting.")
        sys.exit(0)

    # Prepare data for sequential tracking
    target_data_dir = os.path.join(args.target_path, args.idx, args.machine_label)
    os.makedirs(target_data_dir, exist_ok=True)

    # Read ground truth gripper angles and joint angles
    gripper_source_path = os.path.join(args.data_path, "gripper_angle.yaml")
    joint_source_path = os.path.join(args.data_path, args.idx, "api_jp_data.yaml")

    kpts_idx_dict = {
        'PSM3': [11, 12],
        'PSM1': [4, 5],
    }
    keypoints_source_path = os.path.join(args.data_path, args.idx, f"keypoints_{cam_side}.yaml")

    with open(gripper_source_path, "r") as f:
        gripper_data = yaml.safe_load(f)[args.idx][args.machine_label]

    with open(joint_source_path, "r") as f:
        joint_angle_data_yaml = yaml.safe_load(f)
        joint_angle_data = [
            joint_angle_data_yaml[str(i)][args.machine_label] for i in range(len(joint_angle_data_yaml))
        ]

    with open(keypoints_source_path, "r") as f:
        keypoints_data_yaml = yaml.safe_load(f)
        # Extract the relevant keypoints for the current machine, if not both are available, use None
        keypoints_data = []
        for i in range(len(keypoints_data_yaml)):
            kpts_i = keypoints_data_yaml[i]
            # if all(k in kpts_i for k in kpts_idx_dict[args.machine_label]):
            if kpts_idx_dict[args.machine_label][0] in kpts_i and kpts_idx_dict[args.machine_label][1] in kpts_i and kpts_i[kpts_idx_dict[args.machine_label][0]] is not None and kpts_i[kpts_idx_dict[args.machine_label][1]] is not None:
                # print(kpts_idx_dict[args.machine_label])
                kpts_i_selected = [[kpts_i[k][0] // args.downsample_factor, kpts_i[k][1] // args.downsample_factor] for k in kpts_idx_dict[args.machine_label]]
                keypoints_data.append(kpts_i_selected)
            else:
                keypoints_data.append(None)

    print("[INFO] Starting SAM2 tracking on full video...")

    cap = cv2.VideoCapture(target_video_path)
    frame_idx = 0
    NUM_FRAMES = args.num_frames if args.num_frames > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while frame_idx < NUM_FRAMES:
            ret, frame_full = cap.read()
            if not ret:
                break

            # downsample frame
            frame_ds = cv2.resize(frame_full, (args.width, args.height))

            # first frame already initialized
            if frame_idx == 0:
                out_mask_logits = torch.from_numpy(mask_ds).unsqueeze(0).unsqueeze(0).float().to(predictor.device)
            else:
                _, out_mask_logits = predictor.track(frame_ds)

            mask = (out_mask_logits.squeeze() > 0).cpu().numpy().astype(np.uint8) * 255

            # save format: target_data_dir/{i}/{i:05d}.png
            frame_dir = os.path.join(target_data_dir, f"{frame_idx}")
            os.makedirs(frame_dir, exist_ok=True)

            out_path = os.path.join(frame_dir, f"{frame_idx:05d}.png")
            cv2.imwrite(out_path, mask)

            # Save the joint angles, jaw angles, and keypoints for this frame
            if gripper_data is not None:
                gripper_out_path = os.path.join(frame_dir, f"jaw_{frame_idx:04d}.npy")
                np.save(gripper_out_path, gripper_data[frame_idx])
            if joint_angle_data is not None:
                joint_out_path = os.path.join(frame_dir, f"joint_{frame_idx:04d}.npy")
                np.save(joint_out_path, joint_angle_data[frame_idx])
            if keypoints_data is not None and keypoints_data[frame_idx] is not None:
                keypoints_out_path = os.path.join(frame_dir, f"keypoints_{frame_idx:04d}.npy")
                np.save(keypoints_out_path, keypoints_data[frame_idx])
            if frame_idx == 0:
                # Save the clicked keypoints for the first frame instead
                keypoints_out_path = os.path.join(frame_dir, f"keypoints_{frame_idx:04d}.npy")
                np.save(keypoints_out_path, tooltips_ds)

            if frame_idx % 10 == 0:
                overlay = frame_ds.copy()
                color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(overlay, 0.7, color, 0.3, 0)

                # Plot keypoints if available
                if keypoints_data is not None and keypoints_data[frame_idx] is not None:
                    for kpts in keypoints_data[frame_idx]:
                        cv2.circle(overlay, (int(kpts[0]), int(kpts[1])), 5, (0, 255, 255), -1)

                cv2.imshow("Tracking", overlay)
                cv2.waitKey(1)

            frame_idx += 1

    cv2.destroyAllWindows()
    cap.release()

    print(f"[INFO] Tracking complete. Saved {frame_idx} frames to:")
    print(f"       {target_data_dir}")
