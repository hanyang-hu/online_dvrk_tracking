import torch
from sam2.build_sam import build_sam2_camera_predictor
import cv2
import numpy as np
import time

# import os
# os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

# -------------------------
# Config
# -------------------------
sam2_checkpoint = "./checkpoints/sam2.1_hiera_s_endo18.pth"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

idx = "000000"
machine_label = "PSM3"
video_path = f"../data/online_videos/{idx}.mp4"

# -------------------------
# Build predictor
# -------------------------
predictor = build_sam2_camera_predictor(
    model_cfg,
    sam2_checkpoint,
    vos_optimized=True,
)

# -------------------------
# Video
# -------------------------
cap = cv2.VideoCapture(video_path)

# -------------------------
# Interactive state
# -------------------------
click_points = []   # [[x, y], ...]
click_labels = []   # 1 = FG, 0 = BG
first_frame = None
vis_frame = None
init_done = False
time_lst = []

# -------------------------
# Mouse callback
# -------------------------
def mouse_callback(event, x, y, flags, param):
    global click_points, click_labels, vis_frame, predictor

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # SHIFT + left click → background
    if flags & cv2.EVENT_FLAG_SHIFTKEY:
        label = 0
    else:
        label = 1

    click_points.append([x, y])
    click_labels.append(label)

    pts = np.array(click_points, dtype=np.float32)
    lbs = np.array(click_labels, dtype=np.int64)

    _, _, mask_logits = predictor.add_new_points(
        frame_idx=0,
        obj_id=0,
        points=pts,
        labels=lbs,
    )

    mask = (mask_logits.squeeze() > 0).cpu().numpy().astype(np.uint8) * 255
    color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    vis_frame[:] = cv2.addWeighted(first_frame, 0.7, color, 0.3, 0)

    for p, l in zip(click_points, click_labels):
        c = (0, 255, 0) if l == 1 else (0, 0, 255)
        cv2.circle(vis_frame, tuple(p), 5, c, -1)


# -------------------------
# Main loop
# -------------------------
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if not init_done:
            # ---------- initialization frame ----------
            predictor.load_first_frame(frame)

            first_frame = frame.copy()
            vis_frame = frame.copy()

            cv2.namedWindow("frame")
            cv2.setMouseCallback("frame", mouse_callback)

            print("[INFO] Left click: FG | SHIFT + Left click: BG")
            print("[INFO] Press ENTER to start tracking")
            print("[INFO] Press r to reset prompts")

            while True:
                cv2.imshow("frame", vis_frame)
                key = cv2.waitKey(10) & 0xFF

                if key == 13:  # ENTER
                    init_done = True
                    cv2.setMouseCallback("frame", lambda *args: None)
                    break

                elif key == ord('r'):
                    click_points.clear()
                    click_labels.clear()
                    vis_frame[:] = first_frame

                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit(0)
            
                # Add instructions on how to use the interface
                cv2.putText(
                    vis_frame,
                    "Left click: FG | SHIFT + Left click: BG | ENTER: Start | r: Reset | q: Quit",
                    (10, vis_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            continue

        # ---------- tracking ----------
        torch.cuda.synchronize()
        start_time = time.time()
        out_obj_ids, out_mask_logits = predictor.track(frame)
        torch.cuda.synchronize()
        end_time = time.time()
        time_lst.append(end_time - start_time)

        mask = (out_mask_logits.squeeze() > 0).cpu().numpy().astype(np.uint8) * 255
        color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.7, color, 0.3, 0)

        avg_time = sum(time_lst) / len(time_lst)
        fps = 1 / avg_time if avg_time > 0 else 0
        cv2.putText(
            blended,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.imshow("frame", blended)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# -------------------------
# Stats
# -------------------------
if len(time_lst) > 0:
    avg_time = sum(time_lst) / len(time_lst)
    fps = 1 / avg_time
    print(f"Average Inference Time: {avg_time * 1000:.4f} ms")
    print(f"FPS: {fps:.2f}")
