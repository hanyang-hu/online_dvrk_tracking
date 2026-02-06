import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2


"""
python process_surgpose_video.py --traj_id 000000
python process_surgpose_video.py --traj_id 000001
python process_surgpose_video.py --traj_id 000002
python process_surgpose_video.py --traj_id 000003
python process_surgpose_video.py --traj_id 000004
python process_surgpose_video.py --traj_id 000005
python process_surgpose_video.py --traj_id 000006
python process_surgpose_video.py --traj_id 000007
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process SurgPose video into frames")
    parser.add_argument('--traj_id', type=str, default="000000", help='Trajectory ID to process')
    args = parser.parse_args()

    data_dir = "./datasets/"
    video_path = data_dir + f"surgpose_raw/{args.traj_id}/regular/left_video.mp4"
    output_dir = data_dir + f"surgpose_traj/{args.traj_id}/"

    os.makedirs(output_dir, exist_ok=True)

    target_frames = 1001
    # target_width, target_height = 640, 480
    original_height = 1400, 986

    # # Compute crop width to match target aspect ratio
    # crop_width = int(original_height * target_width / target_height)  # ≈ 1314
    # crop_height = original_height
    # crop_x, crop_y = 0, 0  # left-top crop

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample 1001 frames evenly
    indices = [int(i * total_frames / (target_frames - 1)) for i in range(target_frames)]

    frame_id = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in indices:
            # # Crop horizontal space, keep full vertical
            # cropped = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

            # # Resize to target (uniform scaling)
            # resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            # cv2.imwrite(os.path.join(output_dir, f"{saved_count:05d}.png"), resized)
            cv2.imwrite(os.path.join(output_dir, f"{saved_count:05d}.png"), frame)

            saved_count += 1
        frame_id += 1

    cap.release()
    print(f"Saved {saved_count} frames to {output_dir}")

