import os
import torch
import argparse
import numpy as np
import glob
import cv2
import time
import imageio

from cotracker.predictor import CoTrackerOnlinePredictor


def get_reference_keypoints_auto(
        ref_img_path, num_keypoints=2, ref_img=None,
        quality_level=0.1, min_distance=5, block_size=15
    ):
    # Read data
    if ref_img_path is None and ref_img is None:
        raise ValueError("Either ref_img_path or ref_mask must be provided.")
    cv_img = ref_img if ref_img_path is None else cv2.imread(ref_img_path)

    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    ref_img = cv2.inRange(cv_img, np.ones(3) * 128, np.ones(3) * 255) 
    binary_mask = ref_img.astype(np.uint8)
    region_mask = (binary_mask > 0).astype(np.uint8)
    max_corners = num_keypoints        # Maximum number of corners to find
    
    corners = cv2.goodFeaturesToTrack(
        binary_mask,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        mask=region_mask
    )

    output_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Draw the corners
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(output_image, (int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-1)  # Red circles for corners

    ref_keypoints = corners
     
    return ref_keypoints.squeeze(1).tolist() 


if __name__ == "__main__":
    # Read the frames and relevant data from the data directory.
    parser = argparse.ArgumentParser(description="CoTracker Test Script")
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task/data folder')
    parser.add_argument('--use_sam2_mask', action='store_true', help='Whether to use SAM2 masks')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Mask overlay ratio if using SAM2 masks')
    parser.add_argument('--use_enhanced_mask', action='store_true', help='Whether to use enhanced masks')
    args = parser.parse_args()
    task_name = args.task_name

    data_dir = os.path.join("./data/consecutive_prediction", task_name)
    
    frame_start = 0
    frame_end = len(os.listdir(data_dir))

    ref_mask_path_lst = []
    masks_lst = []
    frames_lst = []

    for i in range(frame_start, frame_end):
        frame_dir = os.path.join(data_dir, f"{i}")

        # Find the mask
        mask_lst = glob.glob(os.path.join(frame_dir, "*.png"))
        if len(mask_lst) == 0:
            raise ValueError(f"No mask found in {frame_dir}")
        if len(mask_lst) > 1:
            raise ValueError(f"Multiple masks found in {frame_dir}")

        mask_path = mask_lst[0]
        # frame = cv2.imread(frame_path)
        XXXX = mask_path.split("/")[-1].split(".")[0][1:]

        # Get mask filename
        ref_mask_path = os.path.join(frame_dir, "0" + XXXX + ".png")

        ref_mask_path_lst.append(ref_mask_path)
        masks_lst.append(cv2.imread(mask_path))

        frame_lst = glob.glob(os.path.join(frame_dir, "*.jpg"))
        if len(frame_lst) == 0:
            raise ValueError(f"No frame found in {frame_dir}")
        if len(frame_lst) > 1:
            raise ValueError(f"Multiple frames found in {frame_dir}")
        frame_path = frame_lst[0]
        frame = cv2.imread(frame_path)
        frames_lst.append(frame)

    if args.use_enhanced_mask:
        # overlay enhanced masks on frames to enhance tracking
        for idx in range(len(frames_lst)):
            img = cv2.cvtColor(masks_lst[idx], cv2.COLOR_BGR2GRAY)
            # kernel = np.ones((5,5),np.uint8)
            # gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
            # # stronger erosion to make mask thinner; adjust size and iterations as needed
            # kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            # erosion = cv2.erode(img, kernel_erode, iterations=1)
            # masks_lst[idx] = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)

            keypoints = get_reference_keypoints_auto(
                ref_img_path=None,
                num_keypoints=3,
                ref_img=img,
            )
            # Draw dots for keypoints
            for kp in keypoints:
                cv2.circle(masks_lst[idx], (int(kp[0]), int(kp[1])), radius=3, color=(0, 0, 0), thickness=-1) 



    alpha = 1 - args.mask_ratio if args.use_sam2_mask else 1.
    # overlay masks on frames to enhance tracking
    for idx in range(len(frames_lst)):
        frame = cv2.addWeighted(frames_lst[idx], alpha, masks_lst[idx], 1-alpha, 0)

        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        frame_sharp = cv2.filter2D(frame, -1, kernel)

        frames_lst[idx] = frame_sharp


    # Obtain the reference keypoints as the query input
    queries = torch.tensor(get_reference_keypoints_auto(ref_img_path=ref_mask_path_lst[0], num_keypoints=5)).cuda().float()
    # Add the time dimension to match the format [time, x coord, y coord]
    # For initialization, time is always 0
    queries = torch.cat([
        torch.zeros(queries.shape[0], 1, device=queries.device),
        queries
    ], dim=1)[:, :3]  # Ensure shape is (N, 3): [0, x, y]


    model = CoTrackerOnlinePredictor(
        checkpoint=os.path.join(
            './cotracker/checkpoints/scaled_online.pth'
        )
    ).cuda()
    # model.support_grid_size = 10

   # Online tracking
    window_frames = []
    is_first_step = True

    def _process_step(window_frames, is_first_step, queries):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :])
            )
            .float()
            .permute(0, 3, 1, 2)[None]
            .cuda()
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries,
            # add_support_grid=True
        )

    time_lst = []

    # Use torch.cuda.synchronize for accurate timing and measure total time
    torch.cuda.synchronize()
    total_start_time = time.time()

    for i, frame in enumerate(frames_lst):
        torch.cuda.synchronize()
        start_time = time.time()

        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                queries = queries[None] if is_first_step else None,
            )
            is_first_step = False
        window_frames.append(frame)

        torch.cuda.synchronize()
        end_time = time.time()
        time_lst.append((end_time - start_time) * 1000)

    torch.cuda.synchronize()
    total_end_time = time.time()
    total_time_ms = (total_end_time - total_start_time) * 1000
    avg_time_per_frame = total_time_ms / len(frames_lst)

    print(np.array(time_lst))

    print(f"Average time per frame (torch synchronized): {np.mean(time_lst):.2f} ms")
    print(f"Total time: {total_time_ms:.2f} ms, Average per frame: {avg_time_per_frame:.2f} ms")

    # Process the final frames if video length is not a multiple of model.step
    if len(window_frames) % model.step != 1:
        pred_tracks, pred_visibility = _process_step(
            window_frames[-(i % model.step) - model.step - 1 :],
            is_first_step,
            queries,
        )

    print("Tracks are computed")

    # Extract keypoints from pred_tracks
    # pred_tracks: (1, T, N, 2) where T = number of frames, N = number of keypoints
    kpts_tracks = []
    pred_tracks_np = pred_tracks.squeeze(0).cpu().numpy()  # (T, N, 2)
    num_kpts = pred_tracks_np.shape[1]
    num_frames = pred_tracks_np.shape[0]
    for kpt_idx in range(num_kpts):
        kpts_tracks.append([pred_tracks_np[t, kpt_idx].tolist() for t in range(num_frames)])

    # Visualize and save as GIF (do not save images)
    frames = []
    # Process visibility into a (T, N) boolean array if available
    try:
        pred_visibility_np = pred_visibility.squeeze(0).cpu().numpy()
        if pred_visibility_np.ndim == 3 and pred_visibility_np.shape[-1] == 1:
            pred_visibility_np = pred_visibility_np[..., 0]
        # If visibility is probabilistic, threshold at 0.5
        if pred_visibility_np.dtype == np.bool_:
            pred_vis_bool = pred_visibility_np
        else:
            pred_vis_bool = pred_visibility_np > 0.5
        # Ensure shape is (T, N)
        pred_vis_bool = np.asarray(pred_vis_bool)
    except Exception:
        # Fallback: assume everything visible
        pred_vis_bool = np.ones((num_frames, num_kpts), dtype=bool)

    for t_idx in range(num_frames):
        frame = frames_lst[t_idx].copy()

        # Add title on the left upper part
        bag_number = ''.join(filter(str.isdigit, task_name))
        title_text = f"Bag {bag_number}" if bag_number else f"Task {task_name}"
        cv2.putText(
            frame,
            title_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Plot tracked keypoints: past frames as lines, current frame as circle
        for kpt_idx, track_kpts in enumerate(kpts_tracks):
            # Draw lines for the previous 10 frames
            for i in range(max(0, t_idx - 10), t_idx):
                if i + 1 < len(track_kpts):
                    pt1 = tuple(map(int, track_kpts[i]))
                    pt2 = tuple(map(int, track_kpts[i + 1]))
                    # Determine visibility for this segment: require both endpoints visible to draw green
                    vis_i = False
                    vis_ip1 = False
                    if i < pred_vis_bool.shape[0] and kpt_idx < pred_vis_bool.shape[1]:
                        vis_i = bool(pred_vis_bool[i, kpt_idx])
                    if (i + 1) < pred_vis_bool.shape[0] and kpt_idx < pred_vis_bool.shape[1]:
                        vis_ip1 = bool(pred_vis_bool[i + 1, kpt_idx])
                    seg_visible = vis_i and vis_ip1
                    color = (0, 255, 0) if seg_visible else (0, 0, 255)
                    cv2.line(frame, pt1, pt2, color=color, thickness=2)
            # Draw circle for the current frame (green if visible, red otherwise)
            if t_idx < len(track_kpts):
                kp = track_kpts[t_idx]
                vis_now = True
                if t_idx < pred_vis_bool.shape[0] and kpt_idx < pred_vis_bool.shape[1]:
                    vis_now = bool(pred_vis_bool[t_idx, kpt_idx])
                color = (0, 255, 0) if vis_now else (0, 0, 255)
                cv2.circle(frame, (int(kp[0]), int(kp[1])), radius=4, color=color, thickness=-1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame_rgb)

    # Save as GIF in the current folder
    gif_path = f"./cotracker_{task_name}.gif"
    imageio.mimsave(gif_path, frames, duration=0.05)



"""
python scripts/sequential_tracing.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 1 --use_pts_loss True --use_cyd_loss False --use_nvdiffrast --track_kpts True --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 10 --final_iters 300 --use_prev_joint_angles True --use_weighting_mask False --difficulty "rw1"
python scripts/video_generator.py --bag_id 1
rm tracking/*.png

python scripts/sequential_tracing.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 1 --use_pts_loss True --use_cyd_loss False --use_nvdiffrast --track_kpts True --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 10 --final_iters 300 --use_prev_joint_angles True --use_weighting_mask False --difficulty "rw5"
python scripts/video_generator.py --bag_id 5
rm tracking/*.png

python scripts/sequential_tracing.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 1 --use_pts_loss True --use_cyd_loss False --use_nvdiffrast --track_kpts True --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 10 --final_iters 300 --use_prev_joint_angles True --use_weighting_mask False --difficulty "rw6"
python scripts/video_generator.py --bag_id 6
rm tracking/*.png

python scripts/sequential_tracing.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 1 --use_pts_loss True --use_cyd_loss False --use_nvdiffrast --track_kpts True --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 10 --final_iters 300 --use_prev_joint_angles True --use_weighting_mask False --difficulty "rw7"
python scripts/video_generator.py --bag_id 7
rm tracking/*.png

python scripts/sequential_tracing.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 1 --use_pts_loss True --use_cyd_loss False --use_nvdiffrast --track_kpts True --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 10 --final_iters 300 --use_prev_joint_angles True --use_weighting_mask False --difficulty "rw8"
python scripts/video_generator.py --bag_id 8
rm tracking/*.png

python scripts/sequential_tracing.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 1 --use_pts_loss True --use_cyd_loss False --use_nvdiffrast --track_kpts True --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 10 --final_iters 300 --use_prev_joint_angles True --use_weighting_mask False --difficulty "rw9"
python scripts/video_generator.py --bag_id 9
rm tracking/*.png

python scripts/sequential_tracing.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 1 --use_pts_loss True --use_cyd_loss False --use_nvdiffrast --track_kpts True --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 10 --final_iters 300 --use_prev_joint_angles True --use_weighting_mask False --difficulty "rw10"
python scripts/video_generator.py --bag_id 10
rm tracking/*.png

python scripts/sequential_tracing.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 1 --use_pts_loss True --use_cyd_loss False --use_nvdiffrast --track_kpts True --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 10 --final_iters 300 --use_prev_joint_angles True --use_weighting_mask False --difficulty "rw11"
python scripts/video_generator.py --bag_id 11
rm tracking/*.png

python scripts/sequential_tracing.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 1 --use_pts_loss True --use_cyd_loss False --use_nvdiffrast --track_kpts True --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 10 --final_iters 300 --use_prev_joint_angles True --use_weighting_mask False --difficulty "rw12"
python scripts/video_generator.py --bag_id 12
rm tracking/*.png

python scripts/sequential_tracing.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 1 --use_pts_loss True --use_cyd_loss False --use_nvdiffrast --track_kpts True --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 10 --final_iters 300 --use_prev_joint_angles True --use_weighting_mask False --difficulty "rw14"
python scripts/video_generator.py --bag_id 14
rm tracking/*.png

python scripts/sequential_tracing.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 1 --use_pts_loss True --use_cyd_loss False --use_nvdiffrast --track_kpts True --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 10 --final_iters 300 --use_prev_joint_angles True --use_weighting_mask False --difficulty "rw15"
python scripts/video_generator.py --bag_id 15
rm tracking/*.png

"""