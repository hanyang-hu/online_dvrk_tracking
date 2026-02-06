import os
import torch
import numpy as np
import cv2
import yaml

from sam2.build_sam import build_sam2_video_predictor


"""
python generate_surgpose_mask.py --traj_id 000000 --machine_label PSM3
python generate_surgpose_mask.py --traj_id 000001 --machine_label PSM3
python generate_surgpose_mask.py --traj_id 000002 --machine_label PSM3
python generate_surgpose_mask.py --traj_id 000003 --machine_label PSM3
python generate_surgpose_mask.py --traj_id 000004 --machine_label PSM3
python generate_surgpose_mask.py --traj_id 000005 --machine_label PSM3
python generate_surgpose_mask.py --traj_id 000006 --machine_label PSM3
python generate_surgpose_mask.py --traj_id 000007 --machine_label PSM3

python generate_surgpose_mask.py --traj_id 000000 --machine_label PSM1
python generate_surgpose_mask.py --traj_id 000001 --machine_label PSM1
python generate_surgpose_mask.py --traj_id 000002 --machine_label PSM1
python generate_surgpose_mask.py --traj_id 000003 --machine_label PSM1
python generate_surgpose_mask.py --traj_id 000004 --machine_label PSM1
python generate_surgpose_mask.py --traj_id 000005 --machine_label PSM1
python generate_surgpose_mask.py --traj_id 000006 --machine_label PSM1
python generate_surgpose_mask.py --traj_id 000007 --machine_label PSM1

cd ../
mkdir -p ./bbox_dvrk_calibration/data/surgpose
cp -r ./Surgical-SAM-2/datasets/surgpose/. ./bbox_dvrk_calibration/data/surgpose/ 
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate SurgPose mask using Surgical SAM 2")
    parser.add_argument('--traj_id', type=str, default="000000", help='Trajectory ID to process')
    parser.add_argument('--machine_label', type=str, default="PSM3", help='SurgPose machine label (e.g., PSM1, PSM3)')

    args = parser.parse_args()

    data_dir = "./datasets/"
    video_dir = data_dir + f"surgpose_traj/{args.traj_id}/"
    bag_idx = int(args.traj_id)
    output_dir = data_dir + f"surgpose/bag{bag_idx}_{args.machine_label}/"

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    os.environ['CUDA_VISIBLE_DEVICES']="1"

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # sam2_checkpoint = "./model_weights/sam2_hiera_small.pt"
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_s_endo18.pth"

    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", '.png']
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)

    predictor.reset_state(inference_state)

    # Read ground truth keypoints from yaml file
    kpts_dir = data_dir + f"surgpose_raw/{args.traj_id}/keypoints_left.yaml"

    with open(kpts_dir, 'r') as f:
        kpts_init = yaml.safe_load(f)[0]
    
    # Extract values from dictionary and convert to numpy array 
    if args.machine_label == "PSM1":
        kpts_indices = [1, 2, 3, 4, 5, 6, 7]
    else:
        kpts_indices = [8, 9, 10, 11, 12, 13, 14]
    # Some frames may have missing keypoints, so we filter them out
    kpts_init_np = np.array([kpts_init[i] for i in kpts_indices if i in kpts_init], dtype=np.float32)  # [K, 2] in (x,y)

    # Use ground truth keypoints as point prompts for the first frame
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = kpts_init_np
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1 for _ in range(len(points))], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # # Plot initial raw image with the selected points
    # initial_frame = cv2.imread(os.path.join(video_dir, f"{ann_frame_idx:05d}.png"))
    # for (x, y) in points:
    #     cv2.circle(initial_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    # os.makedirs(output_dir, exist_ok=True)
    # cv2.imwrite(os.path.join(output_dir, f"{ann_frame_idx:05d}.png"), initial_frame)

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Dummy data paths
    optimized_joint_angles_path = os.path.join(data_dir + "dummy", "optimized_joint_angles.npy")
    jaw_path = os.path.join(data_dir + "dummy", "jaw_0000.npy")
    joint_path = os.path.join(data_dir + "dummy", "joint_0000.npy")
    optimized_ctr_path = os.path.join(data_dir + "dummy", "optimized_ctr.npy")

    # Read the data
    joint_angle_data = np.load(optimized_joint_angles_path)
    jaw_data = np.load(jaw_path)
    joint_data = np.load(joint_path)
    optimized_ctr_data = np.load(optimized_ctr_path)

    bag_name = f"bag{bag_idx}_{args.machine_label}"

    # Save masks as data for tracking
    vis_frame_stride = 1
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            out_mask = (out_mask.astype(np.uint8) * 255)[0]
            out_mask = cv2.resize(out_mask, (out_mask.shape[1]//2, out_mask.shape[0]//2), interpolation=cv2.INTER_NEAREST) # Resize to half resolution

            os.makedirs(data_dir + f"surgpose/bag{bag_idx}_{args.machine_label}/" + f"{out_frame_idx}/", exist_ok=True)
            mask_filename = data_dir + f"surgpose/bag{bag_idx}_{args.machine_label}/" + f"{out_frame_idx}/{out_frame_idx:05d}.png"

            # Copy and rename the dummy data files
            dummy_dir = data_dir + f"surgpose/bag{bag_idx}_{args.machine_label}/" + f"{out_frame_idx}/"
            optimized_joint_angles_dest = os.path.join(dummy_dir, "optimized_joint_angles.npy")
            jaw_dest = os.path.join(dummy_dir, f"jaw_{out_frame_idx:04d}.npy")
            joint_dest = os.path.join(dummy_dir, f"joint_{out_frame_idx:04d}.npy")
            optimized_ctr_dest = os.path.join(dummy_dir, "optimized_ctr.npy")

            # Store the ground truth tip location for the first frame
            if out_frame_idx == 0:
                # Extract values from dictionary and convert to numpy array 
                if args.machine_label == "PSM1":
                    tip_indices = [4, 5]
                else:
                    tip_indices = [11, 12]
                # Some frames may have missing keypoints, so we filter them out
                tip_init_np = np.array([kpts_init[i] for i in tip_indices if i in kpts_init], dtype=np.float32)  # [2,] in (x,y)
                tip_init_np = tip_init_np / 2.0  # Resize to half resolution
                np.save(os.path.join(dummy_dir, f"keypoints_{out_frame_idx:04d}.npy"), tip_init_np)

            # Save the numpy files
            np.save(optimized_joint_angles_dest, joint_angle_data)
            np.save(jaw_dest, jaw_data)
            np.save(joint_dest, joint_data)
            np.save(optimized_ctr_dest, optimized_ctr_data)

            cv2.imwrite(mask_filename, out_mask)

            # print(f"Saved mask to {mask_filename}")
        


            