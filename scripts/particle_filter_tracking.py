import os
import sys

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

from core.StereoCamera import StereoCamera
from core.RobotLink import *
from core.StereoCamera import *
from core.ParticleFilter import *
from core.probability_functions import *
from core.utils import *
import matplotlib.pyplot as plt

from diffcali.models.CtRNet import CtRNet
import nvdiffrast.torch as dr

import argparse
import yaml
import numpy as np
import time

from pathlib import Path
import cv2
import torch
import kornia

from diffcali.utils.skeleton_visualizer import RealTimeVideoWriter, SkeletonVisualizer


"""
python scripts/particle_filter_tracking.py --bag_idx bag1
python scripts/particle_filter_tracking.py --bag_idx bag2
python scripts/particle_filter_tracking.py --bag_idx bag3
python scripts/particle_filter_tracking.py --bag_idx bag4
python scripts/particle_filter_tracking.py --bag_idx bag5
python scripts/particle_filter_tracking.py --bag_idx bag6
"""


def parseCtRNetArgs():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.use_gpu = True
    args.trained_on_multi_gpus = False

    # Camera intrinsics
    # [ 1.02588223e+03,  0.0,  1.67919017e+02,
    #    0.0, 1.02588223e+03, 2.34152707e+02,
    #    0., 0., 1. ]

    # Setting for our custom data
    args.height = 480
    args.width = 640
    args.fx, args.fy, args.px, args.py = 1025.88223, 1025.88223, 167.919017, 234.152707

    args.scale = 1.0

    # scale the camera parameters
    args.width = int(args.width * args.scale)
    args.height = int(args.height * args.scale)
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Particle Filter Tracking for dVRK")
    parser.add_argument("--data_dir", type=str, default="./data/custom/", help="Directory containing the data files")
    parser.add_argument("--bag_idx", type=str, default="bag1", help="Name of the bag to process (e.g., bag1, bag2, etc.)")
    parser.add_argument("--sample_number", type=int, default=1000, help="Number of particles to sample for the particle filter")
    args = parser.parse_args()

    # Load data
    psm_arm = RobotLink(os.path.join(args.data_dir, "LND.json"))
    cam = StereoCamera(os.path.join(args.data_dir, "camera_calibration.yaml"), rectify=True)

    f = open(os.path.join(args.data_dir, 'handeye.yaml'), 'r')
    hand_eye_data = yaml.load(f, Loader=yaml.FullLoader)

    cam_T_b = np.eye(4)
    cam_T_b[:-1, -1] = np.array(hand_eye_data['PSM1_tvec'])/1000.0
    cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(hand_eye_data['PSM1_rvec'])

    # Load the video and joint angle readings
    bag_dir = os.path.join(args.data_dir, args.bag_idx)
    video_path_left = os.path.join(bag_dir, "left.mp4")
    video_path_right = os.path.join(bag_dir, "right.mp4")
    cap_left = cv2.VideoCapture(video_path_left)
    cap_right = cv2.VideoCapture(video_path_right)

    # ---- Real-time faithful recording setup ----
    save_video = True
    out_fps = 30.0  # common, compatible constant FPS for saved file
    if not os.path.exists("./videos/"):
        os.makedirs("./videos/")
    out_path = os.path.join(f"./videos/pf_{args.bag_idx}_realtime_demo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # We'll write at original (pre-resize-back) display size: frame_shape_orig
    # But we don't know it until we read the first frame. So: lazy init later.
    rt_writer = None

    joint_angles_path = os.path.join(bag_dir, "joint_angles.yaml")
    with open(joint_angles_path, 'r') as f:
        joint_angle_data = yaml.load(f, Loader=yaml.FullLoader)
        joint_angles_lst = [joint_angle_data[f"{i}"] for i in range(len(joint_angle_data))]
    joint_angles_np = np.array(joint_angles_lst)

    assert len(joint_angles_np) <= int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT)), "Number of joint angle readings must be less than the number of video frames"
    assert int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT)) == int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT)), "Left and right videos must have the same number of frames"
    print(f"Loaded {len(joint_angles_np)} joint angle readings and {int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))} video frames")


    ctrnet_args = parseCtRNetArgs()
    ctrnet_args.use_nvdiffrast = True

    model = CtRNet(ctrnet_args)

    mesh_dir = "urdfs/dVRK/meshes"
    mesh_files = [
        f"{mesh_dir}/low_res_shaft_multi_cylinder.ply",
        f"{mesh_dir}/low_res_logo_low_res_1.ply",
        f"{mesh_dir}/low_res_jawright_lowres.ply",
        f"{mesh_dir}/low_res_jawleft_lowres.ply",
    ]

    robot_renderer = model.setup_robot_renderer(mesh_files, downscale_factor=1)
    robot_renderer.set_mesh_visibility([True, True, True, True])

    # Specify camera intrinsics and keypoints
    intr = torch.tensor(
        [
            [ctrnet_args.fx, 0, ctrnet_args.px], 
            [0, ctrnet_args.fy, ctrnet_args.py], 
            [0, 0, 1]
        ],
        device="cuda",
        dtype=torch.float32,
    )

    # if args.use_contour_tip_net:
    #     tip_length = 0.0096 # instead of 0.009
    # else:
    #     tip_length = 0.009
    tip_length = 0.0096 # Set the tip length (distance from the last joint to the tip) to 9.6mm, which is more accurate according to the keypoint prompts
    p_local1 = (
        torch.tensor([0.0, 0.0004, tip_length]) 
        .to(torch.float32)
        .to(model.device)
    )
    p_local2 = (
        torch.tensor([0.0, -0.0004, tip_length])
        .to(torch.float32)
        .to(model.device)
    )
    # Initialize the skeleton visualizer
    skeleton_visualizer = SkeletonVisualizer(model, ctrnet_args, args, intr, p_local1, p_local2, thickness=5)

    # Define particle filter parameters
    pf = ParticleFilter(
        num_states=6,
        initialDistributionFunc=sampleNormalDistribution,
        motionModelFunc=additiveGaussianNoise,
        obsModelFunc=pointFeatureObs,
        num_particles=args.sample_number
    )

    initialize = True
    time_lst = []
    pose_lst = []
    joint_lst = []

    # Process each frame
    for frame_idx in range(len(joint_angles_np)):
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        frame_shape_orig = frame_left.shape[1], frame_left.shape[0]  # (width, height)

        if save_video and rt_writer is None:
            rt_writer = RealTimeVideoWriter(
                path=out_path,
                fourcc=fourcc,
                fps=out_fps,
                frame_size=frame_shape_orig  # writing the final displayed resolution
            )

        if not ret_left or not ret_right:
            print(f"End of video reached at frame {frame_idx}")
            break

        start_time = time.time()
        # Run particle filter tracking

        left_img, right_img = frame_left.copy(), frame_right.copy()
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        new_joint_angles = joint_angles_np[frame_idx].copy()

        detected_keypoints_l, left_img  = segmentColorAndGetKeyPoints(left_img,  draw_contours=True)
        detected_keypoints_r, right_img = segmentColorAndGetKeyPoints(right_img, draw_contours=True)

        psm_arm.updateJointAngles(new_joint_angles)

        if initialize:
            initialize=False
            pf.initializeFilter(std=np.array([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2]))
        else:
            pf.predictionStep(std=np.array([2.5e-5, 2.5e-5, 2.5e-5, 1e-4, 1e-4, 1e-4]))
            # pf.predictionStep(std=np.array([5e-5, 5e-5, 5e-5, 5e-4, 5e-4, 5e-4])*5)

        if len(detected_keypoints_l) == 0:
            detected_keypoints_l = np.empty((0, 2))
        if len(detected_keypoints_r) == 0:
            detected_keypoints_r = np.empty((0, 2))

        pf.updateStep(
            point_detections=(detected_keypoints_l,detected_keypoints_r),
            robot_arm=psm_arm,
            cam=cam,
            cam_T_b=cam_T_b,
            joint_angle_readings=new_joint_angles,
            gamma=0.15
        )

        correction_estimation = pf.getMeanParticle()

        T = poseToMatrix(correction_estimation)
        psm_arm.updateJointAngles(new_joint_angles)

        end_time = time.time()
        
        # Extract pose and joint angle estimates for evaluation
        T_4 = np.dot(np.dot(cam_T_b, T), psm_arm.baseToJointT[3]) # Get pose matrix of frame 4
        R, t_vec = T_4[:3, :3], T_4[:3, 3]
        R_ = torch.from_numpy(R).float().cuda()
        T_ = torch.from_numpy(t_vec).float().cuda()
        axis_angle = kornia.geometry.conversions.rotation_matrix_to_axis_angle(R_.unsqueeze(0)).squeeze(0) # Convert rotation matrix to axis-angle representation
        pose_vec = torch.cat([axis_angle, T_], dim=0)
        pose_lst.append(pose_vec)

        joint_angles = torch.from_numpy(new_joint_angles)[-3:].float().cuda()
        joint_angles[-1] /= 2.0
        joint_angles = torch.cat([joint_angles, joint_angles[-1:]], dim=0) # duplicate the last joint angle for the gripper
        joint_lst.append(joint_angles)

        time_lst.append(end_time - start_time)

        # img_list=projectSkeleton(
        #     psm_arm.getSkeletonPoints(),
        #     np.dot(cam_T_b, T),
        #     [left_img, right_img],
        #     cam.projectPoints
        # )

        # display = img_list[0] # Only show left image for better visibility of keypoints and skeleton
        display = left_img.copy()
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB) # Convert to RGB for correct color display in OpenCV
        
        display = skeleton_visualizer.plot_skeleton_overlay(display, pose_vec, joint_angles)

        if len(time_lst) > 10:
            # Compute FPS and display it on the image
            avg_time = sum(time_lst) / len(time_lst)
            fps = 1.0 / avg_time if avg_time > 0 else float('inf')
            # cv2.putText(display, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(
                display,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            # print(time_lst[-10:])

        cv2.imshow("Tracking Result (Left)", display)

        # Add elapsed wall-clock time overlay (proof it's real-time)
        if rt_writer is not None and rt_writer.t0 is not None:
            elapsed = time.perf_counter() - rt_writer.t0
            cv2.putText(
                display,
                f"Wall-clock time: {elapsed:7.3f}s",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        # Real-time faithful write (duplicates frames if slow)
        if rt_writer is not None:
            rt_writer.write_realtime(display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if rt_writer is not None:
        rt_writer.release()
        print(f"Saved real-time faithful video to: {out_path}")

    # Stack pose and joint angle estimates into tensors and save for evaluation
    pose_tensor = torch.stack(pose_lst)
    joint_tensor = torch.stack(joint_lst)

    time_tensor = torch.tensor(time_lst)
    print(f"Average tracking time per frame: {time_tensor[10:].mean().item():.4f} seconds, FPS: {1.0 / time_tensor[10:].mean().item():.2f}")
    
    target_dir = "./pf_tracking_results/"
    os.makedirs(target_dir, exist_ok=True)
    torch.save({"cTr": pose_tensor, "joint_angles": joint_tensor, "time": time_tensor}, os.path.join(target_dir, f"pf_{args.bag_idx}_tracking_results.pt"))