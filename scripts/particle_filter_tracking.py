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
from rosbags.highlevel import AnyReader
from pathlib import Path
import cv2
import torch
from collections import deque


def get_rgb(msg):
    height, width = msg.height, msg.width
    total = len(msg.data)
    pixels = height * width
    channels = total // pixels

    if total % pixels != 0:
        print("Skipping frame: invalid buffer size")
        return None
    
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, channels))
    
    if channels == 3:
        return img
    else:
        return None
    

def transform_mesh(cameras, mesh, R, T, args):
    """
    Transform the mesh from world space to clip space
    Modified from https://github.com/NVlabs/nvdiffrast/issues/148#issuecomment-2090054967
    """
    # world to view transform
    verts = mesh.verts_padded()  #  (B, N_v, 3)
    verts_view = cameras.get_world_to_view_transform(R=R, T=T).transform_points(verts)  # (B, N_v, 3)
    verts_view[...,  :3] *= -1 # due to PyTorch3D camera coordinate conventions
    verts_view_home = torch.cat([verts_view, torch.ones_like(verts_view[..., [0]])], axis=-1) # (B, N_v, 4)

    # projection
    fx, fy = cameras.focal_length[0]
    px, py = cameras.principal_point[0]
    height, width = cameras.image_size[0]
    near, far = args.znear, args.zfar
    A = (2 * fx) / width
    B = (2 * fy) / height
    C = (width - 2 * px) / width
    D = (height - 2 * py) / height
    E = (near + far) / (near - far)
    F = (2 * near * far) / (near - far)
    t_mtx = projectionMatrix = torch.tensor(
        [
            [A, 0, C, 0],
            [0, B, D, 0],
            [0, 0, E, F],
            [0, 0, -1, 0]
        ]
    ).to(verts.device)
    verts_clip = torch.matmul(verts_view_home, t_mtx.transpose(0, 1))

    faces_clip = mesh.faces_padded().to(torch.int32)

    return verts_clip, faces_clip


def render(glctx, pos, pos_idx, resolution: [int, int], antialiasing=False, col=None):
    """
    Silhouette rendering pipeline based on NvDiffRast
    if col is None, render silhouette mask
    otherwise (col is (1, N_v, 3)), render colored image (three channels)
    """
    # Create color attributes
    if col is None:
        col = torch.ones_like(pos[..., :1], dtype=torch.float32) # (B, N_v, 1)
    col_idx = pos_idx

    # Render the mesh
    rast_out, _ = dr.rasterize(glctx, pos, pos_idx, resolution=resolution)
    color   , _ = dr.interpolate(col, rast_out, col_idx)
    if antialiasing:
        color = dr.antialias(color, rast_out, pos, pos_idx)
    return color.squeeze(-1) # (B, H, W)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Particle Filter Tracking")
    parser.add_argument("--data_dir", type=str, default="./data/super/", help="Directory containing the data files")
    parser.add_argument("--bag", type=str, default="grasp1.bag", help="Path to the ROS bag file")
    parser.add_argument("--downsample_scale", type=int, default=4, help="Factor by which to downsample the images for display")
    parser.add_argument("--overlay_skeleton", action='store_true', help="Whether to overlay the robot skeleton on the images")
    parser.add_argument("--overlay_mask", action='store_true', help="Whether to overlay the rendered mask on the images")
    parser.add_argument("--overlay_camera_frame", action='store_true', help="Whether to overlay the camera frame axes on the images")
    parser.add_argument("--sample_number", type=int, default=1000, help="Number of particles to use in the particle filter")
    parser.add_argument("--joint_idx", type=int, default=3, help="Index of the joint to visualize the camera frame for (0-based)")
    parser.add_argument("--frame_skip", type=int, default=20, help="Number of frames to skip between each processing step")
    args = parser.parse_args()

    # Load data
    psm_arm = RobotLink(os.path.join(args.data_dir, "LND.json"))
    cam = StereoCamera(os.path.join(args.data_dir, "camera_calibration.yaml"), rectify=True)

    f = open(os.path.join(args.data_dir, 'handeye.yaml'), 'r')
    hand_eye_data = yaml.load(f, Loader=yaml.FullLoader)

    cam_T_b = np.eye(4)
    cam_T_b[:-1, -1] = np.array(hand_eye_data['PSM1_tvec'])/1000.0
    cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(hand_eye_data['PSM1_rvec'])

    bagpath = Path(os.path.join(args.data_dir, args.bag))

    # Use camera intrinsics (cam.K1, cam.img_size) to update ctrnet args
    args.mesh_dir = "urdfs/dVRK/meshes"
    args.arm = "psm2"

    args.use_gpu = True
    args.trained_on_multi_gpus = False

    args.height = cam.img_size[1]
    args.width = cam.img_size[0]
    args.fx, args.fy, args.px, args.py = cam.K1[0,0], cam.K1[1,1], cam.K1[0,2], cam.K1[1,2]

    # clip space parameters
    args.znear = 1e-3
    args.zfar = 1e9

    args.use_nvdiffrast = True # do not use nvdiffrast in CtRNet

    # Build NvDiffRast model
    model = CtRNet(args)
    mesh_files = [
        f"{args.mesh_dir}/shaft_multi_cylinder.ply",
        f"{args.mesh_dir}/logo_low_res_1.ply",
        f"{args.mesh_dir}/jawright_lowres.ply",
        f"{args.mesh_dir}/jawleft_lowres.ply",
    ]

    robot_renderer = model.setup_robot_renderer(mesh_files)
    robot_renderer.set_mesh_visibility([True, True, True, True])

    glctx = dr.RasterizeCudaContext() # CUDA context (OpenGL is not available in my WSL)
    resolution = (args.height, args.width)

    # Define particle filter parameters
    pf = ParticleFilter(
        num_states=6,
        initialDistributionFunc=sampleNormalDistribution,
        motionModelFunc=additiveGaussianNoise,
        obsModelFunc=pointFeatureObs,
        num_particles=args.sample_number
    )

    initialize=True

    latest_left, latest_right, latest_joint, latest_mask = None, None, None, None
    time_lst = []
    frame_cnt = 0
    
    frame_buffer = deque(maxlen=5000)

    print(f"Loading data from {bagpath} ...")

    with AnyReader([bagpath]) as reader:
        latest_left = None
        latest_right = None
        latest_joint = None
        latest_mask = None
        frame_cnt = 0

        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            msg = reader.deserialize(rawdata, connection.msgtype)

            if topic == "/stereo/slave/left/image":
                latest_left = get_rgb(msg)

            elif topic == "/stereo/slave/right/image":
                latest_right = get_rgb(msg)

            elif topic == "/dvrk/PSM1/slave/state_joint_current":
                latest_joint = np.array(msg.position)

            elif topic == "/stereo/viewer/left/image":
                latest_mask = get_rgb(msg)

            # Once we have a full synchronized set → push frame
            if latest_left is not None and latest_right is not None and latest_joint is not None:

                frame_cnt += 1
                if frame_cnt % args.frame_skip != 0:
                    continue

                frame_buffer.append((
                    latest_left.copy(),
                    latest_right.copy(),
                    latest_joint.copy(),
                    latest_mask.copy() if latest_mask is not None else None
                ))

    print(f"Finished loading data. Starting tracking with {len(frame_buffer)} frames in buffer...")

    for latest_left, latest_right, latest_joint, latest_mask in frame_buffer:

        start_t = time.time()

        left_img, right_img = cam.processImage(latest_left.copy(), latest_right.copy())

        detected_keypoints_l, left_img  = segmentColorAndGetKeyPoints(left_img,  draw_contours=True)
        detected_keypoints_r, right_img = segmentColorAndGetKeyPoints(right_img, draw_contours=True)

        new_joint_angles = latest_joint

        psm_arm.updateJointAngles(new_joint_angles)

        if initialize:
            initialize=False
            pf.initializeFilter(std=np.array([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2]))
        else:
            # pf.predictionStep(std=np.array([2.5e-5, 2.5e-5, 2.5e-5, 1e-4, 1e-4, 1e-4]))
            pf.predictionStep(std=np.array([5e-5, 5e-5, 5e-5, 5e-4, 5e-4, 5e-4]))

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

        end_t = time.time()
        
        time_lst.append(end_t - start_t)

        if args.overlay_skeleton:
            img_list=projectSkeleton(
                psm_arm.getSkeletonPoints(),
                np.dot(cam_T_b, T),
                [left_img, right_img],
                cam.projectPoints
            )
        else:
            img_list = [left_img, right_img]

        display = img_list[0] # Only show left image for better visibility of keypoints and skeleton
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB) # Convert to RGB for correct color display in OpenCV

        if args.overlay_mask:
            # mask_resized=cv2.resize(seg_mask,(display.shape[1],display.shape[0]),interpolation=cv2.INTER_NEAREST)

            # # apply colormap expects 8-bit single channel (0–255)
            # mask_color=cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)

            # # black-out background in colormap
            # mask_color[mask_resized==0]=0

            # Use NvDiffRast to render the mask from the estimated pose
            joint_angles = torch.from_numpy(latest_joint)[-3:].float().cuda()
            joint_angles[-1] /= 2.0
            joint_angles = torch.cat([joint_angles, joint_angles[-1:]], dim=0) # duplicate the last joint angle for the gripper
            model.get_joint_angles(joint_angles)
            robot_mesh = robot_renderer.get_robot_mesh(joint_angles)

            render_T = np.dot(np.dot(cam_T_b, T), psm_arm.baseToJointT[args.joint_idx])
            R, t_vec = render_T[:3, :3].T, render_T[:3, 3]
            R_batched = torch.from_numpy(R).float().cuda().unsqueeze(0)
            T_batched = torch.from_numpy(t_vec).float().cuda().unsqueeze(0)

            negative_mask = T_batched[:, -1] < 0  #flip where negative_mask is True
            T_batched_ = T_batched.clone()
            T_batched_[negative_mask] = -T_batched_[negative_mask]
            R_batched_ = R_batched.clone()
            R_batched_[negative_mask] = -R_batched_[negative_mask]
            pos, pos_idx = transform_mesh(
                cameras=robot_renderer.cameras, mesh=robot_mesh.extend(1),
                R=R_batched_, T=T_batched_, args=args
            ) # project the batched meshes in the clip 
            
            rendered_mask = render(glctx, pos, pos_idx[0], resolution)[0] # shape (H, W)
            mask_color = cv2.applyColorMap((rendered_mask.cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
            mask_color[rendered_mask.cpu().numpy()==0] = 0 # black-out background in colormap

            alpha=0.3
            display=cv2.addWeighted(display,1-alpha, mask_color ,alpha,0)

        if len(time_lst) > 10:
            # Compute FPS and display it on the image
            avg_time = sum(time_lst) / len(time_lst)
            fps = 1.0 / avg_time if avg_time > 0 else float('inf')
            cv2.putText(display, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # print(time_lst[-10:])

        # Plot the camera frame
        if args.overlay_camera_frame:
            axis_length = 0.02
            joint_idx = args.joint_idx

            # Build axis in local joint frame (same format as skeleton)
            points_local = np.array([
                [0, 0, 0],
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, axis_length]
            ])

            # TRANSPOSE like skeleton does (3xN)
            points_local = np.transpose(points_local)

            # Add homogeneous row like skeleton
            points_local = np.concatenate(
                (points_local, np.ones((1, points_local.shape[1])))
            )   # Now 4xN

            # Apply baseToJointT exactly like calculateSkeletonPoints
            points_base = np.dot(psm_arm.baseToJointT[joint_idx], points_local)

            # Now apply camera transform exactly like projectSkeleton
            points_cam = np.dot(np.dot(cam_T_b, T), points_base)

            # Drop last row exactly like projectSkeleton
            points_cam = np.transpose(points_cam[:-1, :])

            # Now project
            proj_l, proj_r = cam.projectPoints(points_cam)

            origin = tuple(proj_l[0].astype(int))
            x_axis = tuple(proj_l[1].astype(int))
            y_axis = tuple(proj_l[2].astype(int))
            z_axis = tuple(proj_l[3].astype(int))

            cv2.line(display, origin, x_axis, (0,0,255), 2)
            cv2.line(display, origin, y_axis, (0,255,0), 2)
            cv2.line(display, origin, z_axis, (255,0,0), 2)

        cv2.imshow("Tracking Result (Left | Right)", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
