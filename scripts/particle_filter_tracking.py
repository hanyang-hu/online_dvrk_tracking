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

import argparse
import yaml
import numpy as np
import time
from rosbags.highlevel import AnyReader
from pathlib import Path
import cv2


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Particle Filter Tracking")
    parser.add_argument("--data_dir", type=str, default="./data/super/", help="Directory containing the data files")
    parser.add_argument("--bag", type=str, default="grasp1.bag", help="Path to the ROS bag file")
    parser.add_argument("--downsample_scale", type=int, default=4, help="Factor by which to downsample the images for display")
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

    # Define particle filter parameters
    pf = ParticleFilter(
        num_states=6,
        initialDistributionFunc=sampleNormalDistribution,
        motionModelFunc=additiveGaussianNoise,
        obsModelFunc=pointFeatureObs,
        num_particles=1000
    )

    initialize=True

    latest_left=None
    latest_right=None
    latest_joint=None
    time_lst = []

    with AnyReader([bagpath]) as reader:
        for connection,timestamp,rawdata in reader.messages():
            topic=connection.topic
            msg=reader.deserialize(rawdata,connection.msgtype)

            if topic=="/stereo/slave/left/image":
                latest_left=get_rgb(msg)
            elif topic=="/stereo/slave/right/image":
                latest_right=get_rgb(msg)
            elif topic=="/dvrk/PSM1/slave/state_joint_current":
                latest_joint=np.array(msg.position)

            if latest_left is None or latest_right is None or latest_joint is None:
                continue

            start_t=time.time()

            left_img, right_img = cam.processImage(latest_left.copy(), latest_right.copy())

            detected_keypoints_l, left_img  = segmentColorAndGetKeyPoints(left_img,  draw_contours=True)
            detected_keypoints_r, right_img = segmentColorAndGetKeyPoints(right_img, draw_contours=True)

            new_joint_angles = latest_joint

            psm_arm.updateJointAngles(new_joint_angles)

            if initialize:
                initialize=False
                pf.initializeFilter(std=np.array([1e-3,1e-3,1e-3,1e-2,1e-2,1e-2]))
            else:
                pf.predictionStep(std=np.array([2.5e-5,2.5e-5,2.5e-5,1e-4,1e-4,1e-4]))

            pf.updateStep(point_detections=(detected_keypoints_l,detected_keypoints_r),
                        robot_arm=psm_arm,
                        cam=cam,
                        cam_T_b=cam_T_b,
                        joint_angle_readings=new_joint_angles,
                        gamma=0.15)

            correction_estimation=pf.getMeanParticle()

            T=poseToMatrix(correction_estimation)
            psm_arm.updateJointAngles(new_joint_angles)

            end_t = time.time()
            
            time_lst.append(end_t-start_t)

            img_list=projectSkeleton(psm_arm.getSkeletonPoints(),
                                    np.dot(cam_T_b,T),
                                    [left_img,right_img],
                                    cam.projectPoints)

            # display=np.hstack((img_list[0],img_list[1]))
            display = left_img # img_list[0] # Only show left image for better visibility of keypoints and skeleton
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB) # Convert to RGB for correct color display in OpenCV

            if len(time_lst) > 10:
                # Compute FPS and display it on the image
                avg_time = sum(time_lst) / len(time_lst)
                fps = 1.0 / avg_time if avg_time > 0 else float('inf')
                cv2.putText(display, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.imshow("Tracking Result (Left | Right)", display)

            if cv2.waitKey(1)&0xFF==27:
                break

    cv2.destroyAllWindows()

