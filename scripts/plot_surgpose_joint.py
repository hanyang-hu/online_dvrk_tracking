import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import numpy as np

idx = "000000"
machine_name = 'PSM1'
pose_filename = f"./pose_results/bag{int(idx)}_{machine_name}_tracking_results_iters50.pth"
gt_joint_filename = f"./data/surgpose/{idx}/api_jp_data.yaml"
gt_gripper_angle_filename = "./data/surgpose/gripper_angle.yaml"
frame_start = 1  

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Load estimated pose data\
    pose_data = torch.load(pose_filename)
    est_joint_angles = pose_data['joint_angles'].cpu().numpy()  # (N, 4)

    # Load ground truth joint angles
    with open(gt_joint_filename, 'r') as f:
        joint_data = yaml.safe_load(f)
    gt_joint_angles_lst = []
    for i in range(frame_start, len(joint_data)):
        joint_i = joint_data[str(i)][machine_name]
        gt_joint_angles_lst.append(joint_i)
    gt_joint_angles = np.stack(gt_joint_angles_lst, axis=0)  # (N, 6)

    # Load ground truth gripper angles
    with open(gt_gripper_angle_filename, 'r') as f:
        gt_gripper_angles_lst = yaml.safe_load(f)[idx][machine_name]
    gt_gripper_angles = np.stack(gt_gripper_angles_lst, axis=0)  # (N,)

    # Plot the joint angles ground truth and estimated in 10 separate subplots
    # Start with the estimated wrist pitch, wrist yaw, jaw1, jaw2
    joint_names = ['Wrist Pitch', 'Wrist Yaw', 'Jaw1', 'Jaw2']
    # plt.figure(figsize=(12, 8))
    # for i in range(4):
    #     plt.subplot(5, 2, i + 1)
    #     if i == 2 or i == 3:
    #         # plot gripper angles
    #         plt.plot(gt_gripper_angles[:], label='Ground Truth', color='g')

    #     plt.plot(est_joint_angles[:, i], label='Estimated', color='b')
    #     plt.title(f'{joint_names[i]} Angle')
    #     plt.xlabel('Frame')
    #     plt.ylabel('Angle (rad)')
    #     # plt.legend()
    #     plt.grid()
    # # Then plot the 6 ground truth joint angles
    # joint_names = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6']
    # for i in range(6):
    #     plt.subplot(5, 2, i + 5)
    #     plt.plot(gt_joint_angles[:, i], label='Ground Truth', color='g')
    #     if i == 4:
    #         # plot wrist pitch
    #         plt.plot(-est_joint_angles[:, 0], label='Estimated Wrist Pitch', color='b')
    #     if i == 5:
    #         # plot wrist yaw
    #         plt.plot(-est_joint_angles[:, 1], label='Estimated Wrist Yaw', color='b')
    #     plt.title(f'{joint_names[i]} Angle')
    #     plt.xlabel('Frame')
    #     plt.ylabel('Angle (rad)')
    #     # plt.legend()
    #     plt.grid()
    # Only plot the 4 estimated joint angles and the corresponding ground truth angles
    plt.figure(figsize=(10, 8))
    # Wrist Pitch
    plt.subplot(4, 1, 1)
    plt.plot(-est_joint_angles[:, 0], label='Estimated', color='b')
    plt.plot(gt_joint_angles[:, 4], label='Ground Truth', color='g')
    plt.title('Wrist Pitch Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid()
    # Wrist Yaw
    plt.subplot(4, 1, 2)
    plt.plot(-est_joint_angles[:, 1], label='Estimated', color='b')
    plt.plot(gt_joint_angles[:, 5], label='Ground Truth', color='g')
    plt.title('Wrist Yaw Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid()
    # Jaw1
    plt.subplot(4, 1, 3)
    plt.plot(2 * est_joint_angles[:, 2], label='Estimated', color='b')
    plt.plot(gt_gripper_angles[:], label='Ground Truth', color='g')
    plt.title('Jaw1 Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid()
    # Jaw2
    plt.subplot(4, 1, 4)
    plt.plot(2 * est_joint_angles[:, 3], label='Estimated', color='b')
    plt.plot(gt_gripper_angles[:], label='Ground Truth', color='g')
    plt.title('Jaw2 Angle')
    plt.xlabel('Frame')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('surgpose_joint_angles_comparison.png')
    print("Saved joint angles comparison plot to surgpose_joint_angles_comparison.png")
    plt.close()