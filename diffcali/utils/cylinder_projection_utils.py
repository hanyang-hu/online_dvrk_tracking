import torch as th

import numpy as np
import cv2
import matplotlib.pyplot as plt


def projectCylinderTorch(position, direction, radius, fx, fy, cx, cy):
    # Position  = Bx3 points on the cylinder
    # Direction = Bx3 normals for each cylinder
    # radius = radius of cylinder
    # outputs e_1, e_2 which are both Bx2

    alpha1 = (
        (1 - direction[:, 0] * direction[:, 0]) * position[:, 0]
        - direction[:, 0] * direction[:, 1] * position[:, 1]
        - direction[:, 0] * direction[:, 2] * position[:, 2]
    )
    beta1 = (
        -direction[:, 0] * direction[:, 1] * position[:, 0]
        + (1 - direction[:, 1] * direction[:, 1]) * position[:, 1]
        - direction[:, 1] * direction[:, 2] * position[:, 2]
    )
    gamma1 = (
        -direction[:, 0] * direction[:, 2] * position[:, 0]
        - direction[:, 1] * direction[:, 2] * position[:, 1]
        + (1 - direction[:, 2] * direction[:, 2]) * position[:, 2]
    )
    # component of pp perpendicular to d. (p - (p*d)d)

    alpha2 = direction[:, 2] * position[:, 1] - direction[:, 1] * position[:, 2]
    beta2 = direction[:, 0] * position[:, 2] - direction[:, 2] * position[:, 0]
    gamma2 = direction[:, 1] * position[:, 0] - direction[:, 0] * position[:, 1]
    # component of pp perpendicular to d.  p (dxp).

    # C = x0*x0 + y0*y0 + z0*z0 - (a*x0 + b*y0 + c*z0)*(a*x0 + b*y0 + c*z0) - radius*radius
    C = (
        position[:, 0] * position[:, 0]
        + position[:, 1] * position[:, 1]
        + position[:, 2] * position[:, 2]
        - th.square(
            position[:, 0] * direction[:, 0]
            + position[:, 1] * direction[:, 1]
            + position[:, 2] * direction[:, 2]
        )
        - radius * radius
    )

    C = th.clamp(C, min=1e-4)
    # if C < 0:
    #     print("Recieved C less than 0")
    #     return (-1, -1), (-1, -1)

    temp = radius / (th.sqrt(C))

    k1 = alpha1 * temp - alpha2
    k2 = beta1 * temp - beta2
    k3 = gamma1 * temp - gamma2

    # Get edges! Fu + Gv = D convert to Au + Bv = 1

    F = k1 / fx
    G = k2 / fy
    D = -k3 + F * cx + G * cy

    e_1 = th.cat(((F / D).unsqueeze(-1), (G / D).unsqueeze(-1)), dim=-1)
    k1 += 2 * alpha2
    k2 += 2 * beta2
    k3 += 2 * gamma2

    F = k1 / fx
    G = k2 / fy
    D = -k3 + F * cx + G * cy

    e_2 = th.cat(((F / D).unsqueeze(-1), (G / D).unsqueeze(-1)), dim=-1)

    return e_1, e_2


def transform_points(position, transform, intr):
    """Apply a 4x4 transformation matrix to position points (Bx3), and project to 2D image coordinates.
    Args:
        position: Tensor of shape (B, 3) with 3D points.
        transform: Tensor of shape (4, 4) representing the transformation matrix.
        intr: Tensor of shape (3, 3) representing the camera intrinsic matrix.
    Returns:
        Transformed points of shape (B, 2) in image coordinates.
    """
    # print(f"testing the shape of input {position.shape}, {transform.shape}, {intr.shape}")
    # Convert position to homogeneous coordinates by adding a 1 to each point
    position_h = th.cat(
        (
            position,
            th.ones(position.shape[0], 1, dtype=position.dtype, device=position.device),
        ),
        dim=-1,
    )  # (B, 4)

    # Apply the transformation matrix
    transformed_position_h = th.matmul(position_h, transform.T)  # (B, 4)
    # print(f"debuggging transform { transform.shape}") # [4, 1, 1]
    # Convert back to non-homogeneous coordinates (drop the last column)
    transformed_position = transformed_position_h[:, :3]  # (B, 3)
    camera_frame_coors = transformed_position
    # Project the transformed 3D points to 2D image coordinates using camera intrinsics
    # Convert to homogeneous coordinates (u, v, 1)

    projected_points_h = th.matmul(intr, transformed_position.T).T  # Shape (B, 3)

    # Normalize by z-coordinate to get (u, v) in image coordinates
    projected_points_2d = projected_points_h[:, :2] / projected_points_h[
        :, 2
    ].unsqueeze(-1)

    return projected_points_2d, camera_frame_coors


def transform_points_b(position, transform, intr):
    """Apply a 4x4 transformation matrix to position points (Bx3), and project to 2D image coordinates.
    Args:
        position: Tensor of shape (B, 3) with 3D points.
        transform: Tensor of shape (4, 4) representing the transformation matrix.  should be (B, 4, 4)
        intr: Tensor of shape (3, 3) representing the camera intrinsic matrix.
    Returns:
        Transformed points of shape (B, 2) in image coordinates.
    """
    # print(f"testing the shape of input {position.shape}, {transform.shape}, {intr.shape}")
    # Convert position to homogeneous coordinates by adding a 1 to each point
    position_h = th.cat(
        (
            position,
            th.ones(position.shape[0], 1, dtype=position.dtype, device=position.device),
        ),
        dim=-1,
    )  # (B, 4)
    # Apply the transformation matrix
    # print(f"debugging shapes: {position_h.unsqueeze(2).shape}, {transform.shape}")

    transformed_position_h = th.matmul(transform, position_h.unsqueeze(-1)).squeeze(
        -1
    )  # (B, 4)
    # Convert back to non-homogeneous coordinates (drop the last column)
    transformed_position = transformed_position_h[:, :3]  # (B, 3)
    camera_frame_coors = transformed_position
    # Project the transformed 3D points to 2D image coordinates using camera intrinsics
    # Convert to homogeneous coordinates (u, v, 1)
    projected_points_h = th.matmul(intr, transformed_position.T).T  # Shape (B, 3)
    # Normalize by z-coordinate to get (u, v) in image coordinates
    projected_points_2d = projected_points_h[:, :2] / projected_points_h[
        :, 2
    ].unsqueeze(
        -1
    )  # (B, 2)

    return projected_points_2d, camera_frame_coors


if __name__ == "__main__":

    import kornia
    import argparse
    import numpy as np
    import torch as th
    import matplotlib.pyplot as plt
    import imageio
    import os
    import sys
    from tqdm import tqdm
    from models.mark_kp import *
    from eval_dvrk.LND_fk import lndFK

    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Add the main directory to sys.path
    sys.path.insert(0, main_dir)

    from models.CtRNet import CtRNet
    from eval_dvrk.optimize import Optimize
    import pdb

    # mesh_file_test = os.listdir("urdfs/dVRK/meshes")
    # print(f"checking the mesh files: {mesh_file_test}")

    def parseArgs():
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "-md",
            "--mesh_dir",
            type=str,
            default="urdfs/dVRK/meshes",
            help="directory to mesh files",
        )
        parser.add_argument(
            "-rf", "--ref_img_file", type=str, help="reference image (mask) file"
        )

        args = parser.parse_args()

        return args

    def parseCtRNetArgs():
        parser = argparse.ArgumentParser()
        args = parser.parse_args("")

        args.use_gpu = True
        args.trained_on_multi_gpus = False

        args.height = 480
        args.width = 640
        args.fx, args.fy, args.px, args.py = (
            882.99611514,
            882.99611514,
            445.06146749,
            190.24049547,
        )
        args.scale = 1.0

        # scale the camera parameters
        args.width = int(args.width * args.scale)
        args.height = int(args.height * args.scale)
        args.fx = args.fx * args.scale
        args.fy = args.fy * args.scale
        args.px = args.px * args.scale
        args.py = args.py * args.scale

        return args

    args = parseArgs()

    mesh_dir = args.mesh_dir

    # adjust the visibility_flags in the CtRNet.py

    mesh_files = [
        f"{mesh_dir}/shaft_low_res_2.ply",
        f"{mesh_dir}/logo_low_res_1.ply",
        f"{mesh_dir}/jawright_lowres.ply",
        f"{mesh_dir}/jawleft_lowres.ply",
    ]

    ctrnet_args = parseCtRNetArgs()
    model = CtRNet(ctrnet_args)
    robot_renderer = model.setup_robot_renderer(mesh_files)
    robot_renderer.set_mesh_visibility([True, True, True, True])
    # Joint 5, 6 and Jaw

    joints = np.load("data/extra_set/joint_0203.npy")
    jaw = np.load("data/extra_set/jaw_0203.npy")

    joint_angles = np.array(
        [
            joints[4],
            joints[5],
            jaw[0] / 2,
            jaw[0] / 2,
        ]
    )
    joint_angles = th.tensor(joint_angles)
    robot_mesh = robot_renderer.get_robot_mesh(joint_angles)

    def buildcTr(cTr_train, cTr_nontrain):
        cTr = th.cat(
            [
                cTr_train[0],
                cTr_train[1],
                cTr_train[2],
            ]
        )

        return cTr

    # Angles are in degrees
    azimuth = th.tensor([0])  # degrees, for example
    elevation = th.tensor([10])  # degrees, for example
    distance = th.tensor([0.15])  # distance from the target point

    # Define the look-at target, usually the center of the scene or object
    pose_matrix = model.from_lookat_to_pose_matrix(distance, elevation, azimuth)

    camera_roll = th.tensor(150)
    roll_rad = th.deg2rad(camera_roll)  # Convert roll angle to radians
    roll_matrix = th.tensor(
        [
            [th.cos(roll_rad), -th.sin(roll_rad), 0],
            [th.sin(roll_rad), th.cos(roll_rad), 0],
            [0, 0, 1],
        ]
    )

    # Apply the roll to the pose matrix
    pose_matrix[:, :3, :3] = th.matmul(pose_matrix[:, :3, :3], roll_matrix)
    print(f"showing the pose_matrix.....{pose_matrix.shape}")  # 1, 4, 4

    cTr = model.pose_matrix_to_cTr(pose_matrix)
    # print(f"showing the projected ctr.....{cTr}")

    """Project the marked keypoints..."""
    intr = th.tensor(
        [
            [ctrnet_args.fx, 0, ctrnet_args.px],
            [0, ctrnet_args.fy, ctrnet_args.py],
            [0, 0, 1],
        ],
        device=joint_angles.device,
        dtype=joint_angles.dtype,
    )

    R_list, t_list = lndFK(joint_angles)
    # print(f"debugging list: {R_list.shape}")
    p_local = th.tensor([0.0, 0.0, 0.0094]).to(joint_angles.dtype)

    p_img1 = get_img_coords(
        p_local,
        R_list[2],
        t_list[2],
        pose_matrix.squeeze().to(joint_angles.dtype),
        intr,
    )
    p_img2 = get_img_coords(
        p_local,
        R_list[3],
        t_list[3],
        pose_matrix.squeeze().to(joint_angles.dtype),
        intr,
    )
    p_img = [p_img1, p_img2]
    # print(f'obtained img coords: {p_img1}')

    cTr_nontrain = None
    model.get_joint_angles(joint_angles)
    rendered_image = model.render_single_robot_mask(
        cTr.squeeze(),
        robot_mesh,
        robot_renderer,
    )

    rendered_np = rendered_image.squeeze().detach().cpu().numpy()
    """project keypoint: """
    rendered_np = mark_points_on_image(rendered_image, p_img)
    # plt.figure()
    plt.imshow(rendered_np)
    plt.axis("off")
    plt.show()
    plt.close()

    # Cylinder parameters

    num_points = 1  # Number of points to sample along the cylinder axis
    radius = 0.0085 / 2  # Radius of the cylinder in meters

    # Define points along the z-axis of the robot frame (i.e., the axis of the cylinder)
    position = th.zeros(
        (num_points, 3), dtype=joint_angles.dtype, device=joint_angles.device
    )  # (B, 3)
    # The direction of the cylinder is aligned along the z-axis
    direction = th.zeros(
        (num_points, 3), dtype=joint_angles.dtype, device=joint_angles.device
    )
    direction[:, 2] = 1.0  # Aligned along z-axis

    # Use the existing pose_matrix (which is already a 4x4 transformation matrix)
    pose_matrix = pose_matrix.squeeze().to(
        joint_angles.dtype
    )  # Ensure pose_matrix is of shape (4, 4)

    # Define camera intrinsic matrix (intr)
    intr = th.tensor(
        [
            [ctrnet_args.fx, 0, ctrnet_args.px],
            [0, ctrnet_args.fy, ctrnet_args.py],
            [0, 0, 1],
        ],
        device=joint_angles.device,
        dtype=joint_angles.dtype,
    )

    # Project the points using the transformation and camera intrinsics
    proj_position_2d, cam_pts_3d_position = transform_points(
        position, pose_matrix, intr
    )
    proj_norm_2d, cam_pts_3d_norm = transform_points(direction, pose_matrix, intr)
    cam_pts_3d_norm = th.nn.functional.normalize(cam_pts_3d_norm)
    # print(f" check normalized norm {cam_pts_3d_norm_t, cam_pts_3d_norm}")

    # Get the 2D points from the tensor (detach, convert to NumPy)
    proj_position_2d = proj_position_2d.detach().cpu().numpy()  # Shape (B, 2)
    proj_norm_2d = proj_norm_2d.detach().cpu().numpy()

    for i in range(proj_position_2d.shape[0]):
        # Projected point
        x, y = int(proj_position_2d[i, 0]), int(proj_position_2d[i, 1])

        # Draw the point on the image
        cv2.circle(
            rendered_np, (x, y), radius=5, color=(0, 255, 0), thickness=-1
        )  # Green color for points

        # Projected normal vector
        norm_x, norm_y = proj_norm_2d[i]

        # Calculate the end point of the normal vector for visualization (scale factor to visualize)
        scale = 0.02  # Adjust this scale factor to make the arrow more visible
        end_x = int(x + scale * (norm_x - x))
        end_y = int(y + scale * (norm_y - y))
        # print(f"debugging the norm {x, y}, {end_x, end_y}")
        # Draw the normal vector as an arrow
        cv2.arrowedLine(
            rendered_np,
            (x, y),
            (end_x, end_y),
            color=(0, 255, 0),
            thickness=3,
            tipLength=0.15,
        )  # Red arrow for normal

    # Visualize the updated image with marked keypoints using matplotlib
    plt.imshow(rendered_np, cmap="gray")
    plt.axis("off")
    plt.show()
    plt.close()

    """now test input in the function..."""
    e_1, e_2 = projectCylinderTorch(
        cam_pts_3d_position,
        cam_pts_3d_norm,
        radius,
        ctrnet_args.fx,
        ctrnet_args.fy,
        ctrnet_args.px,
        ctrnet_args.py,
    )
    # print(f"The obtained line parameters: e1: {e_1}, e2: {e_2}")

    # Extract A and B for each line from e1 and e2
    A1, B1 = e_1[0][0].item(), e_1[0][1].item()
    A2, B2 = e_2[0][0].item(), e_2[0][1].item()

    # Generate a range of u values (horizontal pixel values)
    # Assuming the image width is equal to the shape of `rendered_np`
    u_values = np.linspace(0, rendered_np.shape[1], 1000)

    # Calculate corresponding v values for each line
    # Avoid division by zero by clamping B1 and B2 if necessary
    B1 = B1 if B1 != 0 else 1e-6
    B2 = B2 if B2 != 0 else 1e-6

    v_values_line1 = (1 - A1 * u_values) / B1
    v_values_line2 = (1 - A2 * u_values) / B2

    valid_indices_line1 = (
        (u_values >= 0)
        & (u_values < rendered_np.shape[1])
        & (v_values_line1 >= 0)
        & (v_values_line1 < rendered_np.shape[0])
    )
    valid_u_values_line1 = u_values[valid_indices_line1]
    valid_v_values_line1 = v_values_line1[valid_indices_line1]

    valid_indices_line2 = (
        (u_values >= 0)
        & (u_values < rendered_np.shape[1])
        & (v_values_line2 >= 0)
        & (v_values_line2 < rendered_np.shape[0])
    )
    valid_u_values_line2 = u_values[valid_indices_line2]
    valid_v_values_line2 = v_values_line2[valid_indices_line2]
    # Clip the v values to be within the image height range

    # Plot the image and the lines
    plt.imshow(rendered_np, cmap="gray")
    plt.axis("off")

    # Plot line 1 and line 2 on the image
    plt.plot(
        valid_u_values_line1,
        valid_v_values_line1,
        color="r",
        linestyle="-",
        linewidth=4,
        label="Line 1 (e1)",
    )
    plt.plot(
        valid_u_values_line2,
        valid_v_values_line2,
        color="b",
        linestyle="-",
        linewidth=4,
        label="Line 2 (e2)",
    )

    # Add legend to differentiate the lines
    plt.legend()

    # Show the final figure with the lines overlaid
    plt.show()

    "It's working!!!!!!!!!!!"
