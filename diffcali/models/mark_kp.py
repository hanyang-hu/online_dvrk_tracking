import cv2
import numpy as np
import torch as th

""" This script provides util functions that project keypoints from mesh frame to world (origin) frame"""

# def get_world_coords(p_local, R, t):
#     p_world = R @ p_local + t
#     return p_world

# def get_img_coords(p_world, cTr_matrix, intr):
#     R_c2r = cTr_matrix[:3, :3]
#     t_c2r = cTr_matrix[:3, 3]
#     p_cam = R_c2r @ p_world + t_c2r # camera coords of keypoint
#     homo_coords = intr @ (p_cam/p_cam[2])
#     p_img = homo_coords[:2]

#     return p_img


def get_img_coords(p_local, R, t, cTr_matrix, intr):
    """
    Transforms a local point in the mesh frame to image coordinates.

    Args:
        p_local (torch.Tensor): Local coordinates of the point in the mesh frame (shape: [3]).
        R (torch.Tensor): Rotation matrix of the mesh in the world frame (shape: [3, 3]).
        t (torch.Tensor): Translation vector of the mesh in the world frame (shape: [3]).
        cTr_matrix (torch.Tensor): Camera-to-robot transformation matrix (shape: [4, 4]).
        intr (torch.Tensor): Camera intrinsic matrix (shape: [3, 3]).

    Returns:
        p_img (torch.Tensor): Image coordinates of the point (shape: [2]).
    """
    # Step 1: Transform the point from local mesh coordinates to world coordinates
    p_world = R @ p_local + t  # p_world is a tensor of shape [3]

    # Step 2: Transform the point from world coordinates to camera coordinates
    R_c2r = cTr_matrix[:3, :3]  # Rotation from camera to robot (shape: [3, 3])
    t_c2r = cTr_matrix[:3, 3]  # Translation from camera to robot (shape: [3])
    p_cam = (
        R_c2r @ p_world + t_c2r
    )  # p_cam is the point in camera coordinates (shape: [3])

    # Step 3: Project the point from camera coordinates to image coordinates
    p_cam_normalized = p_cam / p_cam[2]  # Normalize by depth (z-coordinate)
    homo_coords = intr @ p_cam_normalized  # Apply camera intrinsics
    p_img = homo_coords[:2]  # Extract (u, v) image coordinates

    return p_img


def get_img_coords_batch(p_local, R, t, cTr_matrix, intr):
    """
    Transforms a batch of local points in the mesh frame to image coordinates.

    Args:
        p_local (torch.Tensor): Local coordinates of the points in the mesh frame (shape: [3]).
        R (torch.Tensor): Rotation matrices of the mesh in the world frame (shape: [3, 3]).
        t (torch.Tensor): Translation vectors of the mesh in the world frame (shape: [3]).
        cTr_matrix (torch.Tensor): Camera-to-robot transformation matrices (shape: [batch_size, 4, 4]).
        intr (torch.Tensor): Camera intrinsic matrix (shape: [3, 3]).

    Returns:
        p_img (torch.Tensor): Image coordinates of the points (shape: [batch_size, 2]).
    """
    # Extend all the fix parameters
    B = cTr_matrix.shape[0]
    p_local = p_local.unsqueeze(0).expand(B, *[-1 for _ in p_local.shape])  # [B, 3]
    R = R.unsqueeze(0).expand(B, *[-1 for _ in R.shape])  # [B, 3, 3]
    t = t.unsqueeze(0).expand(B, *[-1 for _ in t.shape])  # [B, 3]

    # Step 1: Transform the point from local mesh coordinates to world coordinates
    # p_local: [batch_size, 3], R: [batch_size, 3, 3], t: [batch_size, 3]
    p_world = (
        th.bmm(R, p_local.unsqueeze(-1)).squeeze(-1) + t
    )  # p_world: [batch_size, 3]

    # Step 2: Transform the point from world coordinates to camera coordinates
    # Extract rotation and translation from cTr_matrix
    R_c2r = cTr_matrix[:, :3, :3]  # [batch_size, 3, 3]
    t_c2r = cTr_matrix[:, :3, 3]  # [batch_size, 3]

    p_cam = (
        th.bmm(R_c2r, p_world.unsqueeze(-1)).squeeze(-1) + t_c2r
    )  # p_cam: [batch_size, 3]

    # Step 3: Project the point from camera coordinates to image coordinates
    p_cam_normalized = (
        p_cam / p_cam[:, 2:3]
    )  # Normalize by depth (z-coordinate), [batch_size, 3]

    # p_cam_normalized_mask = th.isnan(p_cam_normalized)
    # if p_cam_normalized_mask.any():
    #     print(f"debugging the z axis : depth: {p_cam}, pose_matrix {cTr_matrix}")

    homo_coords = th.matmul(
        intr, p_cam_normalized.T
    ).T  # Apply camera intrinsics, [batch_size, 3]
    p_img = homo_coords[:, :2]  # Extract (u, v) image coordinates, [batch_size, 2]

    return p_img


def mark_points_on_image(rendered_image, keypoints_img):
    # Convert rendered image to NumPy array if it's a tensor
    rendered_image_np = rendered_image.squeeze().detach().cpu().numpy()
    if rendered_image_np.ndim == 2:
        rendered_image_np = np.stack([rendered_image_np] * 3, axis=-1)  # Convert to RGB

    h, w, _ = rendered_image_np.shape

    # Ensure keypoints_img is iterable
    if isinstance(keypoints_img, th.Tensor):
        if keypoints_img.dim() == 1 and keypoints_img.shape[0] == 2:
            # Single keypoint, wrap it in a list
            keypoints_img = [keypoints_img]
        elif keypoints_img.dim() == 2 and keypoints_img.shape[1] == 2:
            # Multiple keypoints in a tensor of shape [N, 2]
            keypoints_img = [keypoints_img[i] for i in range(keypoints_img.shape[0])]
        else:
            raise ValueError("keypoints_img tensor has an unexpected shape.")

    elif isinstance(keypoints_img, (list, tuple)):
        # If it's a list or tuple, we assume it's already an iterable of keypoints
        pass
    else:
        raise TypeError("keypoints_img must be a tensor or a list/tuple of tensors.")

    # Iterate over keypoints
    for p_img in keypoints_img:
        u_int = int(round(p_img[0].item()))
        v_int = int(round(p_img[1].item()))

        if 0 <= v_int < h and 0 <= u_int < w:
            # Draw a red circle at (u_int, v_int)
            cv2.circle(
                rendered_image_np,
                (u_int, v_int),
                radius=3,
                color=(255, 0, 0),
                thickness=-1,
            )
        else:
            print(f"Projected point ({u_int}, {v_int}) is outside the image bounds.")

    return rendered_image_np
