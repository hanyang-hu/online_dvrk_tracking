import numpy as np
import torch as th


def get_cam_coords(p_local, R, t, cTr_matrix):
    """
    Transforms a local point in the mesh frame to camera coordinates.
    """
    p_world = R @ p_local + t 
    R_c2r = cTr_matrix[:3, :3] 
    t_c2r = cTr_matrix[:3, 3]  
    p_cam = R_c2r @ p_world + t_c2r  
    return p_cam

def get_img_coords(p_local, R, t, cTr_matrix, intr, ret_cam_coords=False):
    """
    Transforms a local point in the mesh frame to image coordinates.
    """
    p_cam = get_cam_coords(p_local, R, t, cTr_matrix)

    p_cam_normalized = p_cam / p_cam[2]  
    homo_coords = intr @ p_cam_normalized  
    p_img = homo_coords[:2]  
    
    if ret_cam_coords:
        return p_img, p_cam
    return p_img

# @th.compile()
def get_img_coords_batch(p_local, R, t, cTr_matrix, intr, ret_cam_coords=False):
    """
    Transforms a batch of local points in the mesh frame to image coordinates.
    """
    B = cTr_matrix.shape[0]
    
    p_local = p_local.unsqueeze(0).expand(B, *[-1 for _ in p_local.shape])   # [B, 3]
    if len(R.shape) < 3:
        R = R.unsqueeze(0).expand(B, *[-1 for _ in R.shape])        # [B, 3, 3]
        t = t.unsqueeze(0).expand(B, *[-1 for _ in t.shape])        # [B, 3]

    p_world = th.bmm(R, p_local.unsqueeze(-1)).squeeze(-1) + t  # p_world: [batch_size, 3]

    R_c2r = cTr_matrix[:, :3, :3]  # [batch_size, 3, 3]
    t_c2r = cTr_matrix[:, :3, 3]   # [batch_size, 3]

    p_cam = th.bmm(R_c2r, p_world.unsqueeze(-1)).squeeze(-1) + t_c2r  # p_cam: [batch_size, 3]

    p_cam_normalized = p_cam / p_cam[:, 2:3]  # Normalize by depth (z-coordinate), [batch_size, 3]
        
    homo_coords = th.matmul(intr, p_cam_normalized.T).T  # Apply camera intrinsics, [batch_size, 3]
    p_img = homo_coords[:, :2]  # Extract (u, v) image coordinates, [batch_size, 2]

    if ret_cam_coords:
        return p_img, p_cam
    return p_img