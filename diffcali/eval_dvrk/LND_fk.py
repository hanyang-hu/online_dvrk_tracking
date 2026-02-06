import torch
import torch as th


@torch.compile()
def lndFK(joint_angles: th.Tensor):
    """
    Forward kinematics of LND starting from Frame 4
    param joint_angles: joint angles at Joint 5, 6, 7, 8 (PyTorch tensor)
    return: rotation matrices at Frame 4, 5, 7, 8; (4, 3, 3) PyTorch tensors
    return: translation vectors at Frame 4, 5, 7, 8; (4, 3) PyTorch tensors
    """
    device = joint_angles.device
    dtype = joint_angles.dtype

    # Clone joint angles to avoid in-place modifications on views
    theta0 = joint_angles[0].clone()
    theta1 = joint_angles[1].clone()
    theta2 = joint_angles[2].clone()
    theta3 = joint_angles[3].clone()

    # Frame 4 (Base)
    T_4 = th.eye(
        4, device=device, dtype=dtype
    )  # Identity matrix since we start at Frame 4

    # Transformation from Frame 4 to Frame 5
    T_4_5 = th.stack(
        [
            th.stack(
                [
                    th.sin(theta0),
                    th.cos(theta0),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.cos(theta0),
                    -th.sin(theta0),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
        ],
        dim=0,
    )

    T_5_6 = th.stack(
        [
            th.stack(
                [
                    th.sin(theta1),
                    th.cos(theta1),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0091, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.cos(theta1),
                    -th.sin(theta1),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
        ],
        dim=0,
    )

    # Mesh transformation matrices (static, not dependent on joint angles)
    T_4_mesh = th.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )

    T_5_mesh = th.tensor(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )

    # Compute transformations from Frame 4 onward
    T_5 = T_4 @ T_4_5
    T_6 = T_5 @ T_5_6
    T_4 = T_4 @ T_4_mesh
    T_5 = T_5 @ T_5_mesh

    # Frame 7: right gripper
    T_6_7 = th.stack(
        [
            th.stack(
                [
                    th.cos(theta2),
                    th.sin(theta2),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    -th.sin(theta2),
                    th.cos(theta2),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
        ],
        dim=0,
    )

    T_7_mesh = th.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )

    T_7 = T_6 @ T_6_7 @ T_7_mesh

    # Frame 8: left gripper
    T_6_8 = th.stack(
        [
            th.stack(
                [
                    th.cos(theta3),
                    -th.sin(theta3),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.sin(theta3),
                    th.cos(theta3),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
            th.stack(
                [
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(0.0, device=device, dtype=dtype),
                    th.tensor(1.0, device=device, dtype=dtype),
                ],
                dim=0,
            ),
        ],
        dim=0,
    )

    T_8 = T_6 @ T_6_8 @ T_7_mesh

    # Extract rotation matrices and translation vectors

    R_list = th.stack([T_4[:3, :3], T_5[:3, :3], T_7[:3, :3], T_8[:3, :3]], dim=0)
    t_list = th.stack([T_4[:3, 3], T_5[:3, 3], T_7[:3, 3], T_8[:3, 3]], dim=0)

    return R_list, t_list


def rotation_z(theta, zeros, ones):
    return torch.stack([
        torch.stack([torch.cos(theta), torch.sin(theta), zeros, zeros], dim=1),
        torch.stack([-torch.sin(theta), torch.cos(theta), zeros, zeros], dim=1),
        torch.stack([zeros, zeros, ones, zeros], dim=1),
        torch.stack([zeros, zeros, zeros, ones], dim=1),
    ], dim=1)


def transform(sin, cos, trans_x=0.0, trans_y=0.0, trans_z=0.0, B=1, zeros=0., ones=1., device=0, dtype=torch.float32):
    tx = torch.full((B,), trans_x, device=device, dtype=dtype)
    ty = torch.full((B,), trans_y, device=device, dtype=dtype)
    tz = torch.full((B,), trans_z, device=device, dtype=dtype)
    return torch.stack([
        torch.stack([sin, cos, zeros, tx], dim=1),
        torch.stack([zeros, zeros, ones, ty], dim=1),
        torch.stack([cos, -sin, zeros, tz], dim=1),
        torch.stack([zeros, zeros, zeros, ones], dim=1),
    ], dim=1)


@torch.compile()
def batch_lndFK(joint_angles: torch.Tensor):
    """
    Batched version of lndFK.
    Args:
        joint_angles (torch.Tensor): (B, 4) joint angles for joints 5–8.
    Returns:
        R_list (B, 4, 3, 3): Rotation matrices at Frames 4, 5, 7, 8
        t_list (B, 4, 3): Translation vectors at Frames 4, 5, 7, 8
    """
    B = joint_angles.shape[0]
    device = joint_angles.device
    dtype = joint_angles.dtype

    theta0, theta1, theta2, theta3 = joint_angles[:, 0], joint_angles[:, 1], joint_angles[:, 2], joint_angles[:, 3]

    zeros = torch.zeros(B, device=device, dtype=dtype)
    ones = torch.ones(B, device=device, dtype=dtype)

    T_4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

    T_4_5 = transform(torch.sin(theta0), torch.cos(theta0), B=B, zeros=zeros, ones=ones, device=device, dtype=dtype)
    T_5_6 = transform(torch.sin(theta1), torch.cos(theta1), trans_x=0.0091, B=B, zeros=zeros, ones=ones, device=device, dtype=dtype)

    T_4_mesh = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], device=device, dtype=dtype).expand(B, -1, -1)

    T_5_mesh = torch.tensor([
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], device=device, dtype=dtype).expand(B, -1, -1)

    T_5 = T_4 @ T_4_5
    T_6 = T_5 @ T_5_6
    T_4 = T_4 @ T_4_mesh
    T_5 = T_5 @ T_5_mesh

    T_6_7 = rotation_z(theta2, zeros, ones)
    T_6_8 = rotation_z(-theta3, zeros, ones)

    T_7_mesh = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], device=device, dtype=dtype).expand(B, -1, -1)

    T_7 = T_6 @ T_6_7 @ T_7_mesh
    T_8 = T_6 @ T_6_8 @ T_7_mesh

    R = torch.stack([T_4[:, :3, :3], T_5[:, :3, :3], T_7[:, :3, :3], T_8[:, :3, :3]], dim=1)
    t = torch.stack([T_4[:, :3, 3], T_5[:, :3, 3], T_7[:, :3, 3], T_8[:, :3, 3]], dim=1)

    return R, t


if __name__ == "__main__":
    import time
    with torch.no_grad():
        # Testing
        B = 1000
        # Random joint angles batch with gradient tracking
        theta_batch = torch.randn(B, 4, dtype=torch.double).cuda()

        # warm-up
        batch_lndFK(theta_batch)
        lndFK(theta_batch[0])

        # Compute batched FK
        start_time = time.time()
        R_batch, T_batch = batch_lndFK(theta_batch)
        end_time = time.time()
        print(f"Batched FK computation time: {(end_time - start_time) * 1000:.4f} ms")

        start_time = time.time()
        # Compute serial FK for each sample
        R_iter, T_iter = None, None
        for i in range(B):
            R_i, T_i = lndFK(theta_batch[i])
            if R_iter is None:
                R_iter = R_i.unsqueeze(0)
                T_iter = T_i.unsqueeze(0)
            else:
                R_iter = torch.cat((R_iter, R_i.unsqueeze(0)), dim=0)
                T_iter = torch.cat((T_iter, T_i.unsqueeze(0)), dim=0)
        end_time = time.time()
        print(f"Iterative FK computation time: {(end_time - start_time) * 1000:.4f} ms")
        print(f"Batch size: {B}, Iterative batch size: {R_iter.shape[0]}")

        # Verify numerical equality
        assert torch.allclose(R_batch, R_iter, atol=1e-16), "Rotations do not match!"
        assert torch.allclose(T_batch, T_iter, atol=1e-16), "Translations do not match!"
        print("Batched and iterative outputs match within tolerance.")

    # # Optional: Check gradient flow
    # # For example, sum all outputs to get a scalar and backprop
    # loss = R_batch.sum() + T_batch.sum()
    # loss.backward()
    # # Ensure gradients exist for the input
    # assert theta_batch.grad is not None
    # print("Gradient check passed; theta.grad is available.")