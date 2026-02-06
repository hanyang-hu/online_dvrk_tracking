import torch
import kornia


@torch.compile()
def enforce_quaternion_consistency(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Enforces the consistency of quaternion signs by aligning them with the first quaternion in the batch.
    Input: quaternions (B, 4) where B is the batch size.
    Output: quaternions with consistent signs.
    """
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("Input quaternions must have shape (B, 4)")

    # Compute the sign consistency based on the first quaternion's vector part
    temp = quaternions[0, 1:].unsqueeze(0)  # Use the first quaternion's vector part for sign consistency
    # signs = torch.sign((temp * quaternions[1:,1:]).sum(dim=-1)).squeeze()  # Compute dot product to determine sign consistency
    # signs = torch.cat([torch.tensor([1.0], device=quaternions.device), signs])
    # signs = signs.unsqueeze(1).to(quaternions.device)  # Reshape to (B, 1) for broadcasting

    signs = torch.sign((temp * quaternions[:,1:]).sum(dim=-1)).unsqueeze(-1)

    return quaternions * signs


# @torch.compile()
def enforce_axis_angle_consistency(axis_angles: torch.Tensor) -> torch.Tensor:
    if axis_angles.ndim != 2 or axis_angles.shape[1] != 3:
        raise ValueError("Input axis angles must have shape (B, 3)")

    temp = axis_angles[0,:].unsqueeze(0)
    signs = torch.sign((temp * axis_angles).sum(dim=-1)).unsqueeze(-1)

    return axis_angles * signs


@torch.compile()
def mix_angle_to_rotmat(mix_angle: torch.Tensor) -> torch.Tensor:
    """
    Converts [alpha, beta, gamma] (in radians) with
      R = R_y(gamma) @ R_x(alpha) @ R_z(beta)
    into (axis, angle) representation of shape (B,3),
    where the 3-vector is axis * angle.
    """
    # unpack
    alpha = mix_angle[:, 0]
    beta  = mix_angle[:, 1]
    gamma = mix_angle[:, 2]

    # pre‑compute sines/cosines
    ca, sa = alpha.cos(), alpha.sin()
    cb, sb = beta.cos(),  beta.sin()
    cg, sg = gamma.cos(), gamma.sin()

    # build each [B,3,3]
    # R_x(alpha)
    R_x = torch.stack([
        torch.stack([torch.ones_like(ca),  torch.zeros_like(ca),  torch.zeros_like(ca)], dim=-1),
        torch.stack([torch.zeros_like(ca), ca,                   -sa], dim=-1),
        torch.stack([torch.zeros_like(ca), sa,                    ca], dim=-1),
    ], dim=-2)

    # R_z(beta)
    R_z = torch.stack([
        torch.stack([ cb,                  -sb,                   torch.zeros_like(cb)], dim=-1),
        torch.stack([ sb,                   cb,                   torch.zeros_like(cb)], dim=-1),
        torch.stack([torch.zeros_like(cb),  torch.zeros_like(cb), torch.ones_like(cb)], dim=-1),
    ], dim=-2)

    # R_y(gamma)
    R_y = torch.stack([
        torch.stack([ cg,                   torch.zeros_like(cg), sg], dim=-1),
        torch.stack([ torch.zeros_like(cg), torch.ones_like(cg),  torch.zeros_like(cg)], dim=-1),
        torch.stack([-sg,                   torch.zeros_like(cg), cg], dim=-1),
    ], dim=-2)

    # compose: R = R_y @ R_x @ R_z
    R = R_y @ (R_x @ R_z)  # (B,3,3)

    return R


@torch.compile()
def mix_angle_to_axis_angle(mix_angle: torch.Tensor) -> torch.Tensor:
    """
    Converts [alpha, beta, gamma] (in radians) with
      R = R_y(gamma) @ R_x(alpha) @ R_z(beta)
    into (axis, angle) representation of shape (B,3),
    where the 3-vector is axis * angle.
    """
    R = mix_angle_to_rotmat(mix_angle)  # (B, 3, 3)

    # convert to axis-angle (axis * angle)
    axis_angle = kornia.geometry.conversions.rotation_matrix_to_axis_angle(R)

    return axis_angle  # (B,3)


@torch.compile()
def axis_angle_to_mix_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Converts axis-angle vectors (B,3) to a mixed angle representation [alpha, beta, gamma] (B,3),
    where R = R_y(gamma) @ R_x(alpha) @ R_z(beta).

    Input:
        axis_angle: (B,3) torch.Tensor where vector = axis * angle
    Output:
        mix_angle: (B,3) torch.Tensor with angles [alpha, beta, gamma] in radians
    """
    R = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle)  # (B, 3, 3)

    # Extract relevant components
    r10 = R[:, 1, 0]
    r11 = R[:, 1, 1]
    r12 = R[:, 1, 2]
    r02 = R[:, 0, 2]
    r22 = R[:, 2, 2]

    # α = asin(-r12)
    alpha = torch.asin(torch.clamp(-r12, -1.0, 1.0))  # [B]

    # cos(α) for division safety
    cos_alpha = torch.cos(alpha)
    cos_alpha = torch.where(cos_alpha.abs() < 1e-6, torch.full_like(cos_alpha, 1e-6), cos_alpha)

    # β = atan2(r10 / cos(α), r11 / cos(α))
    beta = torch.atan2(r10 / cos_alpha, r11 / cos_alpha)

    # γ = atan2(r02 / cos(α), r22 / cos(α))
    gamma = torch.atan2(r02 / cos_alpha, r22 / cos_alpha)

    return torch.stack([alpha, beta, gamma], dim=-1)


def unscented_mix_angle_to_axis_angle(mix_angle: torch.Tensor, stdev: torch.Tensor) -> torch.Tensor:
    """
    Use Unscented Transform to convert mix angles to axis-angle representation.
    Input:
        mix_angle: (3,) torch.Tensor with angles [alpha, beta, gamma] in radians
        stdev: (3,) torch.Tensor with standard deviations for each angle
    Output:
        mean: (3,) torch.Tensor with mean axis-angle vector
        cov: (3,3) torch.Tensor with covariance of the axis-angle vector
    """
    D = 3
    assert mix_angle.shape == (3,)
    assert stdev.shape == (3,)

    # Generate sigma points for each batch element
    c = torch.sqrt(torch.tensor(3.0, device=stdev.device))  # scaling factor
    sigma_offsets = torch.stack([
        torch.zeros(3, device=stdev.device),             # mean
        c * stdev * torch.tensor([1, 0, 0], device=stdev.device),
        c * stdev * torch.tensor([0, 1, 0], device=stdev.device),
        c * stdev * torch.tensor([0, 0, 1], device=stdev.device),
        -c * stdev * torch.tensor([1, 0, 0], device=stdev.device),
        -c * stdev * torch.tensor([0, 1, 0], device=stdev.device),
        -c * stdev * torch.tensor([0, 0, 1], device=stdev.device),
    ]) 
    sigma_points = mix_angle[None, :] + sigma_offsets  # (7, 3)

    # Transform sigma points to axis-angle
    sigma_transformed = mix_angle_to_axis_angle(sigma_points) # (7, 3)
    sigma_transformed = enforce_axis_angle_consistency(sigma_transformed)

    # Weights for UT (symmetric)
    Wm = torch.full((2 * D + 1,), 1.0 / (2 * D), device=mix_angle.device)
    Wm[0] = 0.0  # no weight for the center if desired
    Wc = Wm.clone()

    # Compute mean and covariance
    mean = torch.sum(Wm[:, None] * sigma_transformed, dim=0)  # (3,)
    diffs = sigma_transformed - mean[None, :]  # (7, 3)
    cov = torch.einsum("i,ij,ik->jk", Wc, diffs, diffs)  # (3, 3)

    return mean, cov


def find_local_quaternion_basis(mix_angle: torch.Tensor, stdev: torch.Tensor) -> torch.Tensor:
    D = 3
    assert mix_angle.shape == (3,)
    assert stdev.shape == (3,)

    # Generate sigma points for each batch element
    # c = torch.sqrt(torch.tensor(3.0, device=stdev.device))  # scaling factor
    # sigma_offsets = torch.stack([
    #     torch.zeros(3, device=stdev.device),             # mean
    #     c * stdev * torch.tensor([1, 0, 0], device=stdev.device),
    #     c * stdev * torch.tensor([0, 1, 0], device=stdev.device),
    #     c * stdev * torch.tensor([0, 0, 1], device=stdev.device),
    #     -c * stdev * torch.tensor([1, 0, 0], device=stdev.device),
    #     -c * stdev * torch.tensor([0, 1, 0], device=stdev.device),
    #     -c * stdev * torch.tensor([0, 0, 1], device=stdev.device),
    # ]) 
    # samples = mix_angle[None, :] + sigma_offsets  # (7, 3)
    sample_number = 1000
    samples = torch.randn(sample_number, 3).cuda() * stdev.unsqueeze(0) + mix_angle[None, :]

    # Transform sigma points to quaternions
    axis_angle = mix_angle_to_axis_angle(samples)
    q = kornia.geometry.conversions.axis_angle_to_quaternion(axis_angle)
    q_norm = q / q.norm(dim=1, keepdim=True) # ensure unit norm
    q_proj = q_norm / torch.sum(q_norm * q_norm[0], dim=1, keepdim=True) - q_norm[0].unsqueeze(0) # intersection of the ray {t*q[i]} with the tangent plane at q[0]

    # Find 4D basis of the 3D tangent space by eigen-decomposition
    diff = q_proj - q_proj.mean(dim=0).unsqueeze(0)
    cov_4D = diff.T @ diff / sample_number
    eigvals_4D, eigvecs_4D = torch.linalg.eigh(cov_4D)
    basis_4D = eigvecs_4D[:, 1:]  # remove the direction with (almost) zero variance
    lengthscales = torch.sqrt(eigvals_4D[1:])

    # print(lengthscales)

    return q_norm[0], q_proj[0] @ basis_4D, basis_4D, lengthscales