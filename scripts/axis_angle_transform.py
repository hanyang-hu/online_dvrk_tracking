import torch
import kornia

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffcali.utils.angle_transform_utils import (
    mix_angle_to_axis_angle,
    axis_angle_to_mix_angle
)


if __name__ == "__main__":
    cTr_batch = [
        [-1.4308, -0.7323, -3.6195,  0.0158,  0.0088,  0.1247],
        [-1.2985, -0.2628, -3.2461,  0.0126,  0.0093,  0.1263],
        [-1.4799,  0.1346, -2.6478,  0.0057,  0.0085,  0.1213],
        [-1.5622,  0.3098, -2.5426,  0.0070,  0.0082,  0.1250],
        [-0.8263,  0.7654, -1.8426,  0.0037,  0.0040,  0.1260],
        [-0.8170,  0.6794, -1.8810,  0.0039,  0.0047,  0.1242],
        [-1.1446,  0.4885, -2.3053,  0.0047,  0.0066,  0.1255],
        [-1.6220, -0.1235, -2.8538,  0.0094,  0.0094,  0.1264],
        [-1.6482, -0.3589, -3.1212,  0.0109,  0.0096,  0.1251],
        [-1.6269, -0.8210, -3.4761,  0.0144,  0.0100,  0.1268]
    ]
    cTr_batch = torch.tensor(cTr_batch, dtype=torch.float32).cuda()
    axis_angle_batch = cTr_batch[:, :3]
    print("Axis-Angle (radians):\n", axis_angle_batch)

    # Convert to euler angles
    mix_angles = axis_angle_to_mix_angle(axis_angle_batch)
    print("Euler Angles (radians):\n", mix_angles)

    # Convert back to axis-angle
    axis_angle_converted = mix_angle_to_axis_angle(mix_angles)
    print("Converted Axis-Angle:\n", axis_angle_converted)

    # assert torch.allclose(axis_angle_batch, axis_angle_converted, atol=1e-6), "Axis-angle conversion failed"

    # Convert to rotation matrices
    R = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle_batch)
    R_converted = kornia.geometry.conversions.axis_angle_to_rotation_matrix(axis_angle_converted)

    # print(R)
    # print(R_converted)

    # Check if the conversion is consistent
    rot_diff = torch.norm(R - R_converted, dim=(1, 2))
    print("Rotation matrix difference:", rot_diff)
    assert torch.allclose(R, R_converted, atol=1e-6)

    # Original Euler
    euler = torch.randn(10, 3).cuda()

    # Forward
    axis = mix_angle_to_axis_angle(euler)

    # Inverse
    euler_rec = axis_angle_to_mix_angle(axis)

    # Round-trip
    axis_rec = mix_angle_to_axis_angle(euler_rec)

    # print("Euler original:\n", euler)
    # print("Euler recovered:\n", euler_rec)
    print("Axis diff norm:\n", (axis - axis_rec).norm(dim=1))

    # Original axis
    axis = torch.randn(10, 3).cuda()

    # Forward
    euler = axis_angle_to_mix_angle(axis)

    # Inverse
    axis_rec = mix_angle_to_axis_angle(euler)
    
    # print("Axis original:\n", axis)
    # print("Axis recovered:\n", axis_rec)
    print("Axis diff norm:\n", (axis - axis_rec).norm(dim=1))