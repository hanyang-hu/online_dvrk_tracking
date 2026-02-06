import argparse
import numpy as np
import torch as th

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from diffcali.models.CtRNet import CtRNet
from diffcali.eval_dvrk.optimize import Optimize

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
        "-rf",
        "--ref_img_file",
        type=str,
        required=False,
        default="data/extra_set/00203.jpg",
        help="reference image (mask) file",
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


if __name__ == "__main__":
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

    joint_angles_np = np.array(
        [
            joints[4],
            joints[5],
            jaw[0] / 2,
            jaw[0] / 2,
        ]
    )

    joint_angles = th.tensor(
        joint_angles_np, requires_grad=True, device=model.device, dtype=th.float32
    )

    robot_mesh = robot_renderer.get_robot_mesh(joint_angles)

    axis_angle = th.tensor([9.9789e-01, -1.0730e00, -2.6353e-01])
    xyz = th.tensor([-9.3132e-10, -0.0000e00, 1.5000e-01])

    cTr_aa = th.tensor(
        axis_angle,
        device=model.device,
        requires_grad=True,
    )

    cTr_xy = th.tensor(
        xyz[:2],
        device=model.device,
        requires_grad=True,
    )

    cTr_z = th.tensor(
        xyz[-1:],
        device=model.device,
        requires_grad=True,
    )

    def buildcTr(cTr_train, cTr_nontrain):
        cTr = th.cat(
            [
                cTr_train[0],
                cTr_train[1],
                cTr_train[2],
            ]
        )

        return cTr

    model.get_joint_angles(joint_angles)

    opt = Optimize(
        [cTr_aa, cTr_xy, cTr_z, joint_angles],
        model,
        robot_mesh,
        robot_renderer,
        lr=2e-4,
        cTr_nontrain=None,
        buildcTr=buildcTr,
    )

    set1 = [3, 3, 7]
    set2 = [3, 3, 7]
    xyz_steps = 1
    angles_steps = 5
    opt.readRefImage(args.ref_img_file)
    saving_interval = 5
    # the iteration better greater than 1500
    cTr = opt.optimize(
        iterations=4000,
        save_fig_dir="images/dvrk/optimized/",
        ld1=set1[0],
        ld2=set1[1],
        ld3=set1[2],
        set2=set2,
        xyz_steps=xyz_steps,
        angles_steps=angles_steps,
        saving_interval=saving_interval,
        coarse_step_num=500,
    )

    print(f"Final optimzied cTr: {cTr}")
    print(f"joint angles before: {joint_angles_np}, after: {joint_angles}")
