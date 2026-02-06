import argparse
import numpy as np
import kornia
import torch
import os
import glob
import cv2
import time
import sys
import os
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffcali.models.CtRNet import CtRNet
from diffcali.utils.ui_utils import *
from diffcali.eval_dvrk.batch_optimize import BatchOptimize  # The class we just wrote
from diffcali.eval_dvrk.optimize import Optimize  # Your single-sample class
from diffcali.eval_dvrk.black_box_optimize import BlackBoxOptimize
from diffcali.utils.angle_transform_utils import enforce_axis_angle_consistency
from diffcali.utils.kpts_tracker import KeypointsTracker
# from diffcali.utils.angle_transform_utils import mix_angle_to_axis_angle, axis_angle_to_mix_angle

from diffcali.eval_dvrk.trackers import Tracker

from evotorch.tools.misc import RealOrVector # Union[float, Iterable[float], torch.Tensor]

from contextlib import contextmanager

@contextmanager
def maybe_no_grad(condition: bool):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


torch.cuda.empty_cache()

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", type=str, default="urdfs/dVRK/meshes")
    parser.add_argument("--data_dir", type=str, default="synthetic_data")  
    parser.add_argument("--data_name", type=str, default="0807")
    parser.add_argument("--frame_start", type=int, default=0) 
    parser.add_argument("--frame_end", type=int, default=99)  # End frame for the sequence
    parser.add_argument("--arm", type=str, default="psm2")
    parser.add_argument("--use_nvdiffrast", action="store_true")
    parser.add_argument("--searcher", type=str, default="CMA-ES", choices=["CMA-ES", "XNES", "SNES", "Gradient", "NelderMead", "RandomSearch", "CEM"])  # Search algorithm to use
    parser.add_argument("--online_iters", type=int, default=10)  # Number of iterations for online tracking
    parser.add_argument("--tracking_visualization", action="store_true")  # Whether to visualize the tracking process
    parser.add_argument("--online_lr", type=float, default=1e-3)  # Learning rate for online tracking
    # parser.add_argument("--use_filter", action="store_true")  # Use filter for tracking
    parser.add_argument("--downscale_factor", type=int, default=1)

    parser.add_argument('--use_low_res_mesh', type=str2bool, default=False)

    parser.add_argument('--use_tc_loss', type=str2bool, default=False) # temporal consistency loss
    parser.add_argument('--tc_weight', type=float, default=1e1)
    parser.add_argument('--filter_dt', type=float, default=1/20)
    parser.add_argument('--filter_var', type=float, default=1e-2) # process noise
    parser.add_argument('--filter_P0_pos', type=float, default=1e-2) # initial uncertainty for position
    parser.add_argument('--filter_P0_vel', type=float, default=1e0) # initial uncertainty for velocity
    parser.add_argument('--filter_R0', type=float, default=1e-2) # measurement noise

    parser.add_argument('--track_kpts', type=str2bool, default=False) # whether to track keypoints

    parser.add_argument('--use_render_loss', type=str2bool, default=True)
    parser.add_argument('--use_pts_loss', type=str2bool, default=True)
    parser.add_argument('--use_cyd_loss', type=str2bool, default=False)
    parser.add_argument('--use_dht_loss', type=str2bool, default=False)

    parser.add_argument('--use_opencv_kpts', type=str2bool, default=False)

    # parser.add_argument('--use_mix_angle', type=str2bool, default=False)
    # parser.add_argument('--use_unscented_transform', type=str2bool, default=False)
    # parser.add_argument('--use_local_quaternion', type=str2bool, default=False)
    # parser.add_argument('--use_global_quaternion', type=str2bool, default=True)
    parser.add_argument('--use_weighting_mask', type=str2bool, default=False)
    parser.add_argument('--use_prev_joint_angles', type=str2bool, default=False)

    parser.add_argument('--rotation_parameterization', type=str, default="AxisAngle", choices=["Default", "MixAngle", "UnscentedTransform", "LocalQuaternion", "GlobalQuaternion"])

    parser.add_argument('--mse_weight', type=float, default=6.)
    parser.add_argument('--dist_weight', type=float, default=12e-7)
    parser.add_argument('--app_weight', type=float, default=6e-6)
    parser.add_argument('--pts_weight', type=float, default=5e-3)
    parser.add_argument('--cyd_weight', type=float, default=1e-2)

    parser.add_argument('--popsize', type=int, default=70)

    stdev_init = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=torch.float32).cuda() # Initial standard deviation for CMA-ES
    stdev_init[:3] *= torch.tensor([1e-2, 1e-1, 1e-2], dtype=torch.float32).cuda() # angles (3D) (REMARK: set to 1e-1 if using axis angles)
    stdev_init[3:6] *= 1e-3 # translations (3D)
    stdev_init[6:] *= 1e-2 # joint angles (4D)
    stdev_init = stdev_init.detach()

    parser.add_argument("--stdev_init", type=RealOrVector, default=stdev_init)  # Standard deviation for initial noise in XNES

    parser.add_argument("--log_interval", type=int, default=1000)  # Logging interval for optimization
    args = parser.parse_args()

    args.use_mix_angle = args.rotation_parameterization == "MixAngle"
    args.use_unscented_transform = args.rotation_parameterization == "UnscentedTransform"
    args.use_local_quaternion = args.rotation_parameterization == "LocalQuaternion"
    args.use_global_quaternion = args.rotation_parameterization == "GlobalQuaternion"

    args.use_dht_loss = args.use_dht_loss and not args.use_cyd_loss # cannot use dht_loss and cylinder_loss i the same time

    if args.rotation_parameterization == "Default":
        args.stdev_init[:3] = 1e-1
    
    if args.use_global_quaternion:
        quat_stdev_init = torch.tensor([1e-2, 1e-2, 1e-2, 1e-2], dtype=torch.float32).cuda() # Initial standard deviation for 4D quaternions
        args.stdev_init = torch.cat([quat_stdev_init, args.stdev_init[3:]])

    return args


def parseCtRNetArgs():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.use_gpu = True
    args.trained_on_multi_gpus = False

    args.height = 480
    args.width = 640
    args.fx, args.fy, args.px, args.py = 1025.88223, 1025.88223, 167.919017, 234.152707
    args.scale = 1.0

    # scale the camera parameters
    args.width = int(args.width * args.scale)
    args.height = int(args.height * args.scale)
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale

    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def display_data(data_lst, idx):
    data, idx = data_lst[idx], idx % len(data_lst)
    print(f"Frame {idx}:")
    print(f"  Frame path: {data['ref_mask_path']}")
    print(f"  Ref keypoints: {data['ref_keypoints']}")
    print(f"  Joint angles: {data['joint_angles']}")
    print(f"  Optimized cTr: {data['optim_ctr']}")
    print(f"  Optimized joint angles: {data['optim_joint_angles']}")


def read_data(args):
    """
    Read the frames and relevant data from the data directory.
    """
    data_lst = []

    data_dir = os.path.join(args.data_dir, args.data_name)
    for i in range(args.frame_start, args.frame_end):
        frame_dir = os.path.join(data_dir, f"{i}")

        # Find the mask
        mask_lst = glob.glob(os.path.join(frame_dir, "*.png"))
        if len(mask_lst) == 0:
            raise ValueError(f"No mask found in {frame_dir}")
        if len(mask_lst) > 1:
            raise ValueError(f"Multiple masks found in {frame_dir}")

        mask_path = mask_lst[0]
        # frame = cv2.imread(frame_path)
        XXXX = mask_path.split("/")[-1].split(".")[0][1:]

        # Read ref_img_file of name 0XXXX.jpg
        ref_mask_path = os.path.join(frame_dir, "0" + XXXX + ".png")
        ref_img = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            raise ValueError(f"No ref_img found in {frame_dir}")
        ref_mask = (ref_img / 255.0).astype(np.float32)
        ref_mask = th.tensor(ref_mask, requires_grad=False, dtype=th.float32).cuda()

        # Get reference key points and cylinder detections
        kpts_path = os.path.join(frame_dir, f"keypoints_{i:04d}.npy")
        cyd_path = os.path.join(frame_dir, f"cylinders_{i:04d}.npy")
        if not os.path.exists(kpts_path):
            raise ValueError(f"No keypoints found in {frame_dir}")
        if not os.path.exists(cyd_path):
            raise ValueError(f"No cylinders found in {frame_dir}")
        ref_keypoints_np = np.load(kpts_path)
        det_line_params_np = np.load(kpts_path)
        ref_keypoints = torch.tensor(ref_keypoints_np).cuda()
        det_line_params = torch.tensor(det_line_params_np).cuda()

        # Alternatively, use OpenCV to detect the keypoints
        if args.use_opencv_kpts:
            ref_keypoints = get_reference_keypoints_auto(ref_mask_path, num_keypoints=2)
            ref_keypoints = torch.tensor(ref_keypoints).squeeze().float().cuda()

        # Get joint angles
        joint_path = os.path.join(frame_dir, "joint_" + XXXX + ".npy")
        jaw_path = os.path.join(frame_dir, "jaw_" + XXXX + ".npy")
        if not os.path.exists(joint_path):
            raise ValueError(f"No joint angles found in {frame_dir}")
        if not os.path.exists(jaw_path):
            raise ValueError(f"No jaw angles found in {frame_dir}")
        joints = np.load(joint_path)
        jaw = np.load(jaw_path)
        joint_angles_np = np.array(
            [joints[4], joints[5], jaw[0] / 2, jaw[0] / 2], dtype=np.float32
        )
        joint_angles = th.tensor(
            joint_angles_np, requires_grad=False, dtype=th.float32
        ).cuda() 

        # Get optimized pose and joint angles
        optim_ctr_path = os.path.join(frame_dir, "optimized_ctr.npy")
        optim_joint_path = os.path.join(frame_dir, "optimized_joint_angles.npy")
        optim_ctr_np = np.load(optim_ctr_path)
        optim_joint_angles_np = np.load(optim_joint_path)
        optim_ctr = th.tensor(
            optim_ctr_np, requires_grad=False, dtype=th.float32
        ).cuda()
        optim_joint_angles = th.tensor(
            optim_joint_angles_np, requires_grad=False, dtype=th.float32
        ).cuda()

        data = {
            # "frame": frame,
            "ref_img": cv2.imread(ref_mask_path),
            "ref_mask": ref_mask.clone(),
            "ref_mask_path": ref_mask_path,
            "ref_keypoints": ref_keypoints.clone(),
            "det_line_params": det_line_params.clone(),
            "joint_angles": joint_angles.clone(),
            "optim_ctr": optim_ctr.clone(),
            "optim_joint_angles": optim_joint_angles.clone(),
        }
        # print(data["joint_angles"], joints, jaw)
        data_lst.append(data)

    if args.use_low_res_mesh:
        mesh_files = [
            f"{args.mesh_dir}/low_res_shaft_multi_cylinder.ply",
            f"{args.mesh_dir}/low_res_logo_low_res_1.ply",
            f"{args.mesh_dir}/low_res_jawright_lowres.ply",
            f"{args.mesh_dir}/low_res_jawleft_lowres.ply",
        ]
    else:
        mesh_files = [
            f"{args.mesh_dir}/shaft_multi_cylinder.ply",
            f"{args.mesh_dir}/logo_low_res_1.ply",
            f"{args.mesh_dir}/jawright_lowres.ply",
            f"{args.mesh_dir}/jawleft_lowres.ply",
        ]

    return data_lst, mesh_files


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*copy construct from a tensor.*",
        category=UserWarning
    )

    args = parseArgs()
    ctrnet_args = parseCtRNetArgs()

    args.data_dir = os.path.join("./data", args.data_dir)

    ctrnet_args.use_nvdiffrast = args.use_nvdiffrast
    if ctrnet_args.use_nvdiffrast:
        print("Using NvDiffRast!")

    # Obtain the data
    data_lst, mesh_files = read_data(args)

    # Display the data (except for the images) for the first and last frames
    display_data(data_lst, 0)
    display_data(data_lst, -1)

    # Build the model
    model = CtRNet(ctrnet_args)
    robot_renderer = model.setup_robot_renderer(mesh_files)
    robot_renderer.set_mesh_visibility([True, True, True, True])

    # Simulate perfect initialization
    cTr = data_lst[0]["optim_ctr"]
    joint_angles = data_lst[0]["optim_joint_angles"]

    # Camera intrinsic matrix
    intr = torch.tensor(
        [
            [ctrnet_args.fx, 0, ctrnet_args.px], 
            [0, ctrnet_args.fy, ctrnet_args.py], 
            [0, 0, 1]
        ],
        device="cuda",
        dtype=joint_angles.dtype,
    )

    p_local1 = (
        torch.tensor([0.0, 0.0004, 0.009])
        .to(joint_angles.dtype)
        .to(model.device)
    )
    p_local2 = (
        torch.tensor([0.0, -0.0004, 0.009])
        .to(joint_angles.dtype)
        .to(model.device)
    )
  
    print(f"==== Tracking results ====")

    gc.collect()
    torch.cuda.empty_cache()

    with maybe_no_grad(args.searcher in ["CMA-ES", "XNES", "SNES", "Nelder-Mead", "RandomSearch", "CEM"]):
        tracker = Tracker(
            model=model,
            robot_renderer=robot_renderer,
            init_cTr=cTr,
            init_joint_angles=joint_angles,
            num_iters=args.online_iters,
            stdev_init=args.stdev_init,
            intr=intr,
            p_local1=p_local1,
            p_local2=p_local2,
            searcher=args.searcher,
            args=args,
        )

        if args.track_kpts:
            kpts_tracker = KeypointsTracker()

        # Track the rest of the frames
        mask_lst, joint_angles_lst, ref_keypoints_lst, det_line_params_lst = [], [], [], []
        for i in range(1, len(data_lst)):
            # Get keypoints and cylinder parameters
            ref_keypoints, det_line_params = None, None
            if tracker.problem.kpts_loss or args.tracking_visualization:
                ref_keypoints = data_lst[i]["ref_keypoints"].squeeze().float().cuda()
            if args.use_cyd_loss or args.use_dht_loss or args.tracking_visualization:
                det_line_params = data_lst[i]["det_line_params"].squeeze().float().cuda()

            if args.track_kpts:
                if i == 1:
                    kpts_tracker.initialize(ref_keypoints)
                else:
                    ref_keypoints = kpts_tracker.track(ref_keypoints)

            mask_lst.append(data_lst[i]["ref_mask"])
            joint_angles_lst.append(data_lst[i]["joint_angles"])
            ref_keypoints_lst.append(ref_keypoints)
            det_line_params_lst.append(det_line_params)

        mask_lst = torch.stack(mask_lst, dim=0)
        joint_angles_lst = torch.stack(joint_angles_lst, dim=0)
        ref_keypoints_lst = torch.stack(ref_keypoints_lst, dim=0) if ref_keypoints_lst[0] is not None else None
        try:
            det_line_params_lst = torch.stack(det_line_params_lst, dim=0) if det_line_params_lst[0] is not None else None
        except:
            det_line_params_lst = None

        # Track
        cTr_seq, joint_angles_seq, loss_seq, time_seq, overlay_seq = tracker.track_sequence(
            mask_lst=mask_lst,
            joint_angles_lst=joint_angles_lst,
            ref_keypoints_lst=ref_keypoints_lst,
            det_line_params_lst=det_line_params_lst,
            visualization=args.tracking_visualization,
        )

        # Print tracking results for each frame
        for i in range(1, len(data_lst), len(data_lst) // 10):
            print(f"Frame {i}: Loss = {loss_seq[i-1].item():.4f}, Time = {time_seq[i-1].item():.4f} seconds")

        # Visualize the tracking results
        if args.tracking_visualization:
            for i in range(1, len(data_lst)):
                overlay = overlay_seq[i-1]
                # overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                overlay_path = os.path.join("./tracking/", f"overlay_{i}.png")
                os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
                cv2.imwrite(overlay_path, overlay)

        # Print the average MSE and time
        avg_loss = np.mean(loss_seq.cpu().numpy())
        avg_time = np.mean(time_seq[1:].cpu().numpy()) # remove the first frame as it is the initialization frame
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Time: {avg_time:.4f} seconds")
        print("Tracking completed.")

        import matplotlib.pyplot as plt

        # Plot the prediction results of the 10 dimensions over time
        ref_cTr_seq = torch.cat(
            [
                torch.cat(
                    [
                        data_lst[i]["optim_ctr"].unsqueeze(0),
                        data_lst[i]["optim_joint_angles"].unsqueeze(0)
                    ],
                    dim=1
                ) for i in range(len(data_lst))
            ],
            dim=0
        )

        cTr_seq[:,:3] = enforce_axis_angle_consistency(cTr_seq[:, :3]) 
        ref_cTr_seq[:,:3] = enforce_axis_angle_consistency(ref_cTr_seq[:, :3])

        ref_joint_angles = torch.stack([data_lst[i]["joint_angles"] for i in range(len(data_lst))],dim=0)

        # Create figure with 10 subplots (5 rows x 2 columns)
        fig, axs = plt.subplots(5, 2, figsize=(10, 10), sharex=True)
        if args.downscale_factor == 1:
            fig.suptitle(f"Tracking Results Over Time per Dimension ({args.online_iters} Iters/Frame)", fontsize=16)
        else:
            fig.suptitle(f"Tracking Results Over Time per Dimension ({args.online_iters} Iters/Frame, Downscaled by {args.downscale_factor})", fontsize=16)

        # Flatten axes array for easy iteration
        axs = axs.flatten()

        # Plot each dimension in its own subplot
        for j in range(6):
            ax = axs[j]
            ax.plot(cTr_seq[:, j].cpu().numpy(), label='Predicted', linewidth=1.5)
            ax.plot(ref_cTr_seq[1:, j].cpu().numpy(), label='Reference', linestyle='--', linewidth=1.5)
            ax.set_title(f'Dimension {j}')
            ax.grid(True, alpha=0.4)

        for j in range(6, 10):
            ax = axs[j]
            ax.plot(joint_angles_seq[:, j-6].cpu().numpy(), label='Predicted', linewidth=1.5)
            ax.plot(ref_joint_angles[1:, j-6].cpu().numpy(), label='Reference', linestyle='--', linewidth=1.5)
            ax.set_title(f'Joint Angle {j-6}')
            ax.grid(True, alpha=0.4)

        # Add common labels
        fig.text(0.04, 0.5, 'cTr Values', va='center', rotation='vertical', fontsize=14)

        # Add a single legend below all subplots
        fig.legend(
            ['Predicted', 'Reference'],  # Labels
            loc='lower center',          # Position
            bbox_to_anchor=(0.5, 0.02), # Fine-tune position (x, y)
            ncol=2,                     # Number of columns in legend
            frameon=True,               # Add a frame
            fontsize=12                 # Adjust font size
        )

        # Adjust layout
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.98])  # Make space for suptitle and labels
        plt.show()

        # Command:  python scripts/synthetic_tracking.py   --use_nvdiffrast --tracking_visualization --rotation_parameterization MixAngle --searcher CMA-ES --downscale_factor 1 --online_iters 10 --use_pts_loss False