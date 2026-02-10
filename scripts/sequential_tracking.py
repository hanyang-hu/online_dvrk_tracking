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
import matplotlib
import matplotlib.pyplot as plt
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffcali.models.CtRNet import CtRNet
from diffcali.utils.ui_utils import *
from diffcali.eval_dvrk.batch_optimize import BatchOptimize, HeterogeneousBatchOptimize
from diffcali.eval_dvrk.optimize import Optimize  # Your single-sample class
from diffcali.eval_dvrk.black_box_optimize import BlackBoxOptimize
from diffcali.utils.angle_transform_utils import enforce_axis_angle_consistency, enforce_quaternion_consistency
from diffcali.utils.angle_transform_utils import mix_angle_to_axis_angle, axis_angle_to_mix_angle
from diffcali.utils.contour_tip_net import ContourTipNet, detect_keypoints

from diffcali.eval_dvrk.trackers import Tracker

from TuRBO.turbo.turbo_1 import Turbo1
from diffcali.eval_dvrk.black_box_optimize import BayesOptBatchProblem

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
    parser.add_argument("--data_dir", type=str, default="consecutive_prediction")  
    parser.add_argument("--difficulty", type=str, default="0617")
    parser.add_argument("--frame_start", type=int, default=0) 
    parser.add_argument("--frame_end", type=int, default=-1)  # End frame for the sequence
    parser.add_argument("--batch_opt_lr", type=float, default=3e-3)
    parser.add_argument("--single_opt_lr", type=float, default=5e-4) # if using gradient descent
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument(
        "--batch_iters", type=int, default=100
    )  # Coarse steps per batch
    parser.add_argument(
        "--final_iters", type=int, default=0
    )  # Final single-sample refine using XNES / gradient descent
    parser.add_argument("--arm", type=str, default="psm2")
    parser.add_argument("--sample_number", type=int, default=200)
    parser.add_argument("--use_nvdiffrast", action="store_true")
    parser.add_argument("--use_bo_initializer", action="store_true") # Use Bayesian optimization for initialization (do not rely on joint angle readings)
    parser.add_argument("--searcher", type=str, default="CMA-ES", choices=["CMA-ES", "XNES", "Gradient"])  # Search algorithm to use
    parser.add_argument("--online_iters", type=int, default=10)  # Number of iterations for online tracking
    parser.add_argument("--tracking_visualization", action="store_true")  # Whether to visualize the tracking process
    # parser.add_argument("--online_lr", type=float, default=1e-3)  # Learning rate for online tracking
    # parser.add_argument("--use_filter", action="store_true")  # Use filter for tracking
    parser.add_argument("--no_cache", action="store_true") # Use cached initialization
    parser.add_argument("--downscale_factor", type=int, default=1)

    parser.add_argument('--use_low_res_mesh', type=str2bool, default=False)

    parser.add_argument('--symmetric_jaw', type=str2bool, default=True)

    parser.add_argument('--use_render_loss', type=str2bool, default=True)
    parser.add_argument('--use_pts_loss', type=str2bool, default=True)

    parser.add_argument('--use_prev_joint_angles', type=str2bool, default=False)

    parser.add_argument('--rotation_parameterization', type=str, default="AxisAngle", choices=["AxisAngle", "MixAngle"])

    parser.add_argument('--mse_weight', type=float, default=6.) #  originally 6.
    parser.add_argument('--dist_weight', type=float, default=0.) # originally 12e-7, turned off
    parser.add_argument('--app_weight', type=float, default=6e-6)
    parser.add_argument('--pts_weight', type=float, default=3e-3) # originally 5e-3, use 5e-5 for less pts loss weight
    parser.add_argument('--tip_emd_weight', type=float, default=7e-2) # originally 7e-2 for normalized radial profile

    parser.add_argument('--use_contour_tip_net', type=str2bool, default=True) # whether to use ContourTipNet for keypoint detection
    parser.add_argument('--contour_tip_net_path', type=str, default='./ContourTipNet/models/cnn_model.pth') # path to the ContourTipNet model

    parser.add_argument('--popsize', type=int, default=70)

    parser.add_argument('--filter_option', type=str, default="Kalman", choices=["None", "OneEuro", "OneEuro_orig", "Kalman"]) # which variables to filter

    parser.add_argument('--cos_reparams', type=str2bool, default=True) # whether to use cosine reparameterization (do not use for gradient-based methods), if not, simply clamp the angles within valid ranges

    stdev_init = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=torch.float32).cuda() # Initial standard deviation for CMA-ES
    stdev_init[:3] *= torch.tensor([1e-2, 1e-1, 1e-2], dtype=torch.float32).cuda() # angles (3D) (REMARK: set to 1e-1 if using axis angles)
    stdev_init[3:6] *= 1e-3 # translations (3D)
    stdev_init[6:] *= 5e-2 # joint angles (4D)
    stdev_init = stdev_init.detach()

    parser.add_argument("--stdev_init", type=RealOrVector, default=stdev_init)  # Standard deviation for initial noise in XNES

    parser.add_argument("--log_interval", type=int, default=1000)  # Logging interval for optimization
    args = parser.parse_args()

    args.use_filter = False if args.filter_option == "None" else True

    args.use_mix_angle = (args.rotation_parameterization == "MixAngle")

    if args.rotation_parameterization == "AxisAngle":
        args.stdev_init[:3] = 1e-1

    args.stdev_init[6] *= 2 # wrist pitch
    args.stdev_init[7] *= 2 # wrist yaw
    args.stdev_init[8:] *= 2 # jaws

    if not args.use_prev_joint_angles:
        args.stdev_init[6:] /= 10. # if using joint angle readings, set the stdev for joint angles to a smaller value

    return args


def parseCtRNetArgs():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.use_gpu = True
    args.trained_on_multi_gpus = False

    # args.height = 480
    # args.width = 640
    # args.fx, args.fy, args.px, args.py = 1025.88223, 1025.88223, 167.919017, 234.152707

    # Setting for SurgPose data
    args.height = 986 // 2
    args.width = 1400 // 2
    args.fx, args.fy, args.px, args.py = 1811.910046453570 / 2, 1809.640734154330 / 2, 588.5594517681759 / 2, 477.3975900383616 / 2

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


def buildcTr(cTr_train, cTr_nontrain):
    # Rebuild [angle_axis(3), xyz(3)] from cTr_train
    return th.cat([cTr_train[0], cTr_train[1], cTr_train[2]], dim=0)


def display_data(data_lst, idx):
    data, idx = data_lst[idx], idx % len(data_lst)
    print(f"Frame {idx}:")
    print(f"  Frame path: {data['ref_mask_path']}")
    print(f"  Ref keypoints: {data['ref_keypoints']}")
    print(f"  Joint angles: {data['joint_angles']}")
    # print(f"  Optimized cTr: {data['optim_ctr']}")
    # print(f"  Optimized joint angles: {data['optim_joint_angles']}")


def read_data(args, args_ctrnet=None):
    """
    Read the frames and relevant data from the data directory.
    """
    data_lst = []

    data_dir = os.path.join(args.data_dir, args.difficulty)
    if args.frame_end == -1:
        args.frame_end = len([name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)) and name.isdigit()])

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
        if args.downscale_factor > 1:
            ref_img = cv2.resize(
                ref_img,
                (args_ctrnet.width // args.downscale_factor, args_ctrnet.height // args.downscale_factor),
                interpolation=cv2.INTER_NEAREST,
            )
        # # Select the largest connected component as the mask
        # num_labels, labels_im = cv2.connectedComponents(ref_img)
        # largest_label = 1 + np.argmax([np.sum(labels_im == i) for i in range(1, num_labels)])
        # ref_img[labels_im != largest_label] = 0

        if ref_img is None:
            raise ValueError(f"No ref_img found in {frame_dir}")
        ref_mask = (ref_img / 255.0).astype(np.float32)
        ref_mask = th.tensor(ref_mask, requires_grad=False, dtype=th.float32).cuda()

        # Get reference key points
        kpts_path = os.path.join(frame_dir, "keypoints_" + XXXX + ".npy")
        if i == args.frame_start and not os.path.exists(kpts_path):
            raise ValueError(f"No keypoints file found for the first frame in {frame_dir}, initialization requires reference keypoints!")
        if i == args.frame_start:
            ref_keypoints = np.load(kpts_path)
        elif not args.use_contour_tip_net:
            ref_keypoints = get_reference_keypoints_auto(ref_mask_path, num_keypoints=2) # if not using ContourTipNet, use auto-detected keypoints for the rest of the frames
        elif os.path.exists(kpts_path):
            ref_keypoints = np.load(kpts_path)
        else:
            ref_keypoints = get_reference_keypoints_auto(ref_mask_path, num_keypoints=2) # if not using ContourTipNet, use auto-detected keypoints for the rest of the frames
        # print(np.load(kpts_path) if os.path.exists(kpts_path) else "No keypoints file found, using auto-detected keypoints.")
        # print(f"Reference keypoints for frame {i}: {ref_keypoints}")
        ref_keypoints = torch.tensor(ref_keypoints).squeeze().float().cuda()
        # ref_keypoints = ref_keypoints / args.downscale_factor

        # Get joint angles
        joint_path = os.path.join(frame_dir, "joint_" + XXXX + ".npy")
        jaw_path = os.path.join(frame_dir, "jaw_" + XXXX + ".npy")
        if not os.path.exists(joint_path):
            raise ValueError(f"No joint angles found in {frame_dir}")
        if not os.path.exists(jaw_path):
            raise ValueError(f"No jaw angles found in {frame_dir}")
        joints = np.load(joint_path)
        jaw = np.load(jaw_path)
        if jaw.ndim == 0:
            jaw = np.array([jaw]) # make it 1D
        joint_angles_np = np.array(
            [joints[4], joints[5], jaw[0] / 2, jaw[0] / 2], dtype=np.float32
        )
        joint_angles = th.tensor(
            joint_angles_np, requires_grad=False, dtype=th.float32
        ).cuda() 

        # Get optimized pose and joint angles
        optim_ctr_path = os.path.join(frame_dir, "optimized_ctr.npy")
        optim_joint_path = os.path.join(frame_dir, "optimized_joint_angles.npy")
        if not os.path.exists(optim_ctr_path):
            optim_ctr = None
        else:
            optim_ctr_np = np.load(optim_ctr_path)
            optim_ctr = th.tensor(
                optim_ctr_np, requires_grad=False, dtype=th.float32
            ).cuda()
        if not os.path.exists(optim_joint_path):
            optim_joint_angles = None
        else:
            optim_joint_angles_np = np.load(optim_joint_path)
            optim_joint_angles = th.tensor(
                optim_joint_angles_np, requires_grad=False, dtype=th.float32
            ).cuda()

        # If optimized joint angles are available, use them to replace the joint angle readings (synthetic trajectory noise is wrongly large)
        if optim_joint_angles is not None:
            joint_angles = optim_joint_angles.clone()

        data = {
            # "frame": frame,
            "ref_img": cv2.imread(ref_mask_path),
            "ref_mask": ref_mask.clone(),
            "ref_mask_path": ref_mask_path,
            "ref_keypoints": ref_keypoints.clone(),
            "joint_angles": joint_angles.clone(),
            "optim_ctr": optim_ctr.clone() if optim_ctr is not None else None,
            "optim_joint_angles": optim_joint_angles.clone() if optim_joint_angles is not None else None,
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


def initialization(model, init_data, mesh_files):
    """
    Use the method in origin_retracing.py to initialize the pose and joint angles.
    """
    ref_keypoints = init_data["ref_keypoints"]
    joint_angles = init_data["joint_angles"] if not args.use_prev_joint_angles else torch.zeros_like(init_data["joint_angles"]).cuda()
    joint_angles_read = joint_angles.clone() 
    model.get_joint_angles(joint_angles)
    # robot_mesh = robot_renderer.get_robot_mesh(joint_angles)

    # print(init_data["ref_mask_path"])

    bo_batch_problem = BayesOptBatchProblem(
        model=model,
        robot_renderer=robot_renderer,
        ref_mask_file=init_data["ref_mask_path"],
        ref_keypoints=ref_keypoints,
        fx=ctrnet_args.fx,
        fy=ctrnet_args.fy,
        px=ctrnet_args.px,
        py=ctrnet_args.py,
        batch_size=args.batch_size,
        ld1=3,
        ld2=3,
        ld3=3,
        batch_iters=args.batch_iters,
        lr=args.batch_opt_lr,
    )

    assert args.sample_number % args.batch_size == 0, "Sample number must be divisible by batch size."

    if args.use_bo_initializer:
        start_time = time.time()
        print("Using Bayesian optimization for initialization (without joint angle readings)...")

        # Optimize over [z, elevation, camera_roll_local, camera_roll, wrist pitch, wrist yaw, jaw1, jaw2]
        turbo = Turbo1(
            f=bo_batch_problem,
            lb=np.array([ 0.10, 90.-60.,   0.,   0.,  -1.5707,     -1.3963, 0.]),
            ub=np.array([ 0.17, 90.-30., 360., 360.,   0.,          1.3963, 1.5707]),
            n_init=args.batch_size,
            max_evals=args.sample_number,
            batch_size=args.batch_size,
            max_cholesky_size=1000,
            n_training_steps=50,
            verbose=True,
            min_cuda=1000,
            device='cuda',
            batch_eval=True, # Use batch evaluation
        )
        turbo.optimize()
        
        end_time = time.time()
        print(f"Bayesian optimization took {end_time - start_time:.2f} seconds.")

    else:
        lb = np.array([ 0.10, 90.-60.,   0.,   0., -1.5707, -1.3963, 0.])
        ub = np.array([ 0.17, 90.-30., 360., 360.,  0.,  1.3963, 1.5707 / 2])

        start_time = time.time()

        if not args.use_prev_joint_angles:
            print("Using random sampling for initialization (with current joint angle readings)...")

            for i in range(args.sample_number // args.batch_size):
                random_inputs = np.random.uniform(lb[:4], ub[:4], size=(args.batch_size, 4)).astype(np.float32)
                random_inputs = np.concatenate([random_inputs, joint_angles_read[:3].unsqueeze(0).expand(args.batch_size, -1).cpu().numpy()], axis=1) # append joint angle readings
                bo_batch_problem(random_inputs)

        else:
            print("Using random sampling for initialization (without joint angle readings)...")

            for i in range(args.sample_number // args.batch_size):
                random_inputs = np.random.uniform(lb, ub, size=(args.batch_size, 7)).astype(np.float32)
                bo_batch_problem(random_inputs)

        end_time = time.time()
        print(f"Random sampling took {end_time - start_time:.2f} seconds.")

    # Get the best cTr and joint angles from the optimization
    optimized_cTr_batch = bo_batch_problem.final_cTr_batch  # shape (N, 6)
    optimized_joint_angles_batch = bo_batch_problem.joint_angles_batch  # shape (N, num_joints)
    optimized_loss_batch = bo_batch_problem.final_loss_batch  # shape (N,)
    valid_mask = th.isfinite(optimized_loss_batch).to(device=optimized_loss_batch.device)
    if th.any(valid_mask):
        valid_losses = optimized_loss_batch[valid_mask]
        valid_cTrs = optimized_cTr_batch[valid_mask]
        valid_joint_angles = optimized_joint_angles_batch[valid_mask]
        best_idx = th.argmin(valid_losses)
        best_cTr = valid_cTrs[best_idx]
        joint_angles = valid_joint_angles[best_idx]
        best_loss = valid_losses[best_idx]
        print("==== Initialization results ====")
        print(f"  Best cTr = {best_cTr}")
        print(f"  Best joint angles = {joint_angles}")
        print(f"  Best loss (with inflated render loss) = {best_loss}")
    else:
        raise ValueError("No valid optimization results from initialization!")

    final_cTr_s = best_cTr

    # Clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()

    return final_cTr_s, joint_angles


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r".*copy construct from a tensor.*",
        category=UserWarning
    )

    args = parseArgs()
    ctrnet_args = parseCtRNetArgs()

    cache_filename = f"./data/cached_initialization/{args.data_dir.replace('/', '_')}_{args.difficulty.replace('/', '_')}_{args.frame_start}.pth"
    args.data_dir = os.path.join("./data", args.data_dir)

    ctrnet_args.use_nvdiffrast = args.use_nvdiffrast
    if ctrnet_args.use_nvdiffrast:
        print("Using NvDiffRast!")

    # Obtain the data
    data_lst, mesh_files = read_data(args, ctrnet_args)

    # Display the data (except for the images) for the first and last frames
    display_data(data_lst, 0)
    display_data(data_lst, -1)

    # Build the model
    model = CtRNet(ctrnet_args)
    robot_renderer = model.setup_robot_renderer(mesh_files)
    robot_renderer.set_mesh_visibility([True, True, True, True])

    # Initialize the model
    if args.no_cache or not os.path.exists(cache_filename):
        if args.data_dir.startswith("./data/synthetic"):
            print("[Using synthetic data, skipping optimization-based initialization...]")
            cTr = data_lst[0]["optim_ctr"].to(model.device)
            joint_angles = data_lst[0]["optim_joint_angles"].to(model.device)
        else:
            print("[Not using cache or the cache file is not found, running optimization-based initialization...]")
            cTr, joint_angles = initialization(
                model, data_lst[0], mesh_files
            )  
            torch.save({'cTr': cTr, 'joint_angles': joint_angles}, cache_filename)
        print(f"[The cTr and joint angles are saved in {cache_filename}.]")

    else:
        print(f"[Found cache file at {cache_filename}.]")
        cache = torch.load(cache_filename)
        cTr = cache['cTr'].to(model.device)
        joint_angles = cache['joint_angles'].to(model.device)

    # After initialization, if using joint angle readings, replace the initial joint angles
    # If the joint angle reading is flipped, rotate around beta by 180 degrees to resolve ambiguity
    if not args.use_prev_joint_angles:
        joint_angles_input = data_lst[0]["joint_angles"].to(model.device)

        # Check if wrist pitch and wrist yaw (first two joints) are flipped around 0
        wrist_pitch_yaw = joint_angles[:2]
        flipped_wrist_pitch_yaw = -joint_angles[:2]

        if torch.norm(wrist_pitch_yaw - joint_angles_input[:2]) > torch.norm(flipped_wrist_pitch_yaw - joint_angles_input[:2]):
            print("Flipping wrist pitch and yaw for left arm to resolve ambiguity.")
            joint_angles_input[:2] = flipped_wrist_pitch_yaw    
            cTr[:3] = axis_angle_to_mix_angle(cTr[:3].unsqueeze(0)).squeeze(0) # convert to mix angle representation for flipping
            cTr[1] += np.pi  # rotate around beta by 180 degrees
            cTr[:3] = mix_angle_to_axis_angle(cTr[:3].unsqueeze(0)).squeeze(0) # convert back to axis angle representation

        joint_angles = joint_angles_input.clone()

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

    if args.use_contour_tip_net:
        tip_length = 0.0096 # instead of 0.009
    else:
        tip_length = 0.009
    p_local1 = (
        torch.tensor([0.0, 0.0004, tip_length]) 
        .to(joint_angles.dtype)
        .to(model.device)
    )
    p_local2 = (
        torch.tensor([0.0, -0.0004, tip_length])
        .to(joint_angles.dtype)
        .to(model.device)
    )
  
    print(f"==== Tracking results ====")

    gc.collect()
    torch.cuda.empty_cache()

    with maybe_no_grad(args.searcher in ["CMA-ES", "XNES"]):
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

        # Track the rest of the frames
        # loss_lst, time_lst = [], []
        mask_lst, joint_angles_lst, ref_keypoints_lst, det_line_params_lst = [], [], [], []
        for i in range(1, len(data_lst)):
            # Get keypoints and cylinder parameters
            ref_keypoints, det_line_params = None, None
            if tracker.problem.kpts_loss or args.tracking_visualization:
                ref_keypoints = data_lst[i]["ref_keypoints"]

            # if len(ref_keypoints) < 2:
            #     # Add dummy keypoints if not enough keypoints detected
            #     ref_keypoints = torch.zeros((2, 2), dtype=torch.float32).cuda()

            mask_lst.append(data_lst[i]["ref_mask"])
            joint_angles_lst.append(data_lst[i]["joint_angles"])
            ref_keypoints_lst.append(ref_keypoints)
            det_line_params_lst.append(det_line_params)

        mask_lst = torch.stack(mask_lst, dim=0)
        joint_angles_lst = torch.stack(joint_angles_lst, dim=0)
        # ref_keypoints_lst = torch.stack(ref_keypoints_lst, dim=0) if ref_keypoints_lst[0] is not None else None
        # try:
        #     det_line_params_lst = torch.stack(det_line_params_lst, dim=0) if det_line_params_lst[0] is not None else None
        # except:
        #     det_line_params_lst = None

        # Track
        cTr_seq, joint_angles_seq, loss_seq, time_seq, overlay_seq = tracker.track_sequence(
            mask_lst=mask_lst,
            joint_angles_lst=joint_angles_lst,
            ref_keypoints_lst=ref_keypoints_lst,
            det_line_params_lst=det_line_params_lst,
            visualization=args.tracking_visualization
        )

        # Print tracking results for each frame
        for i in range(1, len(data_lst), 100):  # Print every 10 frames
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
        if len(loss_seq) > 10:
            avg_loss = np.mean(loss_seq[10:].cpu().numpy())
            avg_time = np.mean(time_seq[10:].cpu().numpy()) # remove the first frame as it is the initialization frame
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Average Time: {avg_time:.4f} seconds")
            print("Tracking completed.")

        # Save the results (label the setup including searcher, number of online iterations, etc.)
        # save_path = f"./pose_results/{args.difficulty.replace('/', '_')}_tracking_results.pth"
        data_label = f"{'surgpose' if args.data_dir.startswith('./data/surgpose') else 'synthetic'}_{args.difficulty.replace('/', '_')}"
        joint_str = 'wo_joint_angles' if args.use_prev_joint_angles else 'w_joint_angles'
        pts_loss_str = 'w_pts_loss' if args.use_pts_loss else 'wo_pts_loss'
        app_loss_str = 'w_app_loss' if args.app_weight > 0 else 'wo_app_loss'
        kpts_det_str = "wo_kpts_det"
        if args.use_pts_loss:
            if args.use_contour_tip_net:
                kpts_det_str = "w_tipnet"
            else:
                kpts_det_str = "w_opencv"
        filter_str = "no_filter" if not args.use_filter else args.filter_option
        save_path = f"/{data_label}.{args.searcher}.{args.online_iters}.{joint_str}.{pts_loss_str}.{kpts_det_str}.{app_loss_str}.{filter_str}.pth"
        save_path = "./pose_results" + save_path
        torch.save({'cTr': cTr_seq, 'joint_angles': joint_angles_seq, 'time': time_seq}, save_path)