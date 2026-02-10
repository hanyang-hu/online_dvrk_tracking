import os
import sys
import argparse
import numpy as np
import kornia
import torch
import cv2
import time
import gc
import torch.nn.functional as F

# ------------------ Path bootstrap ------------------
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

LOCAL_MODULE_DIRS = [
    REPO_ROOT,
    os.path.join(REPO_ROOT, "SurgicalSAM2"),
    os.path.join(REPO_ROOT, "TuRBO"),
]

for p in LOCAL_MODULE_DIRS:
    if p not in sys.path:
        sys.path.insert(0, p)

from diffcali.models.CtRNet import CtRNet
from diffcali.utils.ui_utils import *
from diffcali.utils.skeleton_visualizer import SkeletonVisualizer
from diffcali.eval_dvrk.batch_optimize import BatchOptimize, HeterogeneousBatchOptimize
from diffcali.eval_dvrk.optimize import Optimize
from diffcali.eval_dvrk.black_box_optimize import BlackBoxOptimize
from diffcali.utils.angle_transform_utils import (
    enforce_axis_angle_consistency,
    enforce_quaternion_consistency,
    mix_angle_to_axis_angle,
    axis_angle_to_mix_angle,
)
from diffcali.utils.contour_tip_net import ContourTipNet
from diffcali.eval_dvrk.trackers import Tracker
from TuRBO.turbo.turbo_1 import Turbo1
from diffcali.eval_dvrk.black_box_optimize import BayesOptBatchProblem
from evotorch.tools.misc import RealOrVector

from contextlib import contextmanager

from sam2.build_sam import build_sam2_camera_predictor


@contextmanager
def maybe_no_grad(condition: bool):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


def sam2_inference(func):
    """Run function in torch.inference_mode and bfloat16 autocast (GPU)."""
    def wrapper(*args, **kwargs):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return func(*args, **kwargs)
    return wrapper


def parseArgs():
    """
    python scripts/online_tracking.py --sample_number 1000 --use_nvdiffrast --use_bo_initializer --video_label 000000 --machine_label PSM1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", type=str, default="urdfs/dVRK/meshes")
    parser.add_argument("--batch_opt_lr", type=float, default=3e-3)
    parser.add_argument("--single_opt_lr", type=float, default=5e-4) # if using gradient descent
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument(
        "--batch_iters", type=int, default=100
    )  # Coarse steps per batch
    parser.add_argument(
        "--final_iters", type=int, default=100
    )  # Final single-sample refine using XNES / gradient descent
    parser.add_argument("--arm", type=str, default="psm2")
    parser.add_argument("--sample_number", type=int, default=1000)
    parser.add_argument("--use_bo_initializer", action="store_true") # Use Bayesian optimization for initialization (do not rely on joint angle readings)
    parser.add_argument("--use_nvdiffrast", action="store_true")
    
    parser.add_argument("--searcher", type=str, default="CMA-ES", choices=["CMA-ES", "XNES", "Gradient"])  # Search algorithm to use
    parser.add_argument("--online_iters", type=int, default=3)  # Number of iterations for online tracking
    
    parser.add_argument("--no_cache", action="store_true") # Use cached initialization

    parser.add_argument("--downscale_factor", type=int, default=2)
    parser.add_argument('--use_low_res_mesh', type=str2bool, default=True)

    parser.add_argument('--symmetric_jaw', type=str2bool, default=True)

    parser.add_argument('--use_render_loss', type=str2bool, default=True)
    parser.add_argument('--use_pts_loss', type=str2bool, default=True)

    parser.add_argument('--use_prev_joint_angles', type=str2bool, default=True)

    parser.add_argument('--rotation_parameterization', type=str, default="AxisAngle", choices=["AxisAngle", "MixAngle"])

    parser.add_argument('--mse_weight', type=float, default=6.) #  originally 6.
    parser.add_argument('--dist_weight', type=float, default=0.) # originally 12e-7, turned off
    parser.add_argument('--app_weight', type=float, default=6e-6)
    parser.add_argument('--pts_weight', type=float, default=3e-3) # originally 5e-3, use 5e-5 for less pts loss weight

    parser.add_argument('--use_contour_tip_net', type=str2bool, default=True) # whether to use ContourTipNet for keypoint detection
    parser.add_argument('--contour_tip_net_path', type=str, default='./ContourTipNet/models/cnn_model.pth') # path to the ContourTipNet model

    parser.add_argument('--popsize', type=int, default=70)

    parser.add_argument('--filter_option', type=str, default="Kalman", choices=["None", "OneEuro", "OneEuro_orig", "Kalman"]) # which variables to filter

    parser.add_argument('--cos_reparams', type=str2bool, default=True) # whether to use cosine reparameterization (do not use for gradient-based methods), if not, simply clamp the angles within valid ranges

    parser.add_argument('--video_label', type=str, default='000000') # path to the input video for online tracking
    parser.add_argument('--machine_label', type=str, default='PSM3') # machine label for selecting the video and initial prompts

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

    args.video_path = f'data/online_videos/{args.video_label}/video.mp4' # path to the input video for online tracking
    args.point_prompt_path = f'data/online_videos/{args.video_label}/{args.machine_label}_prompts.txt' # path to the point prompts for the first frame (format: x y label, where label is 1 for foreground and 0 for background)
    args.keypoints_path = f'data/online_videos/{args.video_label}/{args.machine_label}_keypoints.txt' # path to the keypoint prompts for the first frame (format: x y)
    args.joint_init_path = f'data/online_videos/{args.video_label}/{args.machine_label}_joint_init.txt' # Optional: path to the initial joint angles for the first frame (format: 3 visible joints)

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
    return v.lower() in ("yes", "true", "t", "1")


def initialization(model, mask, kpts, joint_angles, mesh_files):
    """
    Use the method in origin_retracing.py to initialize the pose and joint angles.
    """
    ref_keypoints = torch.from_numpy(kpts).to(model.device).float()  # shape (num_kpts, 2)
    joint_angles = torch.from_numpy(joint_angles).to(model.device).float() if joint_angles is not None else torch.zeros(4, device=model.device) # shape (4,)
    joint_angles_read = joint_angles.clone() 
    model.get_joint_angles(joint_angles)

    ref_mask_path = f"./data/online_videos/{args.video_label}/{args.machine_label}_ref_mask.png"

    # # Save the reference mask to the folder of the video
    # mask_np = (mask.squeeze() > 0).cpu().numpy().astype(np.uint8) * 255
    # cv2.imwrite(ref_mask_path, mask_np)

    bo_batch_problem = BayesOptBatchProblem(
        model=model,
        robot_renderer=robot_renderer,
        ref_mask_file=ref_mask_path,
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

        if joint_angles is not None and args.use_prev_joint_angles:
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
    args = parseArgs()
    ctrnet_args = parseCtRNetArgs()

    # Load rendering model
    ctrnet_args.use_nvdiffrast = args.use_nvdiffrast
    if ctrnet_args.use_nvdiffrast:
        print("Using NvDiffRast!")

    model = CtRNet(ctrnet_args)

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
    robot_renderer = model.setup_robot_renderer(mesh_files)
    robot_renderer.set_mesh_visibility([True, True, True, True])

    # Specify camera intrinsics and keypoints
    intr = torch.tensor(
        [
            [ctrnet_args.fx, 0, ctrnet_args.px], 
            [0, ctrnet_args.fy, ctrnet_args.py], 
            [0, 0, 1]
        ],
        device="cuda",
        dtype=torch.float32,
    )

    if args.use_contour_tip_net:
        tip_length = 0.0096 # instead of 0.009
    else:
        tip_length = 0.009
    p_local1 = (
        torch.tensor([0.0, 0.0004, tip_length]) 
        .to(torch.float32)
        .to(model.device)
    )
    p_local2 = (
        torch.tensor([0.0, -0.0004, tip_length])
        .to(torch.float32)
        .to(model.device)
    )

    # Load Surgical SAM 2 predictor
    predictor = build_sam2_camera_predictor(
        "./configs/sam2.1/sam2.1_hiera_s.yaml",
        "./SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth",
        vos_optimized=True,
    )

    # Initialize the skeleton visualizer
    skeleton_visualizer = SkeletonVisualizer(model, ctrnet_args, args, intr, p_local1, p_local2, thickness=5)

    @sam2_inference
    def get_init_mask(*a, **k):
        return predictor.add_new_points(*a, **k)
    
    @sam2_inference
    def get_next_mask(*a, **k):
        return predictor.track(*a, **k)

    # Load initial point prompts and keypoints (two tips)
    init_pts = []
    init_lbs = []
    with open(args.point_prompt_path, "r") as f:
        for line in f:
            x, y, label = line.strip().split()
            init_pts.append([float(x), float(y)])
            init_lbs.append(int(label))
    init_pts = np.array(init_pts, dtype=np.float32)
    init_lbs = np.array([1 if lb == 1 else 0 for lb in init_lbs], dtype=np.int64) # Ensure labels are binary (1 for foreground, 0 for background)

    cap = cv2.VideoCapture(args.video_path)

    init_done = False
    seg_time_lst = []
    track_time_lst = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_shape_orig = (frame.shape[1], frame.shape[0]) # (width, height)
        frame = cv2.resize(frame, (ctrnet_args.width, ctrnet_args.height))

        if not init_done:
            predictor.load_first_frame(frame)

            _, _, out_mask_logits = get_init_mask(
                frame_idx=0,
                obj_id=0,
                points=init_pts,
                labels=init_lbs,
            )

            mask = (out_mask_logits.squeeze() > 0).float()

            cache_filename = f"./data/online_videos/{args.video_label}/{args.machine_label}_init_cache.pth"

            kpts = np.loadtxt(args.keypoints_path) if os.path.exists(args.keypoints_path) else None

            if args.no_cache or not os.path.exists(cache_filename):
                print("[Not using cache or the cache file is not found, running optimization-based initialization...]")
                assert kpts is not None, "Keypoint prompts are required for optimization-based initialization. Please provide the keypoints in the specified path."
                joint_angles = np.loadtxt(args.joint_init_path) if os.path.exists(args.joint_init_path) else None

                cTr, joint_angles = initialization(
                    model, mask, kpts, joint_angles, mesh_files
                )  
                torch.save({'cTr': cTr, 'joint_angles': joint_angles}, cache_filename)
                print(f"[The cTr and joint angles are saved in {cache_filename}.]")
            else:
                print(f"[Found cache file at {cache_filename}.]")
                cache = torch.load(cache_filename)
                cTr = cache['cTr'].to(model.device)
                joint_angles = cache['joint_angles'].to(model.device)

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

            cTr, joint_angles, loss = tracker.track_frame(ref_mask=mask, joint_angles=None, is_init=True, keypoints=torch.from_numpy(kpts).to(model.device).float() if kpts is not None else None)

            init_done = True

            # Clear CUDA cache
            gc.collect()
            torch.cuda.empty_cache()

        else:
            # Trackcing
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_obj_ids, out_mask_logits = get_next_mask(frame)
            torch.cuda.synchronize()
            end_time = time.time()
            seg_time_lst.append(end_time - start_time)

            mask = (out_mask_logits.squeeze() > 0).float()

            torch.cuda.synchronize()
            start_time = time.time()
            cTr, joint_angles, loss = tracker.track_frame(ref_mask=mask, joint_angles=None, is_init=False, keypoints=None)
            torch.cuda.synchronize()
            end_time = time.time()
            track_time_lst.append(end_time - start_time)

        mask = (out_mask_logits.squeeze() > 0).cpu().numpy().astype(np.uint8) * 255
        color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.7, color, 0.3, 0)

        blended = skeleton_visualizer.plot_skeleton_overlay(blended, cTr, joint_angles)
        blended = cv2.resize(blended, frame_shape_orig)

        if len(seg_time_lst) > 10 and len(track_time_lst) > 10:
            avg_time = sum(seg_time_lst[-10:]) / len(seg_time_lst[-10:]) + sum(track_time_lst[-10:]) / len(track_time_lst[-10:])
            loss = loss.item() if isinstance(loss, torch.Tensor) else loss
            fps = 1 / avg_time if avg_time > 0 else 0
            cv2.putText(
                blended,
                f"Loss: {loss:.4f} | FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        cv2.imshow("frame", blended)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Compute the average FPS over the sequence (drop the first 10 frames to exclude initialization time)
    if len(seg_time_lst) > 10 and len(track_time_lst) > 10:
        avg_seg_time = sum(seg_time_lst[10:]) / len(seg_time_lst[10:])
        avg_track_time = sum(track_time_lst[10:]) / len(track_time_lst[10:])
        avg_time = avg_seg_time + avg_track_time
        fps = 1 / avg_time if avg_time > 0 else 0
        print(f"Average FPS (excluding first 10 frames): {fps:.2f}")
    else:
        print("Not enough frames to compute average FPS excluding initialization.")
