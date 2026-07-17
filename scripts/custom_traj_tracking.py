import os
import sys
import argparse
import numpy as np
import kornia
import torch
import cv2
import time
import gc
import yaml
from typing import List, Tuple

# ------------------ Path bootstrap ------------------
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

LOCAL_MODULE_DIRS = [
    REPO_ROOT,
    os.path.join(REPO_ROOT, "SurgicalSAM2"),
    os.path.join(REPO_ROOT, "TuRBO"),
    os.path.join(REPO_ROOT, "ParticleFilter"),
]

for p in LOCAL_MODULE_DIRS:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.StereoCamera import StereoCamera
from core.RobotLink import *
from core.StereoCamera import *
from core.ParticleFilter import *
from core.probability_functions import *
from core.utils import *

from diffcali.models.CtRNet import CtRNet
from diffcali.utils.ui_utils import *
from diffcali.utils.skeleton_visualizer import RealTimeVideoWriter, SkeletonVisualizer
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
from diffcali.utils.link_utils import axisAngleToRotationMatrix
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
        # with torch.inference_mode():
            return func(*args, **kwargs)
    return wrapper

"""

BETA: better qualitatitve and quantitative results, not included in the paper
python scripts/custom_traj_tracking.py --use_nvdiffrast --use_bo_initializer --video_label bag1 --no_cache --use_full_joint_angles
python scripts/custom_traj_tracking.py --use_nvdiffrast --use_bo_initializer --video_label bag2 --no_cache --use_full_joint_angles
python scripts/custom_traj_tracking.py --use_nvdiffrast --use_bo_initializer --video_label bag3 --no_cache --use_full_joint_angles
python scripts/custom_traj_tracking.py --use_nvdiffrast --use_bo_initializer --video_label bag4 --no_cache --use_full_joint_angles
python scripts/custom_traj_tracking.py --use_nvdiffrast --use_bo_initializer --video_label bag5 --no_cache --use_full_joint_angles
python scripts/custom_traj_tracking.py --use_nvdiffrast --use_bo_initializer --video_label bag6 --no_cache --use_full_joint_angles


python scripts/custom_traj_tracking.py --use_nvdiffrast --use_bo_initializer --video_label bag1
python scripts/custom_traj_tracking.py --use_nvdiffrast --use_bo_initializer --video_label bag2 
python scripts/custom_traj_tracking.py --use_nvdiffrast --use_bo_initializer --video_label bag3
python scripts/custom_traj_tracking.py --use_nvdiffrast --use_bo_initializer --video_label bag4
python scripts/custom_traj_tracking.py --use_nvdiffrast --use_bo_initializer --video_label bag5
python scripts/custom_traj_tracking.py --use_nvdiffrast --use_bo_initializer --video_label bag6


python scripts/custom_quantitative_results.py
"""

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", type=str, default="urdfs/dVRK/meshes")
    parser.add_argument("--batch_opt_lr", type=float, default=3e-3)
    parser.add_argument("--single_opt_lr", type=float, default=5e-4) # if using gradient descent
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--dark_factor", type=float, default=0.7) # factor to darken the input image for better optimization performance (set to 1.0 for no darkening)
    parser.add_argument(
        "--batch_iters", type=int, default=100
    )  # Coarse steps per batch
    parser.add_argument(
        "--final_iters", type=int, default=100
    )  # Final single-sample refine using XNES / gradient descent
    parser.add_argument("--arm", type=str, default="psm2")
    parser.add_argument("--sample_number", type=int, default=2000)
    parser.add_argument("--use_bo_initializer", action="store_true") # Use Bayesian optimization for initialization (do not rely on joint angle readings)
    parser.add_argument("--use_nvdiffrast", action="store_true")
    
    parser.add_argument("--searcher", type=str, default="CMA-ES", choices=["CMA-ES", "XNES", "Gradient"])  # Search algorithm to use
    parser.add_argument("--online_iters", type=int, default=3)  # Number of iterations for online tracking
    
    parser.add_argument("--no_cache", action="store_true") # Use cached initialization

    parser.add_argument('--use_lumped_error_init', type=str2bool, default=False) # Whether to initialize each frame using w * T_A (T_A from joint+base FK)
    parser.add_argument('--interactive_prompts', type=str2bool, default=True) # Whether to interactively click SAM prompts on frame 0

    parser.add_argument("--use_full_joint_angles", action="store_true") # Whether to use all 7 joint angles for optimization, if false, only use the 3 visible joint angles and duplicate the jaw angle for the two jaws (since we are using symmetric jaw in tracking)

    parser.add_argument("--downscale_factor", type=int, default=2)
    parser.add_argument('--use_low_res_mesh', type=str2bool, default=True)

    parser.add_argument('--symmetric_jaw', type=str2bool, default=True)

    parser.add_argument('--use_render_loss', type=str2bool, default=True)
    parser.add_argument('--use_pts_loss', type=str2bool, default=True)

    parser.add_argument('--use_prev_joint_angles', type=str2bool, default=False)

    parser.add_argument('--rotation_parameterization', type=str, default="MixAngle", choices=["AxisAngle", "MixAngle"])

    parser.add_argument('--mse_weight', type=float, default=6.) #  originally 6.
    parser.add_argument('--dist_weight', type=float, default=0.) # originally 12e-7, turned off
    parser.add_argument('--app_weight', type=float, default=6e-6)
    parser.add_argument('--pts_weight', type=float, default=3e-3) # originally 5e-3, use 5e-5 for less pts loss weight

    parser.add_argument('--use_contour_tip_net', type=str2bool, default=True) # whether to use ContourTipNet for keypoint detection
    parser.add_argument('--contour_tip_net_path', type=str, default='./ContourTipNet/models/cnn_model.pth') # path to the ContourTipNet model

    parser.add_argument('--popsize', type=int, default=70)

    parser.add_argument('--filter_option', type=str, default="Kalman", choices=["None", "OneEuro", "OneEuro_orig", "Kalman"]) # which variables to filter

    parser.add_argument('--cos_reparams', type=str2bool, default=True) # whether to use cosine reparameterization (do not use for gradient-based methods), if not, simply clamp the angles within valid ranges

    parser.add_argument('--video_label', type=str, default='bag1') # path to the input video for online tracking
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

    if not args.use_prev_joint_angles:
        args.stdev_init[6:] /= 10. # if using joint angle readings, set the stdev for joint angles to a smaller value

    args.video_path = f'data/custom/{args.video_label}/left.mp4' # path to the input video for online tracking
    args.point_prompt_path = f'data/custom/{args.video_label}/{args.machine_label}_prompts.txt' # path to the point prompts for the first frame (format: x y label, where label is 1 for foreground and 0 for background)
    args.keypoints_path = f'data/custom/{args.video_label}/{args.machine_label}_keypoints.txt' # path to the keypoint prompts for the first frame (format: x y)
    args.joint_init_path = f'data/custom/{args.video_label}/{args.machine_label}_joint_init.txt' # Optional: path to the initial joint angles for the first frame (format: 3 visible joints)
    
    if args.searcher == "Gradient" and args.cos_reparams:
        raise ValueError("Cosine reparameterization is not compatible with gradient-based optimization, please set --cos_reparams False")

    return args


def parseCtRNetArgs():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.use_gpu = True
    args.trained_on_multi_gpus = False

    # Camera intrinsics
    # [ 1.02588223e+03,  0.0,  1.67919017e+02,
    #    0.0, 1.02588223e+03, 2.34152707e+02,
    #    0., 0., 1. ]

    # Setting for our custom data
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
    return v.lower() in ("yes", "true", "t", "1")


# def initialization(model, mask, kpts, joint_angles, mesh_files):
#     """
#     Use the method in origin_retracing.py to initialize the pose and joint angles.
#     """
#     ref_keypoints = torch.from_numpy(kpts).to(model.device).float()  # shape (num_kpts, 2)
#     joint_angles = torch.from_numpy(joint_angles).to(model.device).float() if joint_angles is not None else torch.zeros(4, device=model.device) # shape (4,)
#     joint_angles_read = joint_angles.clone() 
#     model.get_joint_angles(joint_angles)

#     ref_mask_path = f"./data/custom/{args.video_label}/{args.machine_label}_ref_mask.png"

#     # # Save the reference mask to the folder of the video
#     # mask_np = (mask.squeeze() > 0).cpu().numpy().astype(np.uint8) * 255
#     # cv2.imwrite(ref_mask_path, mask_np)

#     bo_batch_problem = BayesOptBatchProblem(
#         model=model,
#         robot_renderer=robot_renderer,
#         ref_mask_file=ref_mask_path,
#         ref_keypoints=ref_keypoints,
#         fx=ctrnet_args.fx,
#         fy=ctrnet_args.fy,
#         px=ctrnet_args.px,
#         py=ctrnet_args.py,
#         batch_size=args.batch_size,
#         ld1=3,
#         ld2=3,
#         ld3=3,
#         batch_iters=args.batch_iters,
#         lr=args.batch_opt_lr,
#     )

#     assert args.sample_number % args.batch_size == 0, "Sample number must be divisible by batch size."

#     if args.use_bo_initializer:
#         start_time = time.time()
#         print("Using Bayesian optimization for initialization (without joint angle readings)...")

#         # Optimize over [z, elevation, camera_roll_local, camera_roll, wrist pitch, wrist yaw, jaw1, jaw2]
#         turbo = Turbo1(
#             f=bo_batch_problem,
#             lb=np.array([ 0.10, 90.-60.,   0.,   0.,  -1.5707,     -1.3963, 0.]),
#             ub=np.array([ 0.17, 90.-30., 360., 360.,   0.,          1.3963, 1.5707]),
#             n_init=args.batch_size,
#             max_evals=args.sample_number,
#             batch_size=args.batch_size,
#             max_cholesky_size=1000,
#             n_training_steps=50,
#             verbose=True,
#             min_cuda=1000,
#             device='cuda',
#             batch_eval=True, # Use batch evaluation
#         )
#         turbo.optimize()
        
#         end_time = time.time()
#         print(f"Bayesian optimization took {end_time - start_time:.2f} seconds.")

#     else:
#         lb = np.array([ 0.10, 90.-60.,   0.,   0., -1.5707, -1.3963, 0.])
#         ub = np.array([ 0.17, 90.-30., 360., 360.,  0.,  1.3963, 1.5707 / 2])

#         start_time = time.time()

#         if joint_angles is not None and args.use_prev_joint_angles:
#             print("Using random sampling for initialization (with current joint angle readings)...")

#             for i in range(args.sample_number // args.batch_size):
#                 random_inputs = np.random.uniform(lb[:4], ub[:4], size=(args.batch_size, 4)).astype(np.float32)
#                 random_inputs = np.concatenate([random_inputs, joint_angles_read[:3].unsqueeze(0).expand(args.batch_size, -1).cpu().numpy()], axis=1) # append joint angle readings
#                 bo_batch_problem(random_inputs)

#         else:
#             print("Using random sampling for initialization (without joint angle readings)...")

#             for i in range(args.sample_number // args.batch_size):
#                 random_inputs = np.random.uniform(lb, ub, size=(args.batch_size, 7)).astype(np.float32)
#                 bo_batch_problem(random_inputs)

#         end_time = time.time()
#         print(f"Random sampling took {end_time - start_time:.2f} seconds.")

#     # Get the best cTr and joint angles from the optimization
#     optimized_cTr_batch = bo_batch_problem.final_cTr_batch  # shape (N, 6)
#     optimized_joint_angles_batch = bo_batch_problem.joint_angles_batch  # shape (N, num_joints)
#     optimized_loss_batch = bo_batch_problem.final_loss_batch  # shape (N,)
#     valid_mask = th.isfinite(optimized_loss_batch).to(device=optimized_loss_batch.device)
#     if th.any(valid_mask):
#         valid_losses = optimized_loss_batch[valid_mask]
#         valid_cTrs = optimized_cTr_batch[valid_mask]
#         valid_joint_angles = optimized_joint_angles_batch[valid_mask]
#         best_idx = th.argmin(valid_losses)
#         best_cTr = valid_cTrs[best_idx]
#         joint_angles = valid_joint_angles[best_idx]
#         best_loss = valid_losses[best_idx]
#         print("==== Initialization results ====")
#         print(f"  Best cTr = {best_cTr}")
#         print(f"  Best joint angles = {joint_angles}")
#         print(f"  Best loss (with inflated render loss) = {best_loss}")
#     else:
#         raise ValueError("No valid optimization results from initialization!")

#     final_cTr_s = best_cTr

#     # Clear CUDA cache
#     gc.collect()
#     torch.cuda.empty_cache()

#     return final_cTr_s, joint_angles


def initialization(cam_T_b, joint_angles, psm_arm):
    psm_arm.updateJointAngles(joint_angles)

    T_4 = np.dot(cam_T_b, psm_arm.baseToJointT[3]) # Get pose matrix of frame 4
    R, t_vec = T_4[:3, :3], T_4[:3, 3]
    R_ = torch.from_numpy(R).float().cuda()
    T_ = torch.from_numpy(t_vec).float().cuda()
    axis_angle = kornia.geometry.conversions.rotation_matrix_to_axis_angle(R_.unsqueeze(0)).squeeze(0) # Convert rotation matrix to axis-angle representation
    pose_vec = torch.cat([axis_angle, T_], dim=0)

    visible_joint_angles = torch.from_numpy(joint_angles).float().cuda()[-3:]
    visible_joint_angles[-1] /= 2.0
    visible_joint_angles = torch.cat([visible_joint_angles, visible_joint_angles[-1].unsqueeze(0)], dim=0) # Duplicate the jaw angle for the two jaws, since we are using symmetric jaw in tracking

    return pose_vec, visible_joint_angles


def collect_interactive_prompts(frame: np.ndarray, predictor, get_init_mask) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Collect SAM point prompts interactively on the first frame.

    Controls:
    - Left click: foreground point
    - Right click: background point
    - Enter: confirm prompts
    - r: reset prompts
    - q / Esc: abort
    """
    window_name = "Select SAM prompts (L=FG, R=BG, Enter=Start, r=Reset, q=Quit)"
    points: List[List[float]] = []
    labels: List[int] = []
    out_mask_logits = None

    base = frame.copy()
    vis = frame.copy()

    def draw(mask_logits=None):
        nonlocal vis
        vis = base.copy()

        if mask_logits is not None:
            mask = (mask_logits.squeeze() > 0).cpu().numpy().astype(np.uint8) * 255
            color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.7, color, 0.3, 0)

        for (x, y), lb in zip(points, labels):
            color = (0, 255, 0) if lb == 1 else (0, 0, 255)
            cv2.circle(vis, (int(x), int(y)), 5, color, -1)

        cv2.putText(vis, "Left click: FG | Right click: BG", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(vis, "Enter: start tracking | r: reset | q/Esc: quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    def update_mask():
        nonlocal out_mask_logits
        if len(points) == 0:
            out_mask_logits = None
            draw(None)
            return

        predictor.load_first_frame(base)
        pts_np = np.array(points, dtype=np.float32)
        lbs_np = np.array(labels, dtype=np.int64)
        _, _, out_mask_logits = get_init_mask(
            frame_idx=0,
            obj_id=0,
            points=pts_np,
            labels=lbs_np,
        )
        draw(out_mask_logits)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([float(x), float(y)])
            labels.append(1)
            update_mask()
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append([float(x), float(y)])
            labels.append(0)
            update_mask()

    predictor.load_first_frame(base)
    draw(None)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):  # Enter
            if np.sum(np.array(labels) == 1) == 0:
                print("Please add at least one foreground point before starting.")
                continue
            break
        if key == ord('r'):
            points.clear()
            labels.clear()
            predictor.load_first_frame(base)
            out_mask_logits = None
            draw(None)
        if key in (ord('q'), 27):  # q or Esc
            cv2.destroyWindow(window_name)
            raise RuntimeError("Interactive prompt selection aborted by user.")

    cv2.destroyWindow(window_name)

    if out_mask_logits is None:
        update_mask()

    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int64), out_mask_logits


def cTr_to_matrix(model: CtRNet, cTr: torch.Tensor) -> torch.Tensor:
    return model.cTr_to_pose_matrix(cTr.unsqueeze(0))[0]


def matrix_to_cTr(model: CtRNet, pose_matrix: torch.Tensor) -> torch.Tensor:
    return model.pose_matrix_to_cTr(pose_matrix.unsqueeze(0))[0]


def axis_to_optimizer_rot(cTr_axis: torch.Tensor, use_mix_angle: bool) -> torch.Tensor:
    """Convert axis-angle cTr to the optimizer rotation parameterization if needed."""
    cTr_opt = cTr_axis.clone()
    if use_mix_angle:
        cTr_opt[:3] = axis_angle_to_mix_angle(cTr_axis[:3].unsqueeze(0)).squeeze(0)
    return cTr_opt

if __name__ == "__main__":
    args = parseArgs()
    ctrnet_args = parseCtRNetArgs()

    if args.use_lumped_error_init:
        args.use_full_joint_angles = True
        args.no_cache = True
        print("Using lumped-error initialization: forcing --use_full_joint_angles True and disabling cache.")

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
    robot_renderer = model.setup_robot_renderer(mesh_files, downscale_factor=args.downscale_factor)
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

    # if args.use_contour_tip_net:
    #     tip_length = 0.0096 # instead of 0.009
    # else:
    #     tip_length = 0.009
    tip_length = 0.0096 # Set the tip length (distance from the last joint to the tip) to 9.6mm, which is more accurate according to the keypoint prompts
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

    # print(args.video_path)
    cap = cv2.VideoCapture(args.video_path)

    # ---- Live display only (no file outputs) ----
    save_video = False
    rt_writer = None

    bag_dir = f'data/custom/{args.video_label}'
    joint_angles_path = os.path.join(bag_dir, "joint_angles.yaml")
    # print(joint_angles_path)
    with open(joint_angles_path, 'r') as f:
        joint_angle_data = yaml.load(f, Loader=yaml.FullLoader)
        joint_angles_lst = [joint_angle_data[f"{i}"] for i in range(len(joint_angle_data))]
    joint_angles_np = np.array(joint_angles_lst)
    joint_angles_tensor = torch.from_numpy(joint_angles_np).float().to(model.device)[:,-3:]
    joint_angles_tensor[:, -1] /= 2.0 # Scale down the jaw angles to match the smaller jaw opening in our data
    joint_angles_tensor = torch.cat([joint_angles_tensor, joint_angles_tensor[:, -1].unsqueeze(1)], dim=1) # Duplicate the jaw angle for the two jaws, since we are using symmetric jaw in tracking
    full_joint_angles_init = joint_angles_np[0] # Get the initial joint angles for the first frame (shape (7,))

    # print(joint_angles_tensor.shape)
    print(f"Loaded {len(joint_angles_lst)} joint angle readings from {joint_angles_path}.")
    print(f"Video has {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames.")
    assert len(joint_angles_lst) <= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "The number of joint angle readings exceeds the number of frames in the video, please check the joint angle yaml file and the video."

    f = open(os.path.join("./data/custom/", 'handeye.yaml'), 'r')
    hand_eye_data = yaml.load(f, Loader=yaml.FullLoader)

    cam_T_b = np.eye(4)
    tvec_key = f"{args.machine_label}_tvec"
    rvec_key = f"{args.machine_label}_rvec"
    if tvec_key in hand_eye_data and rvec_key in hand_eye_data:
        cam_T_b[:-1, -1] = np.array(hand_eye_data[tvec_key]) / 1000.0
        cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(hand_eye_data[rvec_key])
        print(f"Using hand-eye calibration for {args.machine_label}.")
    else:
        cam_T_b[:-1, -1] = np.array(hand_eye_data['PSM1_tvec']) / 1000.0
        cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(hand_eye_data['PSM1_rvec'])
        print(f"[Warning] Missing {args.machine_label} hand-eye in handeye.yaml. Falling back to PSM1 hand-eye.")
    psm_arm = RobotLink(os.path.join("./data/custom/", "LND.json"))

    init_done = False
    seg_time_lst = []
    track_time_lst = []
    cTr_lst = []
    joint_lst = []
    w_lumped = torch.eye(4, dtype=torch.float32, device=model.device)

    for frame_idx in range(len(joint_angles_np)):
        ret, frame = cap.read()
        if not ret:
            print(f"End of video reached at frame {frame_idx}")
            break

        frame_shape_orig = (frame.shape[1], frame.shape[0]) # (width, height)
        frame = cv2.resize(frame, (ctrnet_args.width, ctrnet_args.height))

        if save_video and rt_writer is None:
            out_fps = 30.0
            out_path = os.path.join(f"./videos/cma_es_{args.video_label}_realtime_demo.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            rt_writer = RealTimeVideoWriter(
                path=out_path,
                fourcc=fourcc,
                fps=out_fps,
                frame_size=frame_shape_orig  # writing the final displayed resolution
            )

        # Make the frame darker to improve SAM segmentation results (since the original video is quite bright and has low contrast)
        frame = (frame * args.dark_factor).astype(np.uint8)

        if not init_done:
            if args.interactive_prompts:
                _, _, out_mask_logits = collect_interactive_prompts(frame, predictor, get_init_mask)
            else:
                init_pts = []
                init_lbs = []
                with open(args.point_prompt_path, "r") as f:
                    for line in f:
                        x, y, label = line.strip().split()
                        init_pts.append([float(x), float(y)])
                        init_lbs.append(int(label))
                init_pts = np.array(init_pts, dtype=np.float32)
                init_lbs = np.array([1 if lb == 1 else 0 for lb in init_lbs], dtype=np.int64)

                predictor.load_first_frame(frame)
                _, _, out_mask_logits = get_init_mask(
                    frame_idx=0,
                    obj_id=0,
                    points=init_pts,
                    labels=init_lbs,
                )

            mask = (out_mask_logits.squeeze() > 0).float()

            kpts = np.loadtxt(args.keypoints_path) if os.path.exists(args.keypoints_path) else None

            print("[Running initialization without cache...]")
            assert kpts is not None, "Keypoint prompts are required for optimization-based initialization. Please provide the keypoints in the specified path."

            cTr, joint_angles = initialization(
                cam_T_b=cam_T_b,
                joint_angles=joint_angles_np[0],
                psm_arm=psm_arm
            )

            # If using joint angle readings, replace the initial joint angles and fix potential left/right ambiguity.
            if not args.use_prev_joint_angles:
                joint_angles_input = joint_angles_tensor[frame_idx].clone()

                wrist_pitch_yaw = joint_angles[:2]
                flipped_wrist_pitch_yaw = -joint_angles[:2]

                if torch.norm(wrist_pitch_yaw - joint_angles_input[:2]) > torch.norm(flipped_wrist_pitch_yaw - joint_angles_input[:2]):
                    print("Flipping wrist pitch and yaw for left arm to resolve ambiguity.")
                    joint_angles_input[:2] = flipped_wrist_pitch_yaw
                    cTr[:3] = axis_angle_to_mix_angle(cTr[:3].unsqueeze(0)).squeeze(0)
                    cTr[1] += np.pi
                    cTr[:3] = mix_angle_to_axis_angle(cTr[:3].unsqueeze(0)).squeeze(0)

                joint_angles = joint_angles_input.clone()

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

            cTr, joint_angles, loss = tracker.track_frame(
                ref_mask=mask,
                joint_angles=joint_angles_tensor[frame_idx],
                is_init=True,
                keypoints=torch.from_numpy(kpts).to(model.device).float() if kpts is not None else None,
            )

            if args.use_lumped_error_init:
                cTr_A, _ = initialization(
                    cam_T_b=cam_T_b,
                    joint_angles=joint_angles_np[frame_idx],
                    psm_arm=psm_arm,
                )
                T_A = cTr_to_matrix(model, cTr_A)
                T_B = cTr_to_matrix(model, cTr)
                w_lumped = T_B @ torch.linalg.inv(T_A)

            cTr_lst.append(cTr)
            joint_lst.append(joint_angles)

            init_done = True

            # Clear CUDA cache
            gc.collect()
            torch.cuda.empty_cache()

        else:
            # Trackcing
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # with torch.inference_mode():
                out_obj_ids, out_mask_logits = get_next_mask(frame)
            torch.cuda.synchronize()
            end_time = time.time()
            seg_time_lst.append(end_time - start_time)

            mask = (out_mask_logits.squeeze() > 0).float()

            torch.cuda.synchronize()
            start_time = time.time()

            if args.use_full_joint_angles:
                cTr_fk, joint_angles_fk = initialization(
                    cam_T_b=cam_T_b, 
                    joint_angles=joint_angles_np[frame_idx],
                    psm_arm=psm_arm
                )

                if args.use_lumped_error_init:
                    T_A = cTr_to_matrix(model, cTr_fk)
                    T_init = w_lumped @ T_A
                    cTr_init_axis = matrix_to_cTr(model, T_init)
                    cTr_init = axis_to_optimizer_rot(cTr_init_axis, args.use_mix_angle)
                    cTr, joint_angles, loss = tracker.track_frame(
                        ref_mask=mask,
                        joint_angles=joint_angles_fk,
                        is_init=False,
                        keypoints=None,
                        cTr_init=cTr_init,
                    )
                    T_B = cTr_to_matrix(model, cTr)
                    w_lumped = T_B @ torch.linalg.inv(T_A)
                else:
                    cTr_init = axis_to_optimizer_rot(cTr_fk, args.use_mix_angle)
                    cTr, joint_angles, loss = tracker.track_frame(
                        ref_mask=mask,
                        joint_angles=joint_angles_fk,
                        is_init=False,
                        keypoints=None,
                        cTr_init=cTr_init,
                    )
            else:
                cTr, joint_angles, loss = tracker.track_frame(ref_mask=mask, joint_angles=joint_angles_tensor[frame_idx], is_init=False, keypoints=None)
            torch.cuda.synchronize()
            end_time = time.time()
            track_time_lst.append(end_time - start_time)

            cTr_lst.append(cTr)
            joint_lst.append(joint_angles)

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

        # Add elapsed wall-clock time overlay (proof it's real-time)
        if rt_writer is not None and rt_writer.t0 is not None:
            elapsed = time.perf_counter() - rt_writer.t0
            cv2.putText(
                blended,
                f"Wall-clock time: {elapsed:7.3f}s",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        # Real-time faithful write (duplicates frames if slow)
        if rt_writer is not None:
            rt_writer.write_realtime(blended)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if rt_writer is not None:
        rt_writer.release()
        print(f"Saved real-time faithful video to: {out_path}")

    # Compute the average FPS over the sequence (drop the first 10 frames to exclude initialization time)
    if len(seg_time_lst) > 10 and len(track_time_lst) > 10:
        avg_seg_time = sum(seg_time_lst[10:]) / len(seg_time_lst[10:])
        avg_track_time = sum(track_time_lst[10:]) / len(track_time_lst[10:])
        avg_time = avg_seg_time + avg_track_time
        fps = 1 / avg_time if avg_time > 0 else 0
        print(f"Average FPS (excluding first 10 frames): {fps:.2f}")
    else:
        print("Not enough frames to compute average FPS excluding initialization.")

    print("Live tracking finished. No tracking results were saved to disk.")
