import os
import sys
import argparse
import numpy as np
import kornia
import torch
import cv2
import time
import gc
import glob

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
from diffcali.utils.skeleton_visualizer import SkeletonVisualizer, RealTimeVideoWriter
from diffcali.utils.angle_transform_utils import (
    enforce_axis_angle_consistency,
    enforce_quaternion_consistency,
    mix_angle_to_axis_angle,
    axis_angle_to_mix_angle,
)
from diffcali.utils.contour_tip_net import ContourTipNet

# NOTE: switch to BiManualTracker (bi-manual tracking)
from diffcali.eval_dvrk.trackers import BiManualTracker

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


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")


def parseArgs():
    """
    Bi-manual online tracking:
      - assumes SAME video for both arms
      - reads caches separately:
          ./data/online_videos/{video_label}/PSM3_init_cache.pth
          ./data/online_videos/{video_label}/PSM1_init_cache.pth
      - reads prompts / keypoints separately:
          ./data/online_videos/{video_label}/PSM3_prompts.txt
          ./data/online_videos/{video_label}/PSM1_prompts.txt
          ./data/online_videos/{video_label}/PSM3_keypoints.txt
          ./data/online_videos/{video_label}/PSM1_keypoints.txt
      - reads joint readings separately from SurgPose folders and stacks them:
          ./data/surgpose/{video_label}/PSM3/{frame}/...
          ./data/surgpose/{video_label}/PSM1/{frame}/...
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", type=str, default="urdfs/dVRK/meshes")
    parser.add_argument("--batch_opt_lr", type=float, default=3e-3)
    parser.add_argument("--single_opt_lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--batch_iters", type=int, default=100)
    parser.add_argument("--final_iters", type=int, default=100)
    parser.add_argument("--arm", type=str, default="psm2")
    parser.add_argument("--sample_number", type=int, default=1000)
    parser.add_argument("--use_bo_initializer", action="store_true")  # unused in this bi-manual realtime script
    parser.add_argument("--use_nvdiffrast", action="store_true")

    parser.add_argument("--searcher", type=str, default="CMA-ES", choices=["CMA-ES", "XNES", "Gradient"])
    parser.add_argument("--online_iters", type=int, default=3)

    # Cache is assumed available (per your request); keep flag for compatibility but we won't run TuRBO anyway.
    parser.add_argument("--no_cache", action="store_true")

    parser.add_argument("--downscale_factor", type=int, default=2)
    parser.add_argument("--use_low_res_mesh", type=str2bool, default=True)

    parser.add_argument("--symmetric_jaw", type=str2bool, default=True)

    parser.add_argument("--use_render_loss", type=str2bool, default=True)
    parser.add_argument("--use_pts_loss", type=str2bool, default=True)

    parser.add_argument("--use_prev_joint_angles", type=str2bool, default=True)

    parser.add_argument("--rotation_parameterization", type=str, default="MixAngle", choices=["AxisAngle", "MixAngle"])

    parser.add_argument("--mse_weight", type=float, default=6.0)
    parser.add_argument("--dist_weight", type=float, default=0.0)
    parser.add_argument("--app_weight", type=float, default=6e-6)
    parser.add_argument("--pts_weight", type=float, default=3e-3)

    parser.add_argument("--use_contour_tip_net", type=str2bool, default=True)
    parser.add_argument("--contour_tip_net_path", type=str, default="./ContourTipNet/models/cnn_model.pth")

    parser.add_argument("--popsize", type=int, default=70)

    parser.add_argument("--filter_option", type=str, default="Kalman", choices=["None", "OneEuro", "OneEuro_orig", "Kalman"])
    parser.add_argument("--cos_reparams", type=str2bool, default=True)

    parser.add_argument("--video_label", type=str, default="000000")
    # NEW: explicitly set which machine IDs to track together
    parser.add_argument("--machine_left", type=str, default="PSM3")   # left arm
    parser.add_argument("--machine_right", type=str, default="PSM1")  # right arm

    # --- Bi-manual defaults from reference ---
    parser.add_argument("--separate_loss", type=str2bool, default=True)
    parser.add_argument("--soft_separation", type=str2bool, default=False)
    parser.add_argument("--share_depth_buffer", type=str2bool, default=True)
    parser.add_argument("--use_bd_cmaes", type=str2bool, default=True)

    # Base stdev for ONE arm (10 dims: 6 pose + 4 joints)
    stdev_init = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=torch.float32).cuda()
    stdev_init[:3] *= torch.tensor([1e-2, 1e-1, 1e-2], dtype=torch.float32).cuda()
    stdev_init[3:6] *= 1e-3
    stdev_init[6:] *= 5e-2
    stdev_init = stdev_init.detach()
    parser.add_argument("--stdev_init", type=RealOrVector, default=stdev_init)

    parser.add_argument("--log_interval", type=int, default=1000)
    args = parser.parse_args()

    args.use_filter = False if args.filter_option == "None" else True
    args.use_mix_angle = (args.rotation_parameterization == "MixAngle")

    if args.rotation_parameterization == "AxisAngle":
        args.stdev_init[:3] = 1e-1

    # per-arm tuning
    args.stdev_init[6] *= 2
    args.stdev_init[7] *= 2
    args.stdev_init[8:] *= 2

    # if symmetric jaw, many trackers only optimize one jaw
    # (your BiManualTracker / problem likely handles this internally; we keep the stdev as-is here)

    # Bi-manual: duplicate stdev for two arms (match reference behavior)
    args.stdev_init = torch.cat([args.stdev_init, args.stdev_init], dim=0)

    if args.searcher == "Gradient" and args.cos_reparams:
        raise ValueError("Cosine reparameterization is not compatible with gradient-based optimization, set --cos_reparams False")

    args.video_path = f"data/online_videos/{args.video_label}/video.mp4"

    # prompts / kpts / cache per machine
    args.left_point_prompt_path = f"data/online_videos/{args.video_label}/{args.machine_left}_prompts.txt"
    args.right_point_prompt_path = f"data/online_videos/{args.video_label}/{args.machine_right}_prompts.txt"
    args.left_keypoints_path = f"data/online_videos/{args.video_label}/{args.machine_left}_keypoints.txt"
    args.right_keypoints_path = f"data/online_videos/{args.video_label}/{args.machine_right}_keypoints.txt"

    args.left_cache_path = f"./data/online_videos/{args.video_label}/{args.machine_left}_init_cache.pth"
    args.right_cache_path = f"./data/online_videos/{args.video_label}/{args.machine_right}_init_cache.pth"

    return args


def parseCtRNetArgs():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.use_gpu = True
    args.trained_on_multi_gpus = False

    # SurgPose camera
    args.height = 986 // 2
    args.width = 1400 // 2
    args.fx, args.fy, args.px, args.py = (
        1811.910046453570 / 2,
        1809.640734154330 / 2,
        588.5594517681759 / 2,
        477.3975900383616 / 2,
    )

    args.scale = 1.0
    args.width = int(args.width * args.scale)
    args.height = int(args.height * args.scale)
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale

    return args


def load_point_prompts(path):
    pts, lbs = [], []
    with open(path, "r") as f:
        for line in f:
            x, y, label = line.strip().split()
            pts.append([float(x), float(y)])
            lbs.append(int(label))
    pts = np.array(pts, dtype=np.float32)
    lbs = np.array([1 if lb == 1 else 0 for lb in lbs], dtype=np.int64)
    return pts, lbs


def read_joint_angles_sequence(video_label: str, machine_label: str):
    """
    Reads joint + jaw from ./data/surgpose/{video_label}/{machine_label}/{frame}/...
    Returns list[torch.Tensor] each (4,) on CUDA.
    """
    joint_angle_readings = []
    data_dir = os.path.join("./data/surgpose/", video_label, machine_label)
    frame_end = len(
        [
            name
            for name in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, name)) and name.isdigit()
        ]
    )

    for i in range(0, frame_end):
        frame_dir = os.path.join(data_dir, f"{i}")

        mask_lst = glob.glob(os.path.join(frame_dir, "*.png"))
        if len(mask_lst) == 0:
            raise ValueError(f"No mask found in {frame_dir}")
        if len(mask_lst) > 1:
            raise ValueError(f"Multiple masks found in {frame_dir}")

        mask_path = mask_lst[0]
        XXXX = mask_path.split("/")[-1].split(".")[0][1:]

        joint_path = os.path.join(frame_dir, "joint_" + XXXX + ".npy")
        jaw_path = os.path.join(frame_dir, "jaw_" + XXXX + ".npy")
        if not os.path.exists(joint_path):
            raise ValueError(f"No joint angles found in {frame_dir}")
        if not os.path.exists(jaw_path):
            raise ValueError(f"No jaw angles found in {frame_dir}")

        joints = np.load(joint_path)
        jaw = np.load(jaw_path)
        if jaw.ndim == 0:
            jaw = np.array([jaw])

        joint_angles_np = np.array([joints[4], joints[5], jaw[0] / 2, jaw[0] / 2], dtype=np.float32)
        joint_angles = torch.tensor(joint_angles_np, requires_grad=False, dtype=torch.float32).cuda()
        joint_angle_readings.append(joint_angles)

    return joint_angle_readings


if __name__ == "__main__":
    args = parseArgs()
    ctrnet_args = parseCtRNetArgs()

    # Load rendering model
    ctrnet_args.use_nvdiffrast = args.use_nvdiffrast
    if ctrnet_args.use_nvdiffrast:
        print("Using NvDiffRast!")

    model = CtRNet(ctrnet_args)

    # Mesh files
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

    # Read joint angle files for BOTH arms, then stack per-frame into (2, 4)
    joint_left_seq = read_joint_angles_sequence(args.video_label, args.machine_left)   # (T,) of (4,)
    joint_right_seq = read_joint_angles_sequence(args.video_label, args.machine_right) # (T,) of (4,)
    if len(joint_left_seq) != len(joint_right_seq):
        raise ValueError(f"Frame count mismatch: left={len(joint_left_seq)} right={len(joint_right_seq)}")

    joint_angle_readings = [
        torch.stack([joint_left_seq[i], joint_right_seq[i]], dim=0)  # (2, 4)
        for i in range(len(joint_left_seq))
    ]
    print(f"Loaded joint angles for {len(joint_angle_readings)} frames (bi-manual).")

    # Camera intrinsics
    intr = torch.tensor(
        [
            [ctrnet_args.fx, 0, ctrnet_args.px],
            [0, ctrnet_args.fy, ctrnet_args.py],
            [0, 0, 1],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    # Tip locals
    tip_length = 0.0096
    p_local1 = torch.tensor([0.0, 0.0004, tip_length], dtype=torch.float32, device=model.device)
    p_local2 = torch.tensor([0.0, -0.0004, tip_length], dtype=torch.float32, device=model.device)

    # Load Surgical SAM2 predictor
    predictor = build_sam2_camera_predictor(
        "./configs/sam2.1/sam2.1_hiera_s.yaml",
        "./SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth",
        vos_optimized=True,
    )

    # Skeleton visualizers (draw each arm sequentially)
    skeleton_visualizer_left = SkeletonVisualizer(model, ctrnet_args, args, intr, p_local1, p_local2, thickness=5)
    skeleton_visualizer_right = SkeletonVisualizer(model, ctrnet_args, args, intr, p_local1, p_local2, thickness=5)

    @sam2_inference
    def get_init_mask(*a, **k):
        return predictor.add_new_points(*a, **k)

    @sam2_inference
    def get_next_mask(*a, **k):
        return predictor.track(*a, **k)

    # Load initial point prompts / keypoints for BOTH arms
    init_pts_left, init_lbs_left = load_point_prompts(args.left_point_prompt_path)
    init_pts_right, init_lbs_right = load_point_prompts(args.right_point_prompt_path)

    kpts_left = np.loadtxt(args.left_keypoints_path) if os.path.exists(args.left_keypoints_path) else None
    kpts_right = np.loadtxt(args.right_keypoints_path) if os.path.exists(args.right_keypoints_path) else None

    # Keypoints tensor for init (optional)
    init_keypoints = None
    if kpts_left is not None and kpts_right is not None:
        kpts_left_t = torch.from_numpy(np.array(kpts_left, dtype=np.float32)).to(model.device).float().reshape(-1, 2)
        kpts_right_t = torch.from_numpy(np.array(kpts_right, dtype=np.float32)).to(model.device).float().reshape(-1, 2)
        # pad to >=2 if needed (common in your reference)
        if kpts_left_t.shape[0] < 2:
            kpts_left_t = torch.cat([kpts_left_t, kpts_left_t[-1:].repeat(2 - kpts_left_t.shape[0], 1)], dim=0)
        if kpts_right_t.shape[0] < 2:
            kpts_right_t = torch.cat([kpts_right_t, kpts_right_t[-1:].repeat(2 - kpts_right_t.shape[0], 1)], dim=0)
        init_keypoints = torch.stack([kpts_left_t, kpts_right_t], dim=0)  # (2, K, 2)

    # Load caches for BOTH arms (assumed always available per your request)
    if (not os.path.exists(args.left_cache_path)) or (not os.path.exists(args.right_cache_path)):
        raise FileNotFoundError(
            f"Missing cache(s). Expected:\n  {args.left_cache_path}\n  {args.right_cache_path}\n"
            "Per your instruction this script assumes caches exist; generate them with your single-arm pipeline first."
        )

    cache_left = torch.load(args.left_cache_path)
    cache_right = torch.load(args.right_cache_path)

    cTr_left = cache_left["cTr"].to(model.device)          # (6,)
    joint_init_left = cache_left["joint_angles"].to(model.device)   # (4,)

    cTr_right = cache_right["cTr"].to(model.device)        # (6,)
    joint_init_right = cache_right["joint_angles"].to(model.device) # (4,)

    init_cTr = torch.stack([cTr_left, cTr_right], dim=0)                 # (2, 6)
    init_joint_angles = torch.stack([joint_init_left, joint_init_right], dim=0)  # (2, 4)

    # Build bi-manual tracker
    tracker = BiManualTracker(
        model=model,
        robot_renderer=robot_renderer,
        init_cTr=init_cTr,
        init_joint_angles=init_joint_angles,
        num_iters=args.online_iters,
        stdev_init=args.stdev_init,
        intr=intr,
        p_local1=p_local1,
        p_local2=p_local2,
        searcher=args.searcher,
        args=args,
    )

    cap = cv2.VideoCapture(args.video_path)

    # ---- Real-time faithful recording setup ----
    save_video = True
    out_fps = 30.0
    if not os.path.exists("./videos/"):
        os.makedirs("./videos/")
    searcher_name = "cma_es" if args.searcher == "CMA-ES" else "gradient" if args.searcher == "Gradient" else "xnes"
    out_path = os.path.join(
        f"./videos/{searcher_name}_{args.video_label}_BI_MANUAL_{args.machine_left}_{args.machine_right}_realtime_demo_surgpose.mp4"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rt_writer = None

    init_done = False
    cTr_lst, joint_angles_lst = [], []
    seg_time_lst = []
    track_time_lst = []
    frame_idx = 0

    while frame_idx < len(joint_angle_readings):
        ret, frame = cap.read()
        if not ret:
            break

        frame_shape_orig = (frame.shape[1], frame.shape[0])  # (width, height)
        frame = cv2.resize(frame, (ctrnet_args.width, ctrnet_args.height))

        if save_video and rt_writer is None:
            rt_writer = RealTimeVideoWriter(
                path=out_path,
                fourcc=fourcc,
                fps=out_fps,
                frame_size=frame_shape_orig,
            )

        if not init_done:
            predictor.load_first_frame(frame)

            # Add prompts for BOTH objects on the first frame
            # obj_id=0 -> left, obj_id=1 -> right (consistent ordering below)
            _, _, out_mask_logits_left = get_init_mask(
                frame_idx=0,
                obj_id=0,
                points=init_pts_left,
                labels=init_lbs_left,
            )
            _, _, out_mask_logits_right = get_init_mask(
                frame_idx=0,
                obj_id=1,
                points=init_pts_right,
                labels=init_lbs_right,
            )

            # mask_left = (out_mask_logits_left.squeeze() > 0).float()
            # mask_right = (out_mask_logits_right.squeeze() > 0).float()

            # print(out_mask_logits_left.shape, out_mask_logits_right.shape)

            # Stack into (2, H, W) for bi-manual track_frame
            # mask_bi = torch.stack([mask_left, mask_right], dim=0)
            # Convert out_mask_logits_right to binary masks and stack for bi-manual track_frame
            mask_bi = (out_mask_logits_right.squeeze() > 0).float()  # default to right mask if left mask is missing

            # One-time init tracking step (bi-manual)
            cTr, joint_angles, loss = tracker.track_frame(
                ref_mask=mask_bi,
                joint_angles=joint_angle_readings[frame_idx],  # (2, 4)
                is_init=True,
                keypoints=init_keypoints,  # (2, K, 2) or None
            )

            init_done = True

            gc.collect()
            torch.cuda.empty_cache()

        else:
            # Segmentation (multi-object)
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_obj_ids, out_mask_logits = get_next_mask(frame)
            torch.cuda.synchronize()
            end_time = time.time()
            seg_time_lst.append(end_time - start_time)

            # out_mask_logits is typically (Nobj, H, W) or similar; enforce mapping to [left,right]
            # We assume obj_id 0->left, 1->right.
            # Some SAM2 outputs may not preserve ordering; if out_obj_ids is available, use it.
            if isinstance(out_obj_ids, (list, tuple)) and len(out_obj_ids) == out_mask_logits.shape[0]:
                id_to_mask = {int(oid): out_mask_logits[i] for i, oid in enumerate(out_obj_ids)}
                left_logits = id_to_mask.get(0, out_mask_logits[0])
                right_logits = id_to_mask.get(1, out_mask_logits[1] if out_mask_logits.shape[0] > 1 else out_mask_logits[0])
            else:
                left_logits = out_mask_logits[0]
                right_logits = out_mask_logits[1] if out_mask_logits.shape[0] > 1 else out_mask_logits[0]

            mask_left = (left_logits.squeeze() > 0).float()
            mask_right = (right_logits.squeeze() > 0).float()
            mask_bi = torch.stack([mask_left, mask_right], dim=0)  # (2, H, W)
            # mask_bi = (right_logits.squeeze() > 0).float()  # default to right mask if left mask is missing

            # Tracking (bi-manual)
            torch.cuda.synchronize()
            start_time = time.time()
            cTr, joint_angles, loss = tracker.track_frame(
                ref_mask=mask_bi,
                joint_angles=joint_angle_readings[frame_idx],  # (2, 4)
                is_init=False,
                keypoints=None,
            )
            torch.cuda.synchronize()
            end_time = time.time()
            track_time_lst.append(end_time - start_time)

        cTr_lst.append(cTr.clone())               # (2, 6)
        joint_angles_lst.append(joint_angles.clone())  # (2, 4)
        frame_idx += 1

        # --- visualization ---
        # show combined masks (for display only)
        mask_vis = torch.max(mask_bi[0], mask_bi[1]).detach().cpu().numpy().astype(np.uint8) * 255
        color = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.7, color, 0.3, 0)

        # Draw both arms sequentially.
        blended = skeleton_visualizer_left.plot_skeleton_overlay(blended, cTr[0], joint_angles[0])
        blended = skeleton_visualizer_right.plot_skeleton_overlay(blended, cTr[1], joint_angles[1])

        blended = cv2.resize(blended, frame_shape_orig)

        if len(seg_time_lst) > 10 and len(track_time_lst) > 10:
            avg_time = (sum(seg_time_lst[-10:]) / len(seg_time_lst[-10:])) + (sum(track_time_lst[-10:]) / len(track_time_lst[-10:]))
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            fps = 1 / avg_time if avg_time > 0 else 0
            cv2.putText(
                blended,
                f"Loss: {loss_val:.4f} | FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        cv2.imshow("frame", blended)

        # Add elapsed wall-clock time overlay
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

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if rt_writer is not None:
        rt_writer.release()
        print(f"Saved real-time faithful video to: {out_path}")

    # Compute average FPS (exclude first 10)
    if len(seg_time_lst) > 10 and len(track_time_lst) > 10:
        avg_seg_time = sum(seg_time_lst[10:]) / len(seg_time_lst[10:])
        avg_track_time = sum(track_time_lst[10:]) / len(track_time_lst[10:])
        avg_time = avg_seg_time + avg_track_time
        fps = 1 / avg_time if avg_time > 0 else 0
        print(f"Average FPS (excluding first 10 frames): {fps:.2f}")
    else:
        avg_seg_time = float("nan")
        avg_track_time = float("nan")
        avg_time = float("nan")
        fps = 0.0
        print("Not enough frames to compute average FPS excluding initialization.")

    # Save pose results (bi-manual)
    if not os.path.exists("./pose_results"):
        os.makedirs("./pose_results")

    cTr_seq = torch.stack(cTr_lst).cpu()                  # (T, 2, 6)
    joint_angles_seq = torch.stack(joint_angles_lst).cpu()  # (T, 2, 4)

    # match old behavior: store per-frame total time if desired
    if len(seg_time_lst) == len(track_time_lst):
        time_seq = (torch.tensor(seg_time_lst).cpu() + torch.tensor(track_time_lst).cpu())
    else:
        # in case init frame isn't appended to both lists uniformly
        time_seq = torch.tensor([0.0] * len(cTr_lst)).cpu()

    data_label = f"surgpose_{args.video_label}_BI_MANUAL_{args.machine_left}_{args.machine_right}"
    joint_str = "wo_joint_angles" if args.use_prev_joint_angles else "w_joint_angles"
    pts_loss_str = "w_pts_loss" if args.use_pts_loss else "wo_pts_loss"
    app_loss_str = "w_app_loss" if args.app_weight > 0 else "wo_app_loss"
    kpts_det_str = "w_tipnet" if (args.use_pts_loss and args.use_contour_tip_net) else "wo_kpts_det"
    renderer_str = "nvdiffrast" if args.use_nvdiffrast else "pytorch3d"
    filter_str = "no_filter" if not args.use_filter else args.filter_option
    option_label = "sep" if args.separate_loss else "joint"
    sep_label = "softsep" if args.soft_separation else "hardsep"

    save_path = (
        f"./pose_results/BI_MANUAL_{data_label}."
        f"{args.searcher}.{args.online_iters}."
        f"{joint_str}.{pts_loss_str}.{kpts_det_str}.{app_loss_str}."
        f"{filter_str}.{renderer_str}.{option_label}.{sep_label}.pth"
    )

    torch.save({"cTr": cTr_seq, "joint_angles": joint_angles_seq, "time": time_seq}, save_path)

    print(f"Pose results saved to {save_path}.")
    print(f"    Average segmentation time per frame: {avg_seg_time:.4f} seconds.")
    print(f"    Average tracking time per frame: {avg_track_time:.4f} seconds.")
    print(f"    Average total time per frame: {avg_time:.4f} seconds.")
    print(f"    Average FPS (excluding first 10 frames): {fps:.2f}")
    print(f"    Total frames: {len(cTr_lst)}")