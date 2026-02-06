import argparse
import numpy as np
import torch as th
import os
import cv2

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffcali.models.CtRNet import CtRNet
from diffcali.utils.ui_utils import *
from diffcali.utils.detection_utils import detect_lines
from diffcali.eval_dvrk.batch_optimize import BatchOptimize  # The class we just wrote
from diffcali.eval_dvrk.optimize import Optimize  # Your single-sample class
from diffcali.eval_dvrk.black_box_optimize import BlackBoxOptimize

th.cuda.empty_cache()


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", type=str, default="urdfs/dVRK/meshes")
    data_dir = "data/consistency_evaluation/medium/0"
    parser.add_argument("--data_dir", type=str, default=data_dir)  # reference mask
    parser.add_argument(
        "--ref_img_file", type=str, default=os.path.join(data_dir, "00153.png")
    )  # reference mask
    parser.add_argument(
        "--joint_file", type=str, default=os.path.join(data_dir, "joint_0010.npy")
    )  # joint angles
    parser.add_argument(
        "--jaw_file", type=str, default=os.path.join(data_dir, "jaw_0010.npy")
    )  # jaw angles
    parser.add_argument("--batch_opt_lr", type=float, default=3e-3)
    parser.add_argument("--single_opt_lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument(
        "--batch_iters", type=int, default=100
    )  # Coarse steps per batch
    parser.add_argument(
        "--final_iters", type=int, default=1000
    )  # Final single-sample refine
    # parser.add_argument('--data_path', type=str, default='data/extractions/bag_1')'
    parser.add_argument("--arm", type=str, default="psm2")
    parser.add_argument("--sample_number", type=int, default=500)
    parser.add_argument("--use_nvdiffrast", action="store_true")
    parser.add_argument("--pop_size", type=int, default=10)
    args = parser.parse_args()
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


def buildcTr(cTr_train, cTr_nontrain):
    # Rebuild [angle_axis(3), xyz(3)] from cTr_train
    return th.cat([cTr_train[0], cTr_train[1], cTr_train[2]], dim=0)


def main():
    args = parseArgs()
    ctrnet_args = parseCtRNetArgs()

    ctrnet_args.use_nvdiffrast = args.use_nvdiffrast
    if ctrnet_args.use_nvdiffrast:
        print("Using NvDiffRast!")

    # calibrate_indices = np.linspace(0, len(total_frames), 60).astype(int)
    calibrate_indices = [14]
    print(f"check indices to calibrate: {calibrate_indices}")

    optimized_ctr = []

    for index in calibrate_indices:
        # args.ref_img_file = os.path.join(frames_dir, total_frames[index])
        joints = np.load(args.joint_file)
        jaw = np.load(args.jaw_file)

        """Or just for a single image processing"""

        # 1) Build model
        model = CtRNet(ctrnet_args)
        mesh_files = [
            f"{args.mesh_dir}/shaft_multi_cylinder.ply",
            f"{args.mesh_dir}/logo_low_res_1.ply",
            f"{args.mesh_dir}/jawright_lowres.ply",
            f"{args.mesh_dir}/jawleft_lowres.ply",
        ]

        robot_renderer = model.setup_robot_renderer(mesh_files)
        robot_renderer.set_mesh_visibility([True, True, True, True])

        # 2) Joint angles (same for all items, or replicate if needed)
        joint_angles_np = np.array(
            [joints[4], joints[5], jaw[0] / 2, jaw[0] / 2], dtype=np.float32
        )
        joint_angles = th.tensor(
            joint_angles_np, device=model.device, requires_grad=False, dtype=th.float32
        )
        print("Joint angles: ", joint_angles)
        # joint_angles = joint_angles + th.randn_like(joint_angles) * 0.01 # inject noise
        model.get_joint_angles(joint_angles)
        robot_mesh = robot_renderer.get_robot_mesh(joint_angles)

        # 3) Reference mask
        cv_img = cv2.imread(args.ref_img_file, cv2.IMREAD_GRAYSCALE)
        ref_mask_np = (cv_img / 255.0).astype(np.float32)
        ref_mask_t = th.tensor(ref_mask_np, device=model.device)

        """define ref_keypoints"""
        cv_img = cv2.imread(args.ref_img_file)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        ref_img = cv2.inRange(cv_img, np.ones(3) * 128, np.ones(3) * 255) / 255.0
        ref_img = th.tensor(ref_img, dtype=th.float32)

        # ref_keypoints = get_reference_keypoints(ref_img, num_keypoints=2)
        ref_keypoints =  get_reference_keypoints_auto(args.ref_img_file, num_keypoints=2)
        print(f"detect keypoints shape {ref_keypoints}")

        """define ref_keypoints (auto detection)"""
        # cv_img = cv2.imread(args.ref_img_file)
        # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        ref_img = cv2.inRange(cv_img, np.ones(3) * 128, np.ones(3) * 255)
        binary_mask = ref_img.astype(np.uint8)
        detect_lines(binary_mask)

        # 4) Generate all initial cTr in some way (N total). For demo, let's do random.

        N = args.sample_number
        cTr_inits = []
        for i in range(N):
            camera_roll_local = th.empty(1).uniform_(
                0, 360
            )  # Random values in [0, 360]
            camera_roll = th.empty(1).uniform_(0, 360)  # Random values in [0, 360]
            azimuth = th.empty(1).uniform_(0, 360)  # Random values in [0, 360]
            elevation = th.empty(1).uniform_(
                90 - 60, 90 - 30
            )  # Random values in [90-25, 90+25]
            # elevation = 30
            distance = th.empty(1).uniform_(0.10, 0.17)

            pose_matrix = model.from_lookat_to_pose_matrix(
                distance, elevation, camera_roll_local
            )
            roll_rad = th.deg2rad(camera_roll)  # Convert roll angle to radians
            roll_matrix = th.tensor(
                [
                    [th.cos(roll_rad), -th.sin(roll_rad), 0],
                    [th.sin(roll_rad), th.cos(roll_rad), 0],
                    [0, 0, 1],
                ]
            )
            pose_matrix[:, :3, :3] = th.matmul(roll_matrix, pose_matrix[:, :3, :3])
            cTr = model.pose_matrix_to_cTr(pose_matrix)
            if not th.any(th.isnan(cTr)):
                cTr_inits.append(cTr)
        cTr_inits_t = th.cat(cTr_inits, dim=0)
        print(f"All ctr candiates: {cTr_inits_t.shape }")
        bsz = args.batch_size
        final_batch_winners = []
        final_batch_winners_losses = []

        # # TODO: Change this into auto detection & process several frames together to get the RCM. Measure the deviation etc.

        if N <= bsz:
            # Handle small N case (process in a single batch)
            print(f"Small N={N}, processing in a single batch.")
            cTr_batch = cTr_inits_t  # All samples in a single batch
            B = cTr_batch.shape[0]  # Batch size is equal to N

            batch_opt = BatchOptimize(
                cTr_batch=cTr_batch,
                joint_angles=joint_angles,
                model=model,
                robot_mesh=robot_mesh,
                robot_renderer=robot_renderer,
                ref_keypoints=ref_keypoints,
                fx=ctrnet_args.fx,
                fy=ctrnet_args.fy,
                px=ctrnet_args.px,
                py=ctrnet_args.py,
                lr=args.batch_opt_lr,
                batch_size=B,
            )

            batch_opt.readRefImage(args.ref_img_file)

            # Optimize
            best_cTr_in_batch, best_loss_in_batch = batch_opt.optimize_batch(
                iterations=args.batch_iters, grid_search=False, ld1=3, ld2=3, ld3=3
            )

            final_batch_winners.append(best_cTr_in_batch)
            final_batch_winners_losses.append(best_loss_in_batch)
            print(
                f"[Single batch] best loss={best_loss_in_batch:.4f}, ctr={best_cTr_in_batch}"
            )
        else:

            for start in range(0, N, bsz):
                end = min(start + bsz, N)
                cTr_batch = cTr_inits_t[start:end]  # shape (B,6)
                B = cTr_batch.shape[0]

                batch_opt = BatchOptimize(
                    cTr_batch=cTr_batch,
                    joint_angles=joint_angles,
                    model=model,
                    robot_mesh=robot_mesh,
                    robot_renderer=robot_renderer,
                    ref_keypoints=ref_keypoints,
                    fx=ctrnet_args.fx,
                    fy=ctrnet_args.fy,
                    px=ctrnet_args.px,
                    py=ctrnet_args.py,
                    lr=args.batch_opt_lr,
                    batch_size=bsz,
                )

                batch_opt.readRefImage(args.ref_img_file)
                # Coarse optimize
                best_cTr_in_batch, best_loss_in_batch = batch_opt.optimize_batch(
                    iterations=args.batch_iters, grid_search=False, ld1=3, ld2=3, ld3=3
                )

                # final_cTr shape => (B,6), final_losses => (B,), final_angles => (B,)
                # Pick best from this batch

                final_batch_winners.append(best_cTr_in_batch)
                final_batch_winners_losses.append(best_loss_in_batch)
                print(
                    f"[Batch range {start}-{end}] best in batch => loss={best_loss_in_batch:.4f} ctr={best_cTr_in_batch}"
                )

        # 6) Global best
        final_batch_winners_losses_np = np.array(final_batch_winners_losses)
        best_idx_global = np.argmin(final_batch_winners_losses_np)
        best_global_loss = final_batch_winners_losses_np[best_idx_global]
        best_global_cTr = final_batch_winners[best_idx_global]
        print("==== Global best among all batches ====")
        print("loss=", best_global_loss, "cTr=", best_global_cTr.cpu().numpy())

        # Additional: add gaussian noise into the best ctr and rank in the new batch.....
        noisy_bsz = args.batch_size
        temp = best_global_cTr.expand(noisy_bsz, best_global_cTr.shape[-1])  # (B, 6)
        noise = th.randn_like(temp)
        angle_std_scale = 0.1
        xyz_std_scale = 0.00001
        noise[:, :3] *= angle_std_scale  # Scale angles
        noise[:, 3:] *= xyz_std_scale  # Scale translations
        noisy_ctr = temp + noise  # (B, 6)

        nsy_opt = BatchOptimize(
            cTr_batch=noisy_ctr,
            joint_angles=joint_angles,
            model=model,
            robot_mesh=robot_mesh,
            robot_renderer=robot_renderer,
            ref_keypoints=ref_keypoints,
            fx=ctrnet_args.fx,
            fy=ctrnet_args.fy,
            px=ctrnet_args.px,
            py=ctrnet_args.py,
            lr=args.batch_opt_lr,
            batch_size=noisy_bsz,
        )

        nsy_opt.readRefImage(args.ref_img_file)
        # Coarse optimize
        best_cTr, best_loss = nsy_opt.optimize_batch(
            iterations=args.batch_iters, grid_search=False, ld1=3, ld2=3, ld3=3
        )
        print("==== Global best among noisy cTrs ====")
        print("loss=", best_loss, "cTr=", best_cTr)

        # 7) Black box / gradient-based optimization with the best cTr
        use_bbox_optimizer = False

        if use_bbox_optimizer:
            with th.no_grad():
                bbox_opt = BlackBoxOptimize(
                    model=model,
                    robot_mesh=robot_mesh,
                    robot_renderer=robot_renderer,
                    ref_keypoints=ref_keypoints,
                    ref_mask_file=args.ref_img_file,
                    joint_angles=joint_angles,
                    fx=ctrnet_args.fx,
                    fy=ctrnet_args.fy,
                    px=ctrnet_args.px,
                    py=ctrnet_args.py,
                    ld1=3,
                    ld2=3,
                    ld3=3,
                    center_init=th.cat([best_cTr, joint_angles], dim=0),
                )

                bbox_opt.optimize(args.final_iters)

        else:
            # best_cTr_np = best_global_cTr.squeeze().cpu().numpy()
            best_cTr_np = best_cTr.squeeze().cpu().numpy()
            axis_angle = th.tensor(best_cTr_np[:3], device=model.device, requires_grad=True)
            xy = th.tensor(best_cTr_np[3:5], device=model.device, requires_grad=True)
            z = th.tensor(best_cTr_np[5:], device=model.device, requires_grad=True)

            joint_angles = th.tensor(
                joint_angles_np, requires_grad=True, device=model.device, dtype=th.float32
            )

            model.get_joint_angles(joint_angles)

            cTr_train = [axis_angle, xy, z, joint_angles]

            single_opt = Optimize(
                cTr_train=cTr_train,
                model=model,
                robot_mesh=robot_mesh,
                robot_renderer=robot_renderer,
                lr=args.single_opt_lr,
                cTr_nontrain=None,
                buildcTr=buildcTr,
            )

            single_opt.readRefImage(args.ref_img_file)
            single_opt.ref_keypoints = ref_keypoints
            single_opt.ref_keypoints = th.tensor(
                single_opt.ref_keypoints, device=single_opt.model.device, dtype=th.float32
            )
            single_opt.fx, single_opt.fy, single_opt.px, single_opt.py = (
                ctrnet_args.fx,
                ctrnet_args.fy,
                ctrnet_args.px,
                ctrnet_args.py,
            )
            saving_dir = os.path.join(args.data_dir, "optimization")
            os.makedirs(saving_dir, exist_ok=True)

            final_cTr_s, final_loss_s, final_angle_s = single_opt.optimize(
                iterations=args.final_iters,
                save_fig_dir=saving_dir,
                ld1=3,
                ld2=3,
                ld3=3,
                set2=[3, 3, 3],
                xyz_steps=1,
                angles_steps=3,
                saving_interval=5,
                coarse_step_num=300,
                grid_search=False,
            )

            print(f"Refined cTr = {final_cTr_s.detach().cpu().numpy()}")
            print(f"Refined loss = {final_loss_s}, angle={final_angle_s}")
            print(f"joint angles before: {joint_angles_np}, after: {joint_angles}")

            np.save(
                os.path.join(args.data_dir, f"optimized_ctr_{args.arm}.npy"),
                final_cTr_s.detach().cpu().numpy(),
            )
            np.save(
                os.path.join(args.data_dir, f"optimized_joint_angles_{args.arm}.npy"),
                joint_angles.detach().cpu().numpy(),
            )

            optimized_ctr.append(final_cTr_s)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    main()
