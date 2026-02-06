import torch
import kornia
import numpy as np

import nvdiffrast.torch as dr

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from diffcali.models.mesh_renderer import RobotMeshRenderer


@torch.compile()
def transform_mesh(cameras, mesh, R, T, t_mtx):
    """
    Transform the mesh from world space to clip space
    Modified from https://github.com/NVlabs/nvdiffrast/issues/148#issuecomment-2090054967
    """
    # world to view transform
    verts = mesh.verts_padded()  #  (B, N_v, 3)
    verts_view = cameras.get_world_to_view_transform(R=R, T=T).transform_points(verts)  # (B, N_v, 3)
    verts_view[...,  :3] *= -1 # due to PyTorch3D camera coordinate conventions
    verts_view_home = torch.cat([verts_view, torch.ones_like(verts_view[..., [0]])], axis=-1) # (B, N_v, 4)

    # projection
    verts_clip = torch.matmul(verts_view_home, t_mtx.transpose(0, 1))
    faces_clip = mesh.faces_padded().to(torch.int32)

    return verts_clip, faces_clip


@torch.compile()
def transform_verts(cameras, verts, R, T, t_mtx, bi_manual=False):
    """
    Transform the vertices from world space to clip space
    Inputs:
        cameras: PyTorch3D camera object
        verts: (B, N_v, 3) tensor of vertex positions or (B, 2*N_v, 3) for bi_manual=True
        R: (B, 3, 3) or (B, 2, 3, 3) rotation matrices
        T: (B, 3) or (B, 2, 3) translation vectors
        t_mtx: (4, 4) projection matrix
        bi_manual: whether the input R
    """
    if bi_manual:
        B, N_v2, _ = verts.shape
        assert len(R.shape) == 4 and R.shape[1] == 2, "R should be of shape (B, 2, 3, 3) for bi_manual=True"
        assert len(T.shape) == 3 and T.shape[1] == 2, "T should be of shape (B, 2, 3) for bi_manual=True"
        N_v = N_v2 // 2

        verts_1 = verts[:, :N_v, :] # (B, N_v, 3)
        verts_2 = verts[:, N_v:, :] # (B, N_v, 3)

        verts_view_1 = cameras.get_world_to_view_transform(R=R[:,0], T=T[:,0]).transform_points(verts_1)
        verts_view_1[...,  :3] *= -1 # due to PyTorch3D camera coordinate conventions
        verts_view_home_1 = torch.cat([verts_view_1, torch.ones_like(verts_view_1[..., [0]])], axis=-1) # (B, N_v, 4)

        verts_view_2 = cameras.get_world_to_view_transform(R=R[:,1], T=T[:,1]).transform_points(verts_2)
        verts_view_2[...,  :3] *= -1 # due to PyTorch3D camera coordinate conventions
        verts_view_home_2 = torch.cat([verts_view_2, torch.ones_like(verts_view_2[..., [0]])], axis=-1) # (B, N_v, 4)

        # projection
        verts_clip_1 = torch.matmul(verts_view_home_1, t_mtx.transpose(0, 1))
        verts_clip_2 = torch.matmul(verts_view_home_2, t_mtx.transpose(0, 1))

        verts_clip = torch.cat([verts_clip_1, verts_clip_2], dim=1) # (B, 2*N_v, 4)

        return verts_clip

    else:
        verts_view = cameras.get_world_to_view_transform(R=R, T=T).transform_points(verts)
        verts_view[...,  :3] *= -1 # due to PyTorch3D camera coordinate conventions
        verts_view_home = torch.cat([verts_view, torch.ones_like(verts_view[..., [0]])], axis=-1) # (B, N_v, 4)

        # projection
        verts_clip = torch.matmul(verts_view_home, t_mtx.transpose(0, 1))

    return verts_clip


def render(glctx, pos, pos_idx, resolution: [int, int], antialiasing=False, col=None):
    """
    Silhouette rendering pipeline based on NvDiffRast
    if col is None, render silhouette mask
    otherwise (col is (1, N_v, 3)), render colored image (three channels)
    """
    # Create color attributes
    if col is None:
        col = torch.ones_like(pos[..., :1], dtype=torch.float32) # (B, N_v, 1)
    col_idx = pos_idx

    # Render the mesh
    rast_out, _ = dr.rasterize(glctx, pos, pos_idx, resolution=resolution)
    color   , _ = dr.interpolate(col, rast_out, col_idx)
    if antialiasing:
        color = dr.antialias(color, rast_out, pos, pos_idx)
    return color.squeeze(-1) # (B, H, W)


class CtRNet(torch.nn.Module):
    def __init__(self, args):
        super(CtRNet, self).__init__()

        self.args = args

        if args.use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.use_antialiasing = True

        # set up camera intrinsics

        self.intrinsics = np.array(
            [[args.fx, 0.0, args.px], [0.0, args.fy, args.py], [0.0, 0.0, 1.0]]
        )

        print("Camera intrinsics: {}".format(self.intrinsics))

        self.K = torch.tensor(self.intrinsics, device=self.device, dtype=torch.float)
        self.visibility_flags = [False, True, True, True]

        # NvDiffRast configuration
        self.glctx = dr.RasterizeCudaContext() # CUDA context (OpenGL is not available in my WSL)
        # self.glctx = dr.RasterizeGLContext(output_db=False)
        self.resolution = (args.height, args.width)

        fx, fy = -args.fx, -args.fy # negative
        px, py = args.px, args.py
        height, width = args.height, args.width
        near, far = 1e-3, 1e9 # set near and far clipping planes (if znear is too small, will have numerical issues)
        A = (2 * fx) / width
        B = (2 * fy) / height
        C = (width - 2 * px) / width
        D = (height - 2 * py) / height
        E = (near + far) / (near - far)
        F = (2 * near * far) / (near - far)
        self.t_mtx = projectionMatrix = torch.tensor(
            [
                [A, 0, C, 0],
                [0, B, D, 0],
                [0, 0, E, F],
                [0, 0, -1, 0]
            ]
        ).cuda().contiguous()  # (4, 4)

        # TODO: test
        """
        # set up robot model
        if args.robot_name == "Panda":
            from .robot_arm import PandaArm
            self.robot = PandaArm(args.urdf_file)
        elif args.robot_name == "Baxter_left_arm":
            from .robot_arm import BaxterLeftArm
            self.robot = BaxterLeftArm(args.urdf_file)
        print("Robot model: {}".format(args.robot_name))
        """

    def set_mesh_visibility(self, visibility_flags):
        self.visibility_flags = visibility_flags

    def inference_single_image(self, img, joint_angles):
        # img: (3, H, W)
        # joint_angles: (7)
        # robot: robot model

        # detect 2d keypoints and segmentation masks
        points_2d, segmentation = self.keypoint_seg_predictor(img[None])
        foreground_mask = torch.sigmoid(segmentation)
        _, t_list = self.robot.get_joint_RT(joint_angles)
        points_3d = torch.from_numpy(np.array(t_list)).float().to(self.device)
        if self.args.robot_name == "Panda":
            points_3d = points_3d[
                [0, 2, 3, 4, 6, 7, 8]
            ]  # remove 1 and 5 links as they are overlapping with 2 and 6

        # init_pose = torch.tensor([[  1.5497,  0.5420, -0.3909, -0.4698, -0.0211,  1.3243]])
        # cTr = bpnp(points_2d_pred, points_3d, K, init_pose)
        cTr = self.bpnp(points_2d, points_3d, self.K)

        return cTr, points_2d, foreground_mask

    def inference_batch_images(self, img, joint_angles):
        # img: (B, 3, H, W)
        # joint_angles: (B, 7)
        # robot: robot model

        # detect 2d keypoints and segmentation masks
        points_2d, segmentation = self.keypoint_seg_predictor(img)
        foreground_mask = torch.sigmoid(segmentation)

        points_3d_batch = []
        for b in range(joint_angles.shape[0]):
            _, t_list = self.robot.get_joint_RT(joint_angles[b])
            points_3d = torch.from_numpy(np.array(t_list)).float().to(self.device)
            if self.args.robot_name == "Panda":
                points_3d = points_3d[:, [0, 2, 3, 4, 6, 7, 8]]
            points_3d_batch.append(points_3d[None])

        points_3d_batch = torch.cat(points_3d_batch, dim=0)

        cTr = self.bpnp_m3d(points_2d, points_3d_batch, self.K)

        return cTr, points_2d, foreground_mask

    def cTr_to_pose_matrix(self, cTr):
        """
        cTr: (batch_size, 6)
        pose_matrix: (batch_size, 4, 4)
        """
        batch_size = cTr.shape[0]
        pose_matrix = torch.zeros((batch_size, 4, 4), device=self.device)
        # print(f"debugging the ctr to pose {cTr.shape} ")
        pose_matrix[:, :3, :3] = (
            kornia.geometry.conversions.axis_angle_to_rotation_matrix(cTr[:, :3])
        )
        pose_matrix[:, :3, 3] = cTr[:, 3:]
        pose_matrix[:, 3, 3] = 1
        # if torch.any(torch.isnan(pose_matrix)) :
        #     print(f"output potential Nan numbers ctr: {cTr}")
        #     raise ValueError("NaN detected in inputs to get_img_coords!")
        return pose_matrix

    def pose_matrix_to_cTr(self, pose_matrix):
        """
        pose_matrix: (batch_size, 4, 4)
        cTr: (batch_size, 6)
        """
        batch_size = pose_matrix.shape[0]

        # Extract rotation matrix (3x3) from pose_matrix
        rotation_matrix = pose_matrix[:, :3, :3].contiguous()
        # print(f"check rotation_matrix {rotation_matrix.shape}")

        # Convert rotation matrix to angle-axis representation
        angle_axis = kornia.geometry.conversions.rotation_matrix_to_axis_angle(
            rotation_matrix
        )

        # Extract translation vector (x, y, z) from pose_matrix
        translation = pose_matrix[:, :3, 3]

        # Concatenate angle-axis and translation to get cTr (batch_size, 6)
        cTr = torch.cat((angle_axis, translation), dim=1)

        return cTr.to(self.device)

    def to_valid_R_batch(self, R):
        # R is a batch of 3x3 rotation matrices
        U, S, V = torch.svd(R)
        return torch.bmm(U, V.transpose(1, 2))

    def get_joint_angles(self, joint_angles):
        self.joint_angles = joint_angles

    # def from_lookat_to_pose_matrix(self, dist, elev, azim):

    #     R, T = look_at_view_transform(dist, elev, azim)
    #     batch_size = R.shape[0]
    #     # print(f"test batch_size: {R.shape}")
    #     # Add the translation to form a 4x4 matrix
    #     pose_matrix = torch.eye(4).repeat(batch_size, 1, 1)  # (N, 4, 4)
    #     pose_matrix[:, :3, :3] = R
    #     pose_matrix[:, :3, 3] = T

    #     return pose_matrix

    def from_lookat_to_pose_matrix(self, dist, elev, roll):
        """
        Returns a single 4x4 pose matrix for PyTorch3D rendering, where:
        1) azim is extrinsic about global X,
        2) elev is extrinsic about global Y,
        3) roll is about the *local* Z axis after the first two.
        Then we translate along +Z by 'dist'.

        So effectively:
        final_R = [roll about local Z] * [rotate about global Y] * [rotate about global X].
        """

        # 1) Convert degrees to radians
        # azim_rad = torch.deg2rad(azim)
        elev_rad = torch.deg2rad(elev)
        roll_rad = torch.deg2rad(roll)

        # 2) Basic 3x3 rotations about GLOBAL X, Y, Z
        #    (We'll use the Z one only as a base for local-Z conjugation.)
        #
        # # -- R_x(azim)
        # c_az = torch.cos(azim_rad)
        # s_az = torch.sin(azim_rad)
        # R_azim = torch.tensor([
        #     [ 1.0,  0.0,  0.0],
        #     [ 0.0,  c_az, -s_az],
        #     [ 0.0,  s_az,  c_az],
        # ], dtype=torch.float32)

        # -- R_y(elev)
        c_el = torch.cos(elev_rad)
        s_el = torch.sin(elev_rad)
        R_elev = torch.tensor(
            [
                [c_el, 0.0, s_el],
                [0.0, 1.0, 0.0],
                [-s_el, 0.0, c_el],
            ],
            dtype=torch.float32,
        )

        # -- R_z(roll) : "base" rotation about global Z
        c_rl = torch.cos(roll_rad)
        s_rl = torch.sin(roll_rad)
        Rz_global_roll = torch.tensor(
            [
                [c_rl, -s_rl, 0.0],
                [s_rl, c_rl, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        # 3) Combine azim & elev extrinsically:
        #    If we read right->left, first is X(azim), second is Y(elev).
        #    => R_ae = R_elev @ R_azim
        R_ae = R_elev

        # 4) Construct a rotation about the *local Z* that results from R_ae.
        #
        #    local-Z rotation = R_ae * (rotation about global Z) * R_ae^-1
        #    We'll call that R_roll_localZ.
        R_ae_inv = R_ae.T

        R_roll_localZ = R_ae @ Rz_global_roll

        # 5) Final orientation: R_roll_localZ @ R_ae
        #    In "right-to-left" application to a vector v:
        #      - first apply R_ae (the extrinsic X->Y),
        #      - then apply R_roll_localZ (the local Z roll).
        final_R = R_ae @ Rz_global_roll

        # 6) Translation along +Z by dist
        T = torch.tensor([0.0, 0.0, dist], dtype=torch.float32)

        # 7) Construct 4x4 pose
        pose_matrix = torch.eye(4, dtype=torch.float32)
        pose_matrix[:3, :3] = final_R
        pose_matrix[:3, 3] = T

        return pose_matrix.unsqueeze(0)

    def setup_robot_renderer(self, mesh_files):
        # mesh_files: list of mesh files
        focal_length = [-self.args.fx, -self.args.fy]
        principal_point = [self.args.px, self.args.py]
        image_size = [self.args.height, self.args.width]

        robot_renderer = RobotMeshRenderer(
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=image_size,
            robot=None,
            mesh_files=mesh_files,
            device=self.device,
            visibility_flags=self.visibility_flags,
        )  # TODO: test

        return robot_renderer

    def render_single_robot_mask(self, cTr, robot_mesh, robot_renderer, resolution=None):
        resolution = resolution if resolution is not None else self.resolution
        
        # cTr: (6)
        # img: (1, H, W)
        # print(f"test ctr: {cTr}")
        R = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
            cTr[:3][None]
        )  # (1, 3, 3)
        R = torch.transpose(R, 1, 2)
        # R = to_valid_R_batch(R)
        T = cTr[3:][None]  # (1, 3)

        # """Debugging the negative depth..."""
        # pose_matrix = self.cTr_to_pose_matrix(cTr[None])
        # pose_matrix = pose_matrix[0]

        # # Extract the vertices in world/robot frame and convert them to homogeneous coordinates
        # verts = robot_mesh.verts_packed()  # (V, 3)
        # verts_hom = torch.cat(
        #     [verts, torch.ones((verts.shape[0], 1), device=verts.device)], dim=1
        # )  # (V, 4)

        # # Transform vertices into camera coordinate frame:
        # # Assuming cTr transforms from robot/world frame to camera frame
        # verts_camera_space = (pose_matrix @ verts_hom.T).T  # (V, 4)
        # X_c = verts_camera_space[:, 0]
        # Y_c = verts_camera_space[:, 1]
        # Z_c = verts_camera_space[:, 2]
        # # print(f"checking the min z value {Z_c.min()}")

        if T[0, -1] < 0:
            if self.args.use_nvdiffrast:
                pos, pos_idx = transform_mesh(
                    cameras=robot_renderer.cameras, mesh=robot_mesh,
                    R=-R, T=-T, t_mtx=self.t_mtx
                ) # project the batched meshes in the clip space
                rendered_image = render(self.glctx, pos, pos_idx[0], resolution, self.use_antialiasing)
            else:
                # print("Using PyTorch3D")
                rendered_image = robot_renderer.silhouette_renderer(
                    meshes_world=robot_mesh, R=-R, T=-T
                )[..., 3]
        else:
            if self.args.use_nvdiffrast:
                pos, pos_idx = transform_mesh(
                    cameras=robot_renderer.cameras, mesh=robot_mesh,
                    R=R, T=T, t_mtx=self.t_mtx
                ) # project the batched meshes in the clip space
                rendered_image = render(self.glctx, pos, pos_idx[0], resolution, self.use_antialiasing)
            else:
                # print("Using PyTorch3D")
                rendered_image = robot_renderer.silhouette_renderer(
                    meshes_world=robot_mesh, R=R, T=T
                )[..., 3]
                

        # rendered_image = robot_renderer.silhouette_renderer(meshes_world=robot_mesh, R = R, T = T)

        if torch.isnan(rendered_image).any():
            rendered_image = torch.nan_to_num(rendered_image)

        return rendered_image

    def render_robot_mask_batch_nvdiffrast(self, cTr_batch, verts, faces, robot_renderer, resolution=None, color=None, bi_manual=False):
        assert self.args.use_nvdiffrast, "This function is only for nvdiffrast"

        resolution = resolution if resolution is not None else self.resolution
        B = cTr_batch.shape[0]

        if bi_manual:
            assert len(cTr_batch.shape) == 3 and cTr_batch.shape[1] == 2, "cTr_batch should be of shape (B, 2, 6) for bi_manual=True"
            cTr_batch = cTr_batch.view(-1, cTr_batch.shape[-1]) # (B*2, 6)
            R_batched = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
                cTr_batch[:, :3]
            ).reshape(B, 2, 3, 3)  # (B, 2, 3, 3)
            T_batched = cTr_batch[:, 3:].reshape(B, 2, 3)  # (B, 2, 3)
        else:# 1) Convert angle-axis to rotation matrices (B,3,3), plus translation (B,3)
            R_batched = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
                cTr_batch[:, :3]
            )  # (B,3,3)
            T_batched = cTr_batch[:, 3:]  # (B,3)

        return self.render_robot_mask_batch_nvdiffrast_rotmat(R_batched, T_batched, verts, faces, robot_renderer, resolution, color, bi_manual=bi_manual)    

    def render_robot_mask_batch_nvdiffrast_rotmat(self, R_batched, T_batched, verts, faces, robot_renderer, resolution=None, color=None, bi_manual=False):
        assert self.args.use_nvdiffrast, "This function is only for nvdiffrast"

        resolution = resolution if resolution is not None else self.resolution
        B = R_batched.shape[0]

        if bi_manual:
            assert len(R_batched.shape) == 4 and R_batched.shape[1] == 2, "R_batched should be of shape (B, 2, 3, 3) for bi_manual=True"
            
            R_batched = R_batched.transpose(2, 3) # (B, 2, 3, 3)
            negative_mask = T_batched[:, :, -1] < 0  # shape (B, 2)
            T_batched_ = T_batched.clone()
            T_batched_[negative_mask] = -T_batched_[negative_mask]

            R_batched_ = R_batched.clone()
            R_batched_[negative_mask] = -R_batched_[negative_mask]

            pos = transform_verts(
                cameras=robot_renderer.cameras, verts=verts,
                R=R_batched_, T=T_batched_, t_mtx=self.t_mtx, 
                bi_manual=bi_manual
            ) # project the batched meshes in the clip space
        
        else:
            R_batched = R_batched.transpose(1, 2)

            negative_mask = T_batched[:, -1] < 0  # shape (B,)
            T_batched_ = T_batched.clone()
            T_batched_[negative_mask] = -T_batched_[negative_mask]

            R_batched_ = R_batched.clone()
            R_batched_[negative_mask] = -R_batched_[negative_mask]

            pos = transform_verts(
                cameras=robot_renderer.cameras, verts=verts,
                R=R_batched_, T=T_batched_, t_mtx=self.t_mtx,
                bi_manual=bi_manual
            ) # project the batched meshes in the clip space
        silhouettes = render(self.glctx, pos, faces, resolution, self.use_antialiasing, col=color)

        return silhouettes


    def render_robot_mask_batch(self, cTr_batch, robot_mesh, robot_renderer):
        """
        cTr_batch: (B, 6)
        Return: (B, H, W) silhouette mask in [0..1], single-pass rendering without Python for-loop.
        """
        B = cTr_batch.shape[0]

        # 1) Convert angle-axis to rotation matrices (B,3,3), plus translation (B,3)
        R_batched = kornia.geometry.conversions.axis_angle_to_rotation_matrix(
            cTr_batch[:, :3]
        )  # (B,3,3)
        # If your renderer expects the transpose or some different orientation, do R_batched = R_batched.transpose(1,2) if needed
        R_batched = R_batched.transpose(1, 2)
        T_batched = cTr_batch[:, 3:]  # (B,3)

        # 2) If the mesh is a single Meshes object and you want the SAME geometry for all B transforms,
        #    you can replicate it B times via .extend(B). For example (if using Pytorch3D's Meshes):
        batched_meshes = robot_mesh.extend(
            B
        )  # Now we have B copies of the same geometry.

        # silhouette_renderer should accept R_batched, T_batched of shape (B,3,3), (B,3)
        # and batched_meshes of type Meshes with B items.
        # The result typically has shape (B, H, W, 4) (RGBA).
        negative_mask = T_batched[:, -1] < 0  # shape (B,)
        # Where negative_mask is True, flip
        T_batched_ = T_batched.clone()
        T_batched_[negative_mask] = -T_batched_[negative_mask]

        R_batched_ = R_batched.clone()
        R_batched_[negative_mask] = -R_batched_[negative_mask]

        # 3) Render the batch in a single pass:
        if self.args.use_nvdiffrast:
            pos, pos_idx = transform_mesh(
                cameras=robot_renderer.cameras, mesh=batched_meshes,
                R=R_batched_, T=T_batched_, t_mtx=self.t_mtx
            ) # project the batched meshes in the clip space
            silhouettes = render(self.glctx, pos, pos_idx[0], self.resolution, self.use_antialiasing)

        else:
            # Use PyTorch3D

            batched_silhouettes = robot_renderer.silhouette_renderer(
                meshes_world=batched_meshes, R=R_batched_, T=T_batched_
            )

            # shape: (B, H, W, 4)

            # 4) Extract alpha channel => silhouette (B,H,W)
            silhouettes = batched_silhouettes[..., 3]
            # shape (B,H,W)

        return silhouettes

    def train_on_batch(
        self, img, joint_angles, robot_renderer, criterions, phase="train"
    ):
        # img: (B, 3, H, W)
        # joint_angles: (B, 7)
        with torch.set_grad_enabled(phase == "train"):
            # detect 2d keypoints
            points_2d, segmentation = self.keypoint_seg_predictor(img)

            mask_list = list()
            seg_weight_list = list()

            for b in range(img.shape[0]):
                # get 3d points
                _, t_list = self.robot.get_joint_RT(joint_angles[b])
                points_3d = torch.from_numpy(np.array(t_list)).float().to(self.device)
                if self.args.robot_name == "Panda":
                    points_3d = points_3d[:, [0, 2, 3, 4, 6, 7, 8]]

                # get camera pose
                cTr = self.bpnp(points_2d[b][None], points_3d, self.K)

                # config robot mesh
                robot_mesh = robot_renderer.get_robot_mesh(joint_angles[b])

                # render robot mask
                rendered_image = self.render_single_robot_mask(
                    cTr.squeeze(), robot_mesh, robot_renderer
                )

                mask_list.append(rendered_image)
                points_2d_proj = batch_project(cTr, points_3d, self.K)
                reproject_error = criterions["mse_mean"](
                    points_2d[b], points_2d_proj.squeeze()
                )
                seg_weight = torch.exp(-reproject_error * self.args.reproj_err_scale)
                seg_weight_list.append(seg_weight)

            mask_batch = torch.cat(mask_list, 0)

            loss_bce = 0
            for b in range(segmentation.shape[0]):
                loss_bce = loss_bce + seg_weight_list[b] * criterions["bce"](
                    segmentation[b].squeeze(), mask_batch[b].detach()
                )

            img_ref = torch.sigmoid(segmentation).detach()
            # loss_reproj = 0.0005 * criterionMSE_mean(points_2d, points_2d_proj_batch)
            loss_mse = 0.001 * criterions["mse_sum"](mask_batch, img_ref.squeeze())
            loss = loss_mse + loss_bce

        return loss
