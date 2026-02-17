import torch
import numpy as np

# io utils
from pytorch3d.io import load_ply

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    SoftSilhouetteShader,
    HardPhongShader,
    PointLights,
    PerspectiveCameras,
    Textures,
)

from os.path import exists

from diffcali.eval_dvrk.LND_fk import lndFK, batch_lndFK

# import torch.multiprocessing as mp


class RobotMeshRenderer:
    """
    Class that render robot mesh with differentiable renderer
    """

    def __init__(
        self,
        focal_length,
        principal_point,
        image_size,
        robot,
        mesh_files,
        device,
        visibility_flags,
        channel_ids=[0, 1, 2, 2] # shaft, wrist, gripper left, gripper right
    ):

        self.focal_length = focal_length
        self.principal_point = principal_point
        self.image_size = image_size
        self.device = device
        self.robot = robot
        self.mesh_files = mesh_files
        self.preload_verts = []
        self.preload_faces = []
        self.visibility_flags = visibility_flags
        self.channel_ids = channel_ids

        # preload the mesh to save loading time
        for m_file in mesh_files: 
            assert exists(m_file)
            # preload_verts_i, preload_faces_idx_i, _ = load_obj(m_file)
            preload_verts_i, preload_faces_idx_i = load_ply(m_file)
            # preload_faces_i = preload_faces_idx_i.verts_idx
            preload_faces_i = preload_faces_idx_i
            self.preload_verts.append(preload_verts_i)
            self.preload_faces.append(preload_faces_i)

        # set up differentiable renderer with given camera parameters
        self.cameras = PerspectiveCameras(
            focal_length=[focal_length],
            principal_point=[principal_point],
            device=device,
            in_ndc=False,
            image_size=[image_size],
        )  #  (height, width) !!!!!

        blend_params = BlendParams(sigma=1e-8, gamma=1e-8)
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
            # blur_radius=0.0,
            faces_per_pixel=10,
            max_faces_per_bin=18000,  # max_faces_per_bin=1000000,
        )

        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )

        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=100000,
        )
        # We can add a point light in front of the object.
        lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=device, cameras=self.cameras, lights=lights),
        )

        # # Set multiprocessing
        # mp.set_start_method('spawn')
        # self.pool = mp.Pool(mp.cpu_count())

    def set_mesh_visibility(self, visibility_flags):
        self.visibility_flags = visibility_flags

    def set_mesh_files(self, mesh_files):
        self.mesh_files = mesh_files

    def get_robot_mesh(self, joint_angle, ret_lndFK=False):

        # TODO: test
        # R_list, t_list = self.robot.get_joint_RT(joint_angle)
        R_list, t_list = lndFK(joint_angle)

        # assert len(self.mesh_files) == R_list.shape[0] and len(self.mesh_files) == t_list.shape[0]

        verts_list = []
        faces_list = []
        verts_rgb_list = []
        verts_count = 0

        for i in range(len(self.mesh_files)):
            if not self.visibility_flags[i]:
                continue

            verts_i = self.preload_verts[i].to(joint_angle.device)
            faces_i = self.preload_faces[i].to(joint_angle.device)

            # R = torch.tensor(R_list[i],dtype=torch.float32)
            # t = torch.tensor(t_list[i],dtype=torch.float32)
            R = R_list[i].clone().to(torch.float32)
            t = t_list[i].clone().to(torch.float32)
            # print(f"R requires grad: {R.requires_grad}")  # Should be True
            # print(f"t requires grad: {t.requires_grad}")  # Should be True
            # print(f'Debugging dtype: {R.dtype} and {verts_i.dtype}')
            verts_i = verts_i @ R.T + t
            # verts_i = (R @ verts_i.T).T + t
            faces_i = faces_i + verts_count

            verts_count += verts_i.shape[0]

            verts_list.append(verts_i.to(self.device))
            faces_list.append(faces_i.to(self.device))

            # Initialize each vertex to be white in color.
            color = torch.rand(3).to(joint_angle.device)
            verts_rgb_i = torch.ones_like(verts_i) * color  # (V, 3)
            verts_rgb_list.append(verts_rgb_i.to(self.device))

        verts = torch.concat(verts_list, dim=0)
        faces = torch.concat(faces_list, dim=0)

        # print(verts.shape, faces.shape)

        verts_rgb = torch.concat(verts_rgb_list, dim=0)[None]
        textures = Textures(verts_rgb=verts_rgb)

        # Create a Meshes object
        robot_mesh = Meshes(
            verts=[verts.to(self.device)],
            faces=[faces.to(self.device)],
            textures=textures,
        )

        if ret_lndFK:
            return robot_mesh, R_list, t_list
        else:
            return robot_mesh

    @torch.compile()
    def get_robot_verts_and_faces(self, joint_angle):
        R_list, t_list = lndFK(joint_angle)

        # assert len(self.mesh_files) == R_list.shape[0] and len(self.mesh_files) == t_list.shape[0]

        verts_list = []
        faces_list = []
        verts_rgb_list = []
        verts_count = 0

        for i in range(len(self.mesh_files)):
            if not self.visibility_flags[i]:
                continue

            verts_i = self.preload_verts[i].to(joint_angle.device)
            faces_i = self.preload_faces[i].to(joint_angle.device)

            R = R_list[i].clone().to(torch.float32)
            t = t_list[i].clone().to(torch.float32)
            verts_i = verts_i @ R.T + t
            faces_i = faces_i + verts_count

            verts_count += verts_i.shape[0]

            verts_list.append(verts_i.to(self.device))
            faces_list.append(faces_i.to(self.device))

            # # Initialize each vertex to be white in color.
            # color = torch.rand(3).to(joint_angle.device)
            # verts_rgb_i = torch.ones_like(verts_i) * color  # (V, 3)
            # verts_rgb_list.append(verts_rgb_i.to(self.device))

        verts = torch.concat(verts_list, dim=0)
        faces = torch.concat(faces_list, dim=0).to(torch.int32)

        return verts, faces

    @torch.compile()
    def batch_get_robot_verts_and_faces(self, joint_angles, ret_lndFK=False, ret_col=False, bi_manual=False):
        """
        Batched version of the get_robot_verts_and_faces method
        Args:
            joint_angles: (B, N) tensor of joint angles, or (B, 2, N) if bi_manual is True
            ret_lndFK: whether to return the per-joint R and t from lndFK
            ret_col: whether to render different arms at different color channels
            bi_manual: whether the input joint angles are for bi-manual setup (B, 2, N)
        Returns:
            verts: (B, V, 3) tensor of vertex positions
            faces: (F, 3) tensor of face indices (same for all instances)
            R_list, t_list (optional): per-joint rotation and translation from lndFK, shapes (B, M, 3, 3), (B, M, 3) or (B, 2, M, 3, 3), (B, 2, M, 3) if bi_manual is True
            colors (optional): (B, V, 2) tensor of attribute indicating arm identity (0 for left, 1 for right)
        """
        if bi_manual:
            B, _, _ = joint_angles.shape
        else:
            B, _ = joint_angles.shape
        device = joint_angles.device

        # 1) compute all R and t in one call: shapes [B, M, 3, 3], [B, M, 3]
        if bi_manual:
            R_list, t_list = batch_lndFK(joint_angles.view(-1, joint_angles.shape[-1])) # (B*2, M, 3, 3), (B*2, M, 3)
            R_list = R_list.view(B, 2 * R_list.shape[1], 3, 3) # (B, 2* M, 3, 3)
            t_list = t_list.view(B, 2 * t_list.shape[1], 3) # (B, 2* M, 3)
        else:
            R_list, t_list = batch_lndFK(joint_angles) # (B, M, 3, 3), (B, M, 3)
        R_list = R_list.to(torch.float32)
        t_list = t_list.to(torch.float32)

        # 2) precompute the global faces tensor (same for all batches)
        faces_accum = []
        faces_accum_2 = []
        cum_verts = 0
        colors = torch.zeros((B, 0, 2), device=device)  # for part identity coloring
        for verts_i, faces_i, vis in zip(self.preload_verts, self.preload_faces, self.visibility_flags):
            if not vis:
                continue
            faces_accum.append((faces_i.to(device) + cum_verts).to(torch.int32))
            cum_verts += verts_i.shape[0]
            if ret_col:
                # Create color tensor for this part
                part_color = torch.zeros((1, verts_i.shape[0], 2), device=device)
                part_color[:, :, 0] = 1.0  # One-hot encoding for part identity
                colors = torch.cat([colors, part_color.repeat(B, 1, 1)], dim=1)
        if bi_manual:
            # Repeat the process for the second robot
            for verts_i, faces_i, vis in zip(self.preload_verts, self.preload_faces, self.visibility_flags):
                if not vis:
                    continue
                faces_accum.append((faces_i.to(device) + cum_verts).to(torch.int32))
                cum_verts += verts_i.shape[0]
                if ret_col:
                    # Create color tensor for this part
                    part_color = torch.zeros((1, verts_i.shape[0], 2), device=device)
                    part_color[:, :, 1] = 1.0  # One-hot encoding for part identity
                    colors = torch.cat([colors, part_color.repeat(B, 1, 1)], dim=1)
                
        faces = torch.cat(faces_accum, dim=0)  # shape [F, 3]

        # 3) transform and collect per-mesh verts across the batch
        verts_accum = []
        for i, (verts_i, vis) in enumerate(zip(self.preload_verts, self.visibility_flags)):
            if not vis:
                continue
            # verts_i: [V_i, 3]
            verts_i = verts_i.to(device).to(torch.float32)

            # pick out the i-th transform for every batch element
            # R_list[:, i]: [B, 3, 3],  t_list[:, i]: [B, 3]
            Rb = R_list[:, i]        # [B, 3, 3]
            tb = t_list[:, i]        # [B, 3]

            # apply rotation and translation in batch:
            # rotated[b] = verts_i @ Rb[b].T  --> use einsum
            rotated = torch.einsum('vk,bwk->bvw', verts_i, Rb)  # [B, V_i, 3]
            verts_b = rotated + tb[:, None, :]                  # [B, V_i, 3]

            verts_accum.append(verts_b)
        if bi_manual:
            for i, (verts_i, vis) in enumerate(zip(self.preload_verts, self.visibility_flags)):
                if not vis:
                    continue
                # verts_i: [V_i, 3]
                verts_i = verts_i.to(device).to(torch.float32)

                # pick out the (i + M)-th transform for every batch element
                # R_list[:, i + M]: [B, 3, 3],  t_list[:, i + M]: [B, 3]
                Rb = R_list[:, i + len(self.preload_verts)]        # [B, 3, 3]
                tb = t_list[:, i + len(self.preload_verts)]        # [B, 3]

                # apply rotation and translation in batch:
                # rotated[b] = verts_i @ Rb[b].T  --> use einsum
                rotated = torch.einsum('vk,bwk->bvw', verts_i, Rb)  # [B, V_i, 3]
                verts_b = rotated + tb[:, None, :]                  # [B, V_i, 3]

                verts_accum.append(verts_b)

        # 4) concatenate all per-mesh verts along the vertex dimension
        verts = torch.cat(verts_accum, dim=1)  # [B, V, 3]

        if bi_manual:
            R_list = R_list.view(B, 2, -1, 3, 3)  # (B, 2, M, 3, 3)
            t_list = t_list.view(B, 2, -1, 3)      # (B, 2, M, 3)

        ret = None
        if ret_lndFK:
            ret = verts, faces, R_list, t_list
        else:
            ret = verts, faces

        if ret_col:
            ret = ret + (colors.contiguous(),)

        return ret

