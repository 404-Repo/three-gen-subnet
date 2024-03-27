import math
import numpy as np
from typing import NamedTuple
from dataclasses import dataclass

import torch
from torch import nn

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from simple_knn._C import distCUDA2

from DreamGaussianLib.SphericalHarmonics import eval_sh, SH2RGB, RGB2SH


class GSUtils:
    def __init__(self):
        pass

    def build_rotation(self, r: torch.Tensor):
        norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

        q = r / norm[:, None]
        R = torch.zeros((q.size(0), 3, 3), device="cuda")

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - r * z)
        R[:, 0, 2] = 2 * (x * z + r * y)
        R[:, 1, 0] = 2 * (x * y + r * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - r * x)
        R[:, 2, 0] = 2 * (x * z - r * y)
        R[:, 2, 1] = 2 * (y * z + r * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    def build_scaling_rotation(self, s: torch.Tensor, r: torch.Tensor):
        L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
        R = self.build_rotation(r)

        L[:, 0, 0] = s[:, 0]
        L[:, 1, 1] = s[:, 1]
        L[:, 2, 2] = s[:, 2]

        L = R @ L
        return L

    def strip_tensot_lowerdiag(self, L: torch.Tensor):
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]
        return uncertainty

    def build_covariance_from_scaling_rotation(
        self, scaling: torch.Tensor, scaling_modifier: float, rotation: torch.Tensor
    ):
        L = self.build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = self.strip_tensot_lowerdiag(actual_covariance)
        return symm

    def inverse_sigmoid(self, x: torch.Tensor):
        return torch.log(x / (1 - x))


#######################################################################################################################################################################
@dataclass
class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class BasicGSModel:
    def __init__(self, sh_degree: int):
        self.__active_sh_degree = 0
        self.__max_sh_degree = sh_degree
        self.__xyz = torch.empty(0)
        self.__features_dc = torch.empty(0)
        self.__features_rest = torch.empty(0)
        self.__scaling = torch.empty(0)
        self.__rotation = torch.empty(0)
        self.__opacity = torch.empty(0)
        self.__max_radii2D = torch.empty(0)
        self.__xyz_gradient_accum = torch.empty(0)
        self.__denom = torch.empty(0)
        self.__spatial_lr_scale = 0

        self.__optimizer = None
        self.__percent_dense = 0
        self.__spatial_lr_scale = 0

        self.__gs_utils = GSUtils()

        # setting up function calls (shortcuts)
        self._scaling_activation = torch.exp
        self._scaling_inv_activation = torch.log
        self._covariance_activation = self.__gs_utils.build_covariance_from_scaling_rotation
        self._opacity_activation = torch.sigmoid
        self._inv_opacity_activation = self.__gs_utils.inverse_sigmoid
        self._rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self._scaling_activation(self.__scaling)

    @property
    def get_rotation(self):
        return self._rotation_activation(self.__rotation)

    @property
    def get_xyz(self):
        return self.__xyz

    @property
    def get_active_sh_degree(self):
        return self.__active_sh_degree

    @property
    def get_opacity(self):
        return self._opacity_activation(self.__opacity)

    @property
    def get_features(self):
        features_dc = self.__features_dc
        features_rest = self.__features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_max_sh_degree(self):
        return self.__max_sh_degree

    def create_from_point_cloud(self, pcd: BasicPointCloud, spatial_lr_scale: float = 1.0):
        self.__spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.__max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.__gs_utils.inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self.__xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.__features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.__features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.__scaling = nn.Parameter(scales.requires_grad_(True))
        self.__rotation = nn.Parameter(rots.requires_grad_(True))
        self.__opacity = nn.Parameter(opacities.requires_grad_(True))
        self.__max_radii2D = torch.zeros((self.__xyz.shape[0]), device="cuda")

    def create_from_dictionary(self, data_dict: dict):
        xyz = data_dict["points"]
        opacities = data_dict["opacities"]
        features_dc = data_dict["features_dc"]
        features_rest = data_dict["features_rest"]
        scaling = data_dict["scale"]
        rotation = data_dict["rotation"]
        sh_degree = data_dict["sh_degree"]

        self.__xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self.__features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True)
        )
        self.__features_rest = nn.Parameter(
            torch.tensor(features_rest, dtype=torch.float, device="cuda").contiguous().requires_grad_(True)
        )
        self.__opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self.__scaling = nn.Parameter(torch.tensor(scaling, dtype=torch.float, device="cuda").requires_grad_(True))
        self.__rotation = nn.Parameter(torch.tensor(rotation, dtype=torch.float, device="cuda").requires_grad_(True))
        self.__active_sh_degree = sh_degree

    def get_covariance(self, scaling_modifier: float = 1.0):
        return self._covariance_activation(self.__scaling, scaling_modifier, self.__rotation)


########################################################################################################################################################################

"""
"""


class BasicCamera:
    def __init__(
        self,
        cam2world,
        width: int,
        height: int,
        fovy: float,
        fovx: float,
        znear: float,
        zfar: float,
    ):
        self.__image_width = width
        self.__image_height = height
        self.__FoVy = fovx
        self.__FoVx = fovy
        self.__znear = znear
        self.__zfar = zfar

        world2cam = np.linalg.inv(cam2world)

        # rectify...
        world2cam[1:3, :3] *= -1
        world2cam[:3, 3] *= -1

        self.__world_view_transform = torch.tensor(world2cam).transpose(0, 1).cuda()
        self.__projection_matrix = (
            self._get_projection_matrix(znear=self.__znear, zfar=self.__zfar, fovX=self.__FoVx, fovY=self.__FoVy)
            .transpose(0, 1)
            .cuda()
        )

        self.__full_proj_transform = self.__world_view_transform @ self.__projection_matrix
        self.__camera_center = -torch.tensor(cam2world[:3, 3]).cuda()

    @staticmethod
    def _get_projection_matrix(znear: float, zfar: float, fovX: float, fovY: float):
        tanHalfFovY = math.tan((fovY / 2.0))
        tanHalfFovX = math.tan((fovX / 2.0))

        P = torch.zeros(4, 4)
        z_sign = 1.0

        P[0, 0] = 1.0 / tanHalfFovX
        P[1, 1] = 1.0 / tanHalfFovY
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    @property
    def get_center(self):
        return self.__camera_center

    @property
    def get_fov_x(self):
        return self.__FoVx

    @property
    def get_fov_y(self):
        return self.__FoVy

    @property
    def get_image_width(self):
        return self.__image_width

    @property
    def get_image_height(self):
        return self.__image_height

    @property
    def get_world_view_transform(self):
        return self.__world_view_transform

    @property
    def get_full_proj_transform(self):
        return self.__full_proj_transform

    @property
    def get_projection_matrix(self):
        return self.__projection_matrix


########################################################################################################################################################################

"""
"""


class GSRenderer:
    def __init__(self, sh_degree: int = 3, white_background: bool = True, radius: float = 1.0):
        self.__sh_degree = sh_degree
        self.__white_background = white_background
        self.__radius = radius

        self.__gs_model = BasicGSModel(sh_degree)

        self.__bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )

    def initialize(self, input=None, num_pts: int = 5000, radius: float = 0.5):
        # load checkpoint
        if input is None:
            # init from random point cloud

            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            self.__gs_model.create_from_point_cloud(pcd, 10)

        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            self.__gs_model.create_from_point_cloud(input, 1)
        else:
            self.__gs_model.create_from_dictionary(input)

    def render(
        self,
        viewpoint_camera: BasicCamera,
        scaling_modifier: float = 1.0,
        bg_color: torch.Tensor = None,
        override_color: torch.Tensor = None,
        compute_cov3d_python: bool = False,
        convert_shs_python: bool = False,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.__gs_model.get_xyz,
                dtype=self.__gs_model.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.get_fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.get_fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.get_image_height),
            image_width=int(viewpoint_camera.get_image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.__bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.get_world_view_transform,
            projmatrix=viewpoint_camera.get_full_proj_transform,
            sh_degree=self.__gs_model.get_active_sh_degree,
            campos=viewpoint_camera.get_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.__gs_model.get_xyz
        means2D = screenspace_points
        opacity = self.__gs_model.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3d_python:
            cov3D_precomp = self.__gs_model.get_covariance(scaling_modifier)
        else:
            scales = self.__gs_model.get_scaling
            rotations = self.__gs_model.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_shs_python:
                shs_view = self.__gs_model.get_features.transpose(1, 2).view(
                    -1, 3, (self.__gs_model.get_max_sh_degree + 1) ** 2
                )
                dir_pp = self.__gs_model.get_xyz - viewpoint_camera.get_center.repeat(
                    self.__gs_model.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)

                sh2rgb = eval_sh(self.__gs_model.get_active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.__gs_model.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
