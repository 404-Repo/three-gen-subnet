import numpy as np
import torch
from gsplat.rendering import rasterization

from engine.data_structures import GaussianSplattingData
from engine.rendering.gaussian_splatting.gs_camera import OrbitCamera


class GaussianSplattingRenderer:
    """Class that implements gaussian splatting rasterization and rendering implementation"""

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)

        self._camera_views_proj: torch.Tensor | None = None
        self._camera_intrinsics: torch.Tensor | None = None
        self._camera: OrbitCamera | None = None
        self._bg_color = torch.tensor([1, 1, 1], dtype=torch.float32).to(self._device)

    def setup_cameras(
        self,
        views_number: int,
        img_width: int,
        img_height: int,
        theta_angles: list | None = None,
        phi_angles: list | None = None,
        cam_rad: float = 2.5,
        cam_fov: float = 49.1,
        cam_znear: float = 0.01,
        cam_zfar: float = 100.0,
    ) -> None:
        """Function for setting up all rendering cameras"""
        # setting up the camera
        self._camera = OrbitCamera(img_width, img_height, cam_fov, cam_znear, cam_zfar)

        # setting up tensors for storing camera transforms
        self._camera_views_proj = torch.empty((views_number, 4, 4))
        self._camera_intrinsics = torch.empty((views_number, 3, 3))

        # setting up rendering radial angles
        if theta_angles is None:
            thetas = np.linspace(0, 360, num=views_number)
        else:
            thetas = np.array(theta_angles)

        if phi_angles is None:
            phis = np.full_like(thetas, -15.0)
        else:
            phis = np.array(phi_angles)

        if len(phis) != len(thetas):
            raise ValueError("Input list with phi and theta angles should have the same length.")

        for theta, phi, j in zip(thetas, phis, range(views_number), strict=False):
            self._camera.compute_transform_orbit(phi, theta, cam_rad, is_degree=True)
            self._camera_views_proj[j] = self._camera.world_to_camera_transform
            self._camera_intrinsics[j] = self._camera.intrinsics

    def render(self, gs_data: GaussianSplattingData, bg_color: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Function for rendering gaussian splatting model"""

        if (self._camera_views_proj is None) or (self._camera_intrinsics is None):
            raise RuntimeError("Cameras have not been initialized prior calling to 'render' function.")

        rendered_images, rendered_alphas = self._rasterize_views(gs_data, bg_color)

        # converting tensors to image-like tensors, keep all of them in device memory
        output_rendered_images = [(img * 255).to(torch.uint8) for img in rendered_images]
        return output_rendered_images

    def _rasterize_views(
        self, gs_data: GaussianSplattingData, bg_color: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Function for rasterization of the gaussian splatting model"""

        if self._camera is None:
            raise RuntimeError("The camera object was not initialized.")
        if self._camera_views_proj is None:
            raise RuntimeError("Camera projection matrices have not been computed.")
        if self._camera_intrinsics is None:
            raise RuntimeError("Camera intrinsic matrices have not been computed.")

        # Pre-allocate tensors
        num_cameras = self._camera_views_proj.shape[0]
        background_col = self._bg_color if bg_color is None else bg_color
        backgrounds = background_col.expand(num_cameras, *background_col.shape)

        rendered_colors, rendered_alphas, meta = rasterization(
            gs_data.points,
            gs_data.rotations,
            gs_data.scales,
            gs_data.opacities,
            gs_data.features_dc,
            self._camera_views_proj,
            self._camera_intrinsics,
            self._camera.image_width,
            self._camera.image_height,
            self._camera.z_near,
            self._camera.z_far,
            backgrounds=backgrounds,
            render_mode="RGB",
        )

        if rendered_colors.shape != (num_cameras, self._camera.image_width, self._camera.image_height, 3):
            raise ValueError(f"Unexpected shape for rendered_colors: {rendered_colors.shape}")
        if rendered_alphas.shape != (num_cameras, self._camera.image_width, self._camera.image_height, 1):
            raise ValueError(f"Unexpected shape for rendered_alphas: {rendered_alphas.shape}")

        rendered_images = rendered_colors.clip(0, 1)

        return rendered_images, rendered_alphas
