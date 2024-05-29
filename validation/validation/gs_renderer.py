import os
from typing import List

import torch
from PIL import Image

from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         GaussianRasterizer)

from validation.gs_camera import OrbitCamera


class GaussianRenderer:
    """ Class with implementation of the Gaussian Splatting Renderer """
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=self._device)

    def render(self, camera: OrbitCamera, gaussian_data: dict, bg_color: torch.tensor = None, scale_modifier: float = 1):
        """ Function that render single view for the input data

        Parameters
        ----------
        camera: camera object that will be used for rendering the input data (gaussian splatting model)
        gaussian_data: the dictionary with gaussian splatting data
        bg_color: a torch tensor that will define a background colour for rendered images (optional)
        scale_modifier: a scaling factor that controls the size of the Gaussians

        Returns
        -------
        rendered_image - torch tensor with rendered image
        rendered_alpha - torch tensor with rendered alpha channel
        rendered_depth - torch tensor image with rendered depth map
        """

        means3D = torch.tensor(gaussian_data["points"], dtype=torch.float32).contiguous().squeeze().to(self._device)
        rotations = torch.tensor(gaussian_data["rotation"], dtype=torch.float32).contiguous().squeeze().to(self._device)
        scales = torch.tensor(gaussian_data["scale"], dtype=torch.float32).contiguous().squeeze().to(self._device)
        opacity = torch.tensor(gaussian_data["opacities"], dtype=torch.float32).contiguous().squeeze().to(self._device)
        rgbs = torch.tensor(gaussian_data["features_dc"], dtype=torch.float32).contiguous().squeeze().to(self._device)

        camera_view = camera.world_view_transform
        camera_view_proj = camera.full_projection_transform
        camera_pos = camera.camera_position

        raster_settings = GaussianRasterizationSettings(
            image_height=camera.image_height,
            image_width=camera.image_width,
            tanfovx=camera.tan_half_fov,
            tanfovy=camera.tan_half_fov,
            bg=self._bg_color if bg_color is None else bg_color,
            scale_modifier=scale_modifier,
            viewmatrix=camera_view,
            projmatrix=camera_view_proj,
            sh_degree=0,
            campos=camera_pos,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=torch.zeros_like(means3D, dtype=torch.float32, requires_grad=True).to(self._device),
            shs=None,
            colors_precomp=rgbs,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )

        rendered_image = rendered_image.clamp(0, 1)

        return rendered_image, rendered_alpha, rendered_depth

    @staticmethod
    def save_images(images: List[Image.Image], file_name: str, path: str):
        """ Function for saving rendered images that are defined as PIL images

        Parameters
        ----------
        images: a list of the images stored as PIL.Image
        file_name:  the name of the rendered object
        path: the path to the folder where the images will be saved

        """
        for i, image in enumerate(images):
            img_name = os.path.join(path, file_name+"_"+str(i)+".png")
            image.save(img_name)
