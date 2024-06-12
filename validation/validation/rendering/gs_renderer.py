import os
from typing import List

import torch
from PIL import Image
from gsplat.rendering import rasterization, rasterization_inria_wrapper

from validation.rendering.gs_camera import OrbitCamera


class GaussianRenderer:
    """Class with implementation of the Gaussian Splatting Renderer"""

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._bg_color = torch.tensor([1, 1, 1, 1], dtype=torch.float32, device=self._device)

    def render(
        self, camera: OrbitCamera, gaussian_data: dict, bg_color: torch.Tensor | None = None, scale_modifier: float = 1
    ):
        """Function that render single view for the input data

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

        camera_view_proj = torch.unsqueeze(camera.world_to_camera_transform, 0)
        camera_intr = torch.unsqueeze(camera.intrinsics, 0)

        background_col = self._bg_color if bg_color is None else bg_color
        background = background_col.unsqueeze(0)

        rendered_colors, rendered_alpha, meta = rasterization(
            means3D,
            rotations,
            scales,
            opacity,
            rgbs,
            camera_view_proj,
            camera_intr,
            camera.image_width,
            camera.image_height,
            camera.z_near,
            camera.z_far,
            backgrounds=background,
            render_mode="RGB+D",
        )

        assert rendered_colors.shape == (1, camera.image_height, camera.image_width, 4)
        assert rendered_alpha.shape == (1, camera.image_height, camera.image_width, 1)

        rendered_image = rendered_colors[..., 0:3].squeeze()
        rendered_alpha = rendered_alpha.squeeze()
        rendered_depths = rendered_colors[..., 3:4].squeeze()
        rendered_depths = rendered_depths / rendered_depths.max()

        rendered_image = rendered_image.clamp(0, 1)

        return rendered_image, rendered_alpha, rendered_depths

    @staticmethod
    def save_images(images: List[Image.Image], file_name: str, path: str):
        """Function for saving rendered images that are defined as PIL images

        Parameters
        ----------
        images: a list of the images stored as PIL.Image
        file_name:  the name of the rendered object
        path: the path to the folder where the images will be saved

        """
        if not os.path.exists(path):
            os.mkdir(path)

        for i, image in enumerate(images):
            img_name = os.path.join(path, file_name + "_" + str(i) + ".png")
            image.save(img_name)
