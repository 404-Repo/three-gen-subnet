from pathlib import Path

import torch
from gsplat.rendering import rasterization
from PIL import Image


class GaussianRenderer:
    """Class with implementation of the Gaussian Splatting Renderer"""

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._bg_color = torch.tensor([1, 1, 1, 1], dtype=torch.float32, device=self._device)

    def render(
        self,
        cameras_view_proj: torch.Tensor,
        cameras_intr: torch.Tensor,
        img_res: tuple[int, int],
        z_near: float,
        z_far: float,
        gaussian_data: list[torch.Tensor],
        bg_color: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Function that renders input gaussian splatting data using single camera view
        or multiple camera views as a single batch

        Parameters
        ----------
        cameras_view_proj: torch tensor with camera projection matrix per input view
        cameras_intr: torch tensor with camera intrinsic parameters matrix per input view
        img_res: a tuple with integer values defining image width and image height
        z_near: the float position of the camera near plane
        z_far: the float position of the camera far plane
        gaussian_data: a list of tensors with gaussian splatting data pre-loaded to the same device
        bg_color: a torch tensor that will define a background colour for rendered images (optional)

        Returns
        -------
        rendered_images - torch tensor with rendered image/s
        rendered_alphas - torch tensor with rendered image/s (alpha channel only)
        rendered_depths - torch tensor with rendered image/s (depth maps only)
        """

        means3D = gaussian_data[0]
        rotations = gaussian_data[1]
        scales = gaussian_data[2]
        opacity = gaussian_data[3]
        rgbs = gaussian_data[4]

        # Pre-allocate tensors
        num_cameras = len(cameras_view_proj)
        background_col = self._bg_color if bg_color is None else bg_color
        backgrounds = background_col
        if background_col is not None:
            backgrounds = background_col.expand(num_cameras, *background_col.shape).to(self._device)

        rendered_colors, rendered_alphas, meta = rasterization(
            means3D,
            rotations,
            scales,
            opacity,
            rgbs,
            cameras_view_proj,
            cameras_intr,
            img_res[0],
            img_res[1],
            z_near,
            z_far,
            backgrounds=backgrounds,
            render_mode="RGB+D",
        )

        if rendered_colors.shape != (num_cameras, img_res[0], img_res[1], 4):
            raise ValueError(f"Unexpected shape for rendered_colors: {rendered_colors.shape}")
        if rendered_alphas.shape != (num_cameras, img_res[0], img_res[1], 1):
            raise ValueError(f"Unexpected shape for rendered_alphas: {rendered_alphas.shape}")

        rendered_images = rendered_colors[..., 0:3].clip(0, 1)
        rendered_depths = rendered_colors[..., 3:4]
        rendered_depths = rendered_depths / rendered_depths.max()

        return rendered_images, rendered_alphas, rendered_depths

    @staticmethod
    def save_images(images: list[Image.Image], file_name: str, path: str) -> None:
        """Function for saving rendered images that are defined as PIL images

        Parameters
        ----------
        images: a list of the images stored as PIL.Image
        file_name:  the name of the rendered object
        path: the path to the folder where the images will be saved

        """
        save_path = Path(path)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(images):
            img_name = save_path / f"{file_name}_{i}.png"
            image.save(img_name)
