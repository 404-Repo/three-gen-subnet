from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from engine.data_structures import GaussianSplattingData
from engine.rendering.gaussian_splatting.gs_renderer import GaussianSplattingRenderer
from engine.rendering.gaussian_splatting.gs_utils import transform_gs_data


class Renderer:
    """Class that holds instances of different rendering pipelines"""

    def __init__(self) -> None:
        self._gs_renderer = GaussianSplattingRenderer()

    def render_gs(
        self,
        gs_data: GaussianSplattingData,
        views_number: int,
        img_width: int,
        img_height: int,
        theta_angles: list | np.ndarray | None = None,
        phi_angles: list | np.ndarray | None = None,
        cam_rad: float = 2.5,
        cam_fov: float = 49.1,
        cam_znear: float = 0.01,
        cam_zfar: float = 100.0,
        bg_color: torch.Tensor | None = None,
        ref_bbox_size: float = 1.5,
    ) -> list[torch.Tensor]:
        """Function for rendering gaussian splatting model"""

        self._gs_renderer.setup_cameras(
            views_number, img_width, img_height, theta_angles, phi_angles, cam_rad, cam_fov, cam_znear, cam_zfar
        )

        gs_data_tr = transform_gs_data(gs_data, ref_bbox_size)
        rendered_images = self._gs_renderer.render(gs_data_tr, bg_color)

        return rendered_images

    @staticmethod
    def save_rendered_images(images_torch: list[torch.Tensor], file_name: str, path: str) -> None:
        """Function for saving rendered images that are defined as PIL images"""

        images_np = [(img.detach().cpu().numpy()).astype(np.uint8) for img in images_torch]
        images_pil = [Image.fromarray(img) for img in images_np]

        save_path = Path(path)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(images_pil):
            img_name = save_path / f"{file_name}_{i}.png"
            image.save(img_name)

    @staticmethod
    def save_gif(images: list[torch.Tensor], gif_name: str, file_path: Path, duration: float = 1.5) -> None:
        """Function for generating gif from rendered images"""

        gif_name = gif_name + ".gif"
        gif_path = file_path / gif_name

        images_pil = [img.detach().cpu().numpy() for img in images]
        iio.imwrite(gif_path.as_posix(), images_pil, duration=duration, loop=0)
