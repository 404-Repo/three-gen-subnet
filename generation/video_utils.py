import os
from io import BytesIO

import torch
import numpy as np
import imageio
import tqdm

from DreamGaussianLib.rendering.gs_camera import OrbitCamera
from DreamGaussianLib.rendering.gs_renderer import GaussianRenderer


class VideoUtils:
    def __init__(
        self,
        img_width: int = 512,
        img_height: int = 512,
        cam_rad: float = 4.0,
        azim_step: int = 5,
        elev_step: int = 20,
        elev_start=-60,
        elev_stop=30,
    ):
        self._img_width = img_width
        self._img_height = img_height
        self._cam_rad = cam_rad
        self._azim_step = azim_step
        self._elev_step = elev_step
        self._elev_start = elev_start
        self._elev_stop = elev_stop
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def render_video(
        self,
        points: np.ndarray,
        normals: np.ndarray | None,
        features_dc: np.ndarray,
        features_rest: np.ndarray | None,
        opacities: np.ndarray,
        scale: np.ndarray,
        rotation: np.ndarray,
        sh_degree: int | None,
    ) -> tuple[BytesIO, list[np.ndarray]]:

        means3D = torch.tensor(points, dtype=torch.float32).contiguous().squeeze().to(self._device)
        rotations = torch.tensor(rotation, dtype=torch.float32).contiguous().squeeze().to(self._device)
        scales = torch.tensor(scale, dtype=torch.float32).contiguous().squeeze().to(self._device)
        opacity = torch.tensor(opacities, dtype=torch.float32).contiguous().squeeze().to(self._device)
        rgbs = torch.tensor(features_dc, dtype=torch.float32).contiguous().squeeze().to(self._device)

        gs_data = [means3D, rotations, scales, opacity, rgbs]

        renderer = GaussianRenderer()
        orbit_cam = OrbitCamera(self._img_width, self._img_height)

        images = []
        for elev in range(self._elev_start, self._elev_stop, self._elev_step):
            for azimd in range(0, 360, self._azim_step):
                orbit_cam.compute_transform_orbit(elev, azimd, self._cam_rad)
                img, _, _, _ = renderer.render(
                    orbit_cam.world_to_camera_transform.unsqueeze(0),
                    orbit_cam.intrinsics.unsqueeze(0),
                    (orbit_cam.image_width, orbit_cam.image_height),
                    orbit_cam.z_near,
                    orbit_cam.z_far,
                    gs_data)

                # img = output_dict["image"].permute(1, 2, 0)
                img = img.detach().cpu().squeeze(0).numpy() * 255
                img = img.astype(np.uint8)
                images.append(img)

        buffer = BytesIO()
        with imageio.get_writer(buffer, format="mp4", mode="I", fps=24) as writer:
            for img in images:
                writer.append_data(img)

        buffer.seek(0)

        return buffer, images

    @staticmethod
    def save_video(image_list: list, fps: int, path: os.path):
        writer = imageio.get_writer(path, fps=fps)
        for img, _ in zip(image_list, tqdm.trange(len(image_list))):
            writer.append_data(img)
        writer.close()
