import os
from io import BytesIO

import numpy as np
import imageio
import tqdm

from DreamGaussianLib.CameraUtils import OrbitCamera, orbit_camera
from DreamGaussianLib.GaussianSplattingRenderer import GSRenderer, BasicCamera


class VideoUtils:
    def __init__(
        self,
        img_width: int = 512,
        img_height: int = 512,
        cam_rad: float = 2,
        azim_step: int = 5,
        elev_step: int = 20,
        elev_start=-60,
        elev_stop=30,
    ):
        self.__img_width = img_width
        self.__img_height = img_height
        self.__cam_rad = cam_rad
        self.__azim_step = azim_step
        self.__elev_step = elev_step
        self.__elev_start = elev_start
        self.__elev_stop = elev_stop

    def render_video(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        features_dc: np.ndarray,
        features_rest: np.ndarray,
        opacities: np.ndarray,
        scale: np.ndarray,
        rotation: np.ndarray,
        sh_degree: int,
    ) -> BytesIO:
        data_dict = {
            "points": points,
            "normals": normals,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacities": opacities,
            "scale": scale,
            "rotation": rotation,
            "sh_degree": sh_degree,
        }
        renderer = GSRenderer()
        renderer.initialize(data_dict)

        orbitcam = OrbitCamera(self.__img_width, self.__img_height, r=self.__cam_rad, fovy=49.1)

        images = []
        for elev in range(self.__elev_start, self.__elev_stop, self.__elev_step):
            for azimd in range(0, 360, self.__azim_step):
                pose = orbit_camera(elev, azimd, self.__cam_rad)
                camera = BasicCamera(
                    pose,
                    self.__img_width,
                    self.__img_height,
                    orbitcam.fovy,
                    orbitcam.fovx,
                    orbitcam.near,
                    orbitcam.far,
                )

                output_dict = renderer.render(camera)
                img = output_dict["image"].permute(1, 2, 0)
                img = img.detach().cpu().numpy() * 255
                img = img.astype(np.uint8)
                images.append(img)

        buffer = BytesIO()
        with imageio.get_writer(buffer, format="mp4", mode="I", fps=24) as writer:
            for img in images:
                writer.append_data(img)

        buffer.seek(0)

        return buffer

    def _save_video(self, image_list: list, fps: int, path: os.path):
        writer = imageio.get_writer(path, fps=fps)
        for img, _ in zip(image_list, tqdm.trange(len(image_list))):
            writer.append_data(img)
        writer.close()
