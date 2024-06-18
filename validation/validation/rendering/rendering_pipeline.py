import io

import torch
import numpy as np
import skvideo.io as video
from PIL import Image
from loguru import logger

from validation.rendering.gs_renderer import GaussianRenderer
from validation.rendering.gs_camera import OrbitCamera
from validation.utils import preprocess_dream_gaussian_output


class RenderingPipeline:
    """Class that provides access to the implemented rendering pipelines."""

    def __init__(self, img_width: int, img_height: int, mode="gs"):
        """Constructor

        Parameters
        ----------
        img_width: the width of the rendering image
        img_height: the height of the rendering image
        mode:  selected rendering pipeline. options: "gs" - gaussian splatting
        """

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)

        self._img_width = img_width
        self._img_height = img_height

        # min and maximum elevation angles for the camera position during rendering
        self._cam_min_elev_angle = -17
        self._cam_max_elev_angle = 17

        if mode == "gs":
            self._render = GaussianRenderer()
        else:
            raise ValueError("Only Gaussian Splatting (gs) rendering is currently supported.")

    def render_gaussian_splatting_views(
        self,
        data: dict,
        views: int = 15,
        cam_rad: float = 3.5,
        cam_fov: float = 49.1,
        cam_znear: float = 0.01,
        cam_zfar: float = 100,
        gs_scale: float = 1.0,
        data_ver: int = 1,
    ):
        """Function for rendering multiple views of the preloaded Gaussian Splatting model

        Parameters
        ----------
        data: dictionary with input unpacked data
        views: the amount of views that will be rendered
        cam_rad: the radius of the sphere where camera will be placed & moved along
        cam_fov: the field of view for the camera
        cam_znear: the position of the near camera plane along Z-axis
        cam_zfar: the position of the far camera plane along Z-axis
        gs_scale: the scale of the Gaussians that will be used during rendering
        data_ver: version of the input data format: 0 - default dream gaussian format
                                                    1+ - preparation for PLY

        Returns
        -------
        list with rendered images stored as PIL.Image
        """
        logger.info(" Rendering view of the input Gaussian Splatting Model.")

        camera = OrbitCamera(self._img_width, self._img_height, cam_fov, cam_znear, cam_zfar)

        rendered_images = []
        random_angles_number = views // 2
        step = 360 // random_angles_number
        elevations = np.random.randint(self._cam_min_elev_angle, self._cam_max_elev_angle, random_angles_number)

        for i in range(2):
            for azimuth, elev in zip(range(0, 360, step), elevations):
                if i == 0:
                    camera.compute_transform_orbit(0, azimuth, cam_rad)
                else:
                    camera.compute_transform_orbit(elev, azimuth, cam_rad)

                if data_ver <= 1:
                    data_in = preprocess_dream_gaussian_output(data, camera.camera_position)
                else:
                    data_in = data

                rendered_image, rendered_alpha, rendered_depth = self._render.render(
                    camera, data_in, scale_modifier=np.clip(gs_scale, 0, 1)
                )

                image = rendered_image.detach().cpu().numpy() * 255
                image = image.astype(dtype=np.uint8)

                pil_image = Image.fromarray(image)
                rendered_images.append(pil_image)

        logger.info(" Done.")
        return rendered_images

    @staticmethod
    def render_video_to_images(video_file: str):
        """Function for converting video to images

        Parameters
        ----------
        video_file: path to the video file that will be loaded and converted to a sequence of images

        Returns
        -------
        a list with images stored as PIL.Image
        """
        logger.info(" Converting input video to list of images.")

        video_data = video.vread(video_file)
        images = [Image.fromarray(video_data[i, :, :, :]) for i in range(video_data.shape[0])]

        logger.info(" Done.")
        return images

    def save_rendered_images(self, images: list, file_name: str, path: str):
        """Function for saving rendered images

        Parameters
        ----------
        images: list of images stored as PIL.Image
        file_name: the name of the file that being processed
        path: path to the folder where the rendered images will be stored

        """
        self._render.save_images(images, file_name, path)
