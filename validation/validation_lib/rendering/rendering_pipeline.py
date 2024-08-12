from typing import Any

import numpy as np
import skvideo.io as video
import torch
from loguru import logger
from PIL import Image
from validation_lib.rendering.gs_camera import OrbitCamera
from validation_lib.rendering.gs_renderer import GaussianRenderer
from validation_lib.utils import preprocess_dream_gaussian_output


class RenderingPipeline:
    """Class that provides access to the implemented rendering pipelines."""

    def __init__(self, views: int, mode: str = "gs") -> None:
        """
        Parameters
        ----------
        mode:  selected rendering pipeline. options: "gs" - gaussian splatting
        """

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._views = views

        if mode == "gs":
            self._render = GaussianRenderer()
        else:
            raise ValueError("Only Gaussian Splatting (gs) rendering is currently supported.")

        self._thetas, self._phis = self.get_cameras_distribution2(views)

    def render_gaussian_splatting_views(
        self,
        data: dict,
        img_width: int,
        img_height: int,
        cam_rad: float = 3.5,
        cam_fov: float = 49.1,
        cam_znear: float = 0.01,
        cam_zfar: float = 100,
        data_ver: int = 1,
    ) -> list[torch.Tensor]:
        """Function for rendering multiple views of the preloaded Gaussian Splatting model

        Parameters
        ----------
        data: dictionary with input unpacked data
        img_width: the width of the rendering image
        img_height: the height of the rendering image
        cam_rad: the radius of the sphere where camera will be placed & moved along
        cam_fov: the field of view for the camera
        cam_znear: the position of the near camera plane along Z-axis
        cam_zfar: the position of the far camera plane along Z-axis
        data_ver: version of the input data format: 0 - corresponds to dream gaussian
                                                    1 - corresponds to new data format (default)

        Returns
        -------
        rendered_images: list with rendered images stored as PIL.Image
        """

        # setting up the camera
        camera = OrbitCamera(img_width, img_height, cam_fov, cam_znear, cam_zfar)

        # setting up tensors for storing camera transforms
        camera_views_proj = torch.empty((self._views, 4, 4)).to(self._device)
        camera_intrs = torch.empty((self._views, 3, 3)).to(self._device)

        for theta, phi, j in zip(self._thetas, self._phis, range(self._views), strict=False):
            dtheta = np.random.uniform(-5, 5)
            camera.compute_transform_orbit(phi, theta + dtheta, cam_rad, is_degree=True)
            camera_views_proj[j] = camera.world_to_camera_transform
            camera_intrs[j] = camera.intrinsics

        # data conversion (if we use dream gaussian project, tmp)
        if data_ver <= 1:
            data_proc = preprocess_dream_gaussian_output(data, camera.camera_position)
        else:
            data_proc = data

        # converting input data to tensors on GPU
        means3D = torch.tensor(data_proc["points"], dtype=torch.float32).contiguous().squeeze().to(self._device)
        rotations = torch.tensor(data_proc["rotation"], dtype=torch.float32).contiguous().squeeze().to(self._device)
        scales = torch.tensor(data_proc["scale"], dtype=torch.float32).contiguous().squeeze().to(self._device)
        opacity = torch.tensor(data_proc["opacities"], dtype=torch.float32).contiguous().squeeze().to(self._device)
        rgbs = torch.tensor(data_proc["features_dc"], dtype=torch.float32).contiguous().squeeze().to(self._device)

        # preparing data to send for rendering
        gaussian_data = [means3D, rotations, scales, opacity, rgbs]

        rendered_images, rendered_alphas, rendered_depths = self._render.render(
            camera_views_proj,
            camera_intrs,
            (camera.image_width, camera.image_height),
            camera.z_near,
            camera.z_far,
            gaussian_data,
        )
        # converting tensors to image-like tensors, keep all of them in device memory
        result_rendered_images = [(img * 255).to(torch.uint8) for img in rendered_images]

        logger.info(" Done.")
        return result_rendered_images

    @staticmethod
    def render_video_to_images(video_file: str) -> list[Image.Image]:
        """Function for converting video to images

        Parameters
        ----------
        video_file: path to the video file that will be loaded and converted to a sequence of images

        Returns
        -------
        images: a list with images stored as PIL.Image
        """
        video_data = video.vread(video_file)
        images = [Image.fromarray(video_data[i, :, :, :]) for i in range(video_data.shape[0])]
        return images

    def save_rendered_images(self, images: torch.Tensor | list[torch.Tensor], file_name: str, path: str) -> None:
        """Function for saving rendered images

        Parameters
        ----------
        images: list of images stored as PIL.Image
        file_name: the name of the file that being processed
        path: path to the folder where the rendered images will be stored

        """

        images_np = [(img.detach().cpu().numpy()).astype(np.uint8) for img in images]
        images_pil = [Image.fromarray(img) for img in images_np]
        self._render.save_images(images_pil, file_name, path)

    @staticmethod
    def get_cameras_distribution1(views: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Fibonacci points distribution on the sphere.

        Parameters
        ----------
        views: the amount of views for which the position on the sphere will be generated

        Returns
        -------
        thetas: numpy array with generated theta angles (azimuth angles) according to the distribution
        phis: numpy array with generated theta angles (elevation angles) according to the distribution
        """
        thetas = []
        phis = []

        phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians

        for i in range(views):
            y = 1 - (i / float(views - 1)) * 2  # y goes from 1 to -1
            theta_angle = phi * i  # golden angle increment

            # Calculate spherical coordinates, -80, 80 phi angles
            phi_tmp = (np.rad2deg(np.arccos(y)) / 180.0) * 160.0 - 80.0
            theta = np.rad2deg(theta_angle % (2 * np.pi)) % 360

            thetas.append(theta)
            phis.append(phi_tmp)
        return np.array(thetas), np.array(phis)

    @staticmethod
    def get_cameras_distribution2(views: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Spiral based points distribution on sphere.

        Parameters
        ----------
        views: the amount of views for which the position on the sphere will be generated

        Returns
        -------
        thetas: numpy array with generated theta angles (azimuth angles) according to the distribution
        phis: numpy array with generated theta angles (elevation angles) according to the distribution
        """
        n = views
        x = 0.1 + 1.2 * views
        thetas = []
        phis = []
        start = -1.0 + 1.0 / (n - 1.0)
        increment = (2.0 - 2.0 / (n - 1.0)) / (n - 1.0)
        for j in range(0, n):
            s = start + j * increment
            theta = np.rad2deg(np.pi / 2.0 * np.copysign(1, s) * (1.0 - np.sqrt(1.0 - abs(s)))) % 360
            phi = np.rad2deg(s * x) - 90
            thetas.append(theta)
            phis.append(phi)

        return np.array(thetas), np.array(phis)

    @staticmethod
    def get_cameras_distribution3(views: int) -> tuple[list[Any], list[Any]]:
        """
        Equal angularly distanced points distribution on sphere

        Parameters
        ----------
        views: the amount of views for which the position on the sphere will be generated

        Returns
        -------
        thetas: numpy array with generated theta angles (azimuth angles) according to the distribution
        phis: numpy array with generated theta angles (elevation angles) according to the distribution
        """
        dlong = np.pi * (3.0 - np.sqrt(5.0))  # ~2.39996323
        dz = 2.0 / views
        long = 0.0
        z = 1.0 - dz / 2.0

        phis = []
        thetas = []

        for _ in range(0, views):
            phi = np.rad2deg(np.arccos(z)) - 90  # polar angle
            theta = np.rad2deg(long % (2 * np.pi)) % 360  # azimuthal angle

            z = z - dz
            long = long + dlong

            phis.append(phi)
            thetas.append(theta)

        return thetas, phis

    @staticmethod
    def get_cameras_distribution4(views: int) -> tuple[np.ndarray, np.ndarray]:
        """
        "Sunflower" points distribution on the sphere

        Parameters
        ----------
        views: the amount of views for which the position on the sphere will be generated

        Returns
        -------
        thetas: numpy array with generated theta angles (azimuth angles) according to the distribution
        phis: numpy array with generated theta angles (elevation angles) according to the distribution
        """
        indices = np.arange(0, views, dtype=float) + 0.5
        phis = np.arccos(1 - 2 * indices / views)
        thetas_prep = np.pi * (1 + 5**0.5) * indices

        phis = [np.rad2deg(phi) - 90 for phi in phis]
        thetas = [np.rad2deg(theta) % 360 for theta in thetas_prep]

        return np.array(thetas), np.array(phis)
