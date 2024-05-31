import base64
import io
import sys

import torch
import numpy as np
import skvideo.io as video
from PIL import Image
from loguru import logger

from validation.gs_renderer import GaussianRenderer
from validation.gs_camera import OrbitCamera
from validation.hdf5_loader import HDF5Loader
from validation.utils import preproceses_dream_gaussian_output


class RenderingPipeline:
    """ Class that provides access to the implemented rendering pipelines. """
    def __init__(self, img_width: int, img_height: int, mode="gs"):
        """ Constructor

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
        self._hdf5_loader = HDF5Loader()

        # min and maximum elevation angles for the camera position during rendering
        self._cam_min_elev_angle = -15
        self._cam_max_elev_angle = 15

        if mode == "gs":
            self._render = GaussianRenderer()
        else:
            raise ValueError("Only Gaussian Splatting (gs) rendering is currently supported.")

    def prepare_data(self, data: bytes):
        """ Function for preloading input data (unpacking) and testing whether it fits in the GPU memory

        Parameters
        ----------
        data: input data stored as bytes.

        Returns
        -------
        True - if the data was loaded successfully and fit in the GPU memory, dictionary with unpacked data
        False - otherwise, None
        """
        logger.info(" Preloading input data.")

        gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()
        gpu_available_memory = gpu_memory_free - torch.cuda.memory_reserved() - torch.cuda.memory_allocated()

        pcl_raw = base64.b64decode(data)
        pcl_buffer = io.BytesIO(pcl_raw)
        data_dict = self._hdf5_loader.unpack_point_cloud_from_io_buffer(pcl_buffer)
        logger.info(" Done.")

        mem_check = self._check_memory_footprint(data_dict, gpu_available_memory)
        if mem_check:
            return True, data_dict
        else:
            return False, None

    @staticmethod
    def _check_memory_footprint(data_dict: dict, memory_limit: int):
        """ Function that checks whether the input data will fit in the GPU Memory

        Parameters
        ----------
        data_dict: raw input data that was received by the validator
        memory_limit: the amount of memory that is currently available in the GPU

        Returns
        -------
        True - if the input data fits in the GPU memory
        False - otherwise
        """
        # unpack data
        data_arr = [np.array(data_dict["points"]),
                    np.array(data_dict["normals"]),
                    np.array(data_dict["features_dc"]),
                    np.array(data_dict["features_rest"]),
                    np.array(data_dict["opacities"])]

        total_memory_bytes = 0
        for d in data_arr:
            total_memory_bytes += sys.getsizeof(d)

        if total_memory_bytes < memory_limit:
            logger.info(f" Total VRAM available: {memory_limit / 1024 ** 3} Gb")
            logger.info(f" Total VRAM allocated: {(torch.cuda.memory_allocated()) / 1024 ** 3} Gb")
            logger.info(f" Total data size to load to VRAM: {total_memory_bytes / 1024 ** 3} Gb")
            return True
        else:
            logger.warning(f" Total VRAM available: {memory_limit / 1024 ** 3} Gb")
            logger.warning(f" Total VRAM allocated: {(torch.cuda.memory_allocated()) / 1024 ** 3} Gb")
            logger.warning(f" Total data size to load to VRAM: {total_memory_bytes / 1024 ** 3} Gb")
            logger.warning(f" Input data size exceeds the available VRAM free memory!")
            logger.warning(f" Input data will not be further processed.\n")

            return False

    def render_gaussian_splatting_views(self, data: dict,
                                        views: int = 15,
                                        cam_rad: float = 3.5,
                                        cam_fov: float = 49.1,
                                        cam_znear: float = 0.01,
                                        cam_zfar: float = 100,
                                        gs_scale: float = 1.0,
                                        data_ver: int = 1):
        """ Function for rendering multiple views of the preloaded Gaussian Splatting model

        Parameters
        ----------
        data: dictionary with input unpacked data
        views: the amount of views that will be rendered
        cam_rad: the radius of the sphere where camera will be placed & moved along
        cam_fov: the field of view for the camera
        cam_znear: the position of the near camera plane along Z-axis
        cam_zfar: the position of the far camera plane along Z-axis
        gs_scale: the scale of the Gaussians that will be used during rendering
        data_ver: version of the input data format: 0 - corresponds to dream gaussian
                                                    1 - corresponds to new data format (default)

        Returns
        -------
        list with rendered images stored as PIL.Image
        """
        logger.info(" Rendering view of the input Gaussian Splatting Model.")

        camera = OrbitCamera(self._img_width, self._img_height, cam_fov, cam_znear, cam_zfar)

        rendered_images = []
        step = 360 // views

        for azimuth in range(0, 360, step):
            elevation = np.random.randint(self._cam_min_elev_angle, self._cam_max_elev_angle)
            camera.compute_transform_orbit(elevation, azimuth, cam_rad)

            if data_ver == 0:
                data_in = preproceses_dream_gaussian_output(data, camera.camera_position)
            elif data_ver == 1:
                data_in = data
            else:
                data_in = data
                logger.warning(f" The maximum data version for processing is < 1 >. Fall back to it. "
                               f"Rendering results might be unpredictable.")

            rendered_image, rendered_alpha, rendered_depth = self._render.render(camera,
                                                                                 data_in,
                                                                                 scale_modifier=np.clip(gs_scale, 0, 1))

            image = rendered_image.permute(1, 2, 0)
            image = image.detach().cpu().numpy() * 255
            image = image.astype(dtype=np.uint8)

            pil_image = Image.fromarray(image)
            rendered_images.append(pil_image)

        logger.info(" Done.")
        return rendered_images

    @staticmethod
    def render_video_to_images(video_file: str):
        """ Function for converting video to images

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
        """ Function for saving rendered images

        Parameters
        ----------
        images: list of images stored as PIL.Image
        file_name: the name of the file that being processed
        path: path to the folder where the rendered images will be stored

        """
        self._render.save_images(images, file_name, path)
