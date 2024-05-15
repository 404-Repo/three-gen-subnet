import base64
import io
import sys

import torch
import numpy as np
import skvideo.io as video

from PIL import Image
from lib.camera_utils import orbit_camera, OrbitCamera
from lib.hdf5_loader import HDF5Loader
from lib.gaussian_splatting_renderer import GSRenderer, BasicCamera


class Renderer:
    """ Rendering pipeline for the input data """
    def __init__(self, img_width, img_height, device="cuda"):
        """

        Parameters
        ----------
        img_width  - the width of the rendering image
        img_height - the height of the rendering image
        device - string, device that will be used for rendering
        """
        self._device = torch.device(device)
        self._img_width = img_width
        self._img_height = img_height
        self._hdf5_loader = HDF5Loader()
        self._data_dict = {}
        self._renderer = None

    def render_gaussian_splatting_views(self, views: int = 10, cam_rad=1.5, cam_elev=0):
        """ Function for rendering Gaussian Splats

        Parameters
        ----------
        views   - the amount of views to render
        cam_rad - the radius of the sphere on which the rendering camera will be placed
        cam_elev - default camera elevation

        Returns
        -------
        list of rendered images
        """
        print("[INFO] Start scoring the response.")

        assert len(self._data_dict.keys()) > 0

        orbitcam = OrbitCamera(self._img_width, self._img_height, r=cam_rad, fovy=49.1)
        self._renderer.initialize(self._data_dict)

        rendered_images = []
        step = 360 // views

        # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
        min_ver = max(min(-20, -20 - cam_elev), -30 - cam_elev)
        max_ver = min(max(20, 20 - cam_elev), 30 - cam_elev)

        for azimd in range(0, 360, step):
            ver = np.random.randint(min_ver, max_ver)

            pose = orbit_camera(cam_elev + ver, azimd, cam_rad)
            camera = BasicCamera(
                pose,
                self._img_width,
                self._img_height,
                orbitcam.fovy,
                orbitcam.fovx,
                orbitcam.near,
                orbitcam.far,
            )

            output_dict = self._renderer.render(camera)
            img = output_dict["image"].permute(1, 2, 0)
            img = img.detach().cpu().numpy() * 255
            img = np.concatenate((img, 255 * np.ones((img.shape[0], img.shape[1], 1))), axis=2).astype(np.uint8)
            img = Image.fromarray(img)
            rendered_images.append(img)
        return rendered_images

    @staticmethod
    def render_video_to_images(video_file: str):
        """ Function for converting video to the list of images

        Parameters
        ----------
        video_file - the video file that will be converted to the list of images

        Returns
        -------
        a list with images
        """
        video_data = video.vread(video_file)
        images = [Image.fromarray(video_data[i, :, :, :]) for i in range(video_data.shape[0])]
        return images

    def init_gaussian_splatting_renderer(self, data: str, sh_degree: int = 3, white_background: bool = True, radius: float = 1.0):
        """ Function for initializing the Gaussian Splatting rendering

        Parameters
        ----------
        data             - input data stored as bytes
        sh_degree        - degree of spherical harmonics
        white_background - boolean variable that defines whether to use or not white background
        radius           - the radius of the initial sphere

        Returns
        -------
        True if the data fits the VRAM and initialisation was successful
        False otherwise
        """

        self._renderer = GSRenderer(sh_degree, white_background, radius)
        gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()
        gpu_available_memory = gpu_memory_free - torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        self._unpacking_data(data)
        return self._check_memory_footprint(self._data_dict, gpu_available_memory)

    def _unpacking_data(self, data: str):
        """ Function that unpacks input data from bytes to dictionary

        Parameters
        ----------
        data - packed data, bytes

        Returns
        -------
        unpacked data in the dictionary format
        """

        pcl_raw = base64.b64decode(data)
        pcl_buffer = io.BytesIO(pcl_raw)
        self._data_dict = self._hdf5_loader.unpack_point_cloud_from_io_buffer(pcl_buffer)

    @staticmethod
    def _check_memory_footprint(data_dict: dict, memory_limit: int):
        """ Function that checks whether the input data will fit in the GPU VRAM

        Parameters
        ----------
        pcl_raw - raw input data that was received by the validator
        memory_limit - the amount of memory that is currently available in the GPU

        Returns True if the size of the input data can be fit in the VRAM, otherwise return False
        -------

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
            # print("\n[INFO] Total VRAM available: ", memory_limit / 1024**3, " Gb")
            # print("[INFO] Total VRAM allocated: ", (torch.cuda.memory_allocated()+torch.cuda.memory_reserved()) / 1024**3, " Gb")
            # print("[INFO] Total data size to load to VRAM: ", total_memory_bytes/1024**3, " Gb\n")
            return True
        else:
            print("\n[INFO] Total VRAM: ", memory_limit / 1024**3, " Gb")
            print("[INFO] Total VRAM allocated: ", (torch.cuda.memory_allocated()+torch.cuda.memory_reserved()) / 1024**3, " Gb")
            print("[INFO] Total data size to load to VRAM: ", total_memory_bytes / 1024**3, " Gb")
            print("[INFO] Input data size exceeds the available VRAM free memory!")
            print("[INFO] Input data will not be further processed.\n")
            return False
