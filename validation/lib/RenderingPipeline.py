import base64
import io
import torch
import numpy as np
import skvideo.io as video

from PIL import Image
from lib.camera_utils import orbit_camera, OrbitCamera
from lib.hdf5_loader import HDF5Loader
from lib.gaussian_splatting_renderer import GSRenderer, BasicCamera


class Renderer:
    def __init__(self, img_width, img_height, device="cuda"):
        self.__device = torch.device(device)
        self.__img_width = img_width
        self.__img_height = img_height
        self.__hdf5_loader = HDF5Loader()
        self.__renderer = None

    def render_mesh_views(self):
        pass

    def render_gaussian_splatting_views(self, data: str, views: int = 10, cam_rad=1.5, cam_elev=0, save_images: bool = False):
        print("[INFO] Start scoring the response.")

        orbitcam = OrbitCamera(self.__img_width, self.__img_height, r=cam_rad, fovy=49.1)

        pcl_raw = base64.b64decode(data)
        pcl_buffer = io.BytesIO(pcl_raw)
        data_dict = self.__hdf5_loader.unpack_point_cloud_from_io_buffer(pcl_buffer)

        self.__renderer.initialize(data_dict)

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
                self.__img_width,
                self.__img_height,
                orbitcam.fovy,
                orbitcam.fovx,
                orbitcam.near,
                orbitcam.far,
            )

            output_dict = self.__renderer.render(camera)
            img = output_dict["image"].permute(1, 2, 0)
            img = img.detach().cpu().numpy() * 255
            img = np.concatenate((img, 255 * np.ones((img.shape[0], img.shape[1], 1))), axis=2).astype(np.uint8)
            img = Image.fromarray(img)
            rendered_images.append(img)
        return rendered_images

    def render_nerf_views(self):
        pass

    @staticmethod
    def render_video_to_images(video_file: str):
        video_data = video.vread(video_file)
        images = [Image.fromarray(video_data[i, :, :, :]) for i in range(video_data.shape[0])]
        return images

    def init_gaussian_splatting_renderer(self,  sh_degree: int = 3, white_background: bool = True, radius: float = 1.0):
        self.__renderer = GSRenderer(sh_degree, white_background, radius)
