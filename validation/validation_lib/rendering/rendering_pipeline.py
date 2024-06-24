import torch
import numpy as np
import skvideo.io as video
from PIL import Image
from loguru import logger

from validation_lib.rendering.gs_renderer import GaussianRenderer
from validation_lib.rendering.gs_camera import OrbitCamera
from validation_lib.utils import preprocess_dream_gaussian_output


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
        views: int = 16,
        cam_rad: float = 3.5,
        cam_fov: float = 49.1,
        cam_znear: float = 0.01,
        cam_zfar: float = 100,
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
        data_ver: version of the input data format: 0 - corresponds to dream gaussian
                                                    1 - corresponds to new data format (default)

        Returns
        -------
        list with rendered images stored as PIL.Image
        """

        logger.info(" Rendering view of the input Gaussian Splatting Model.")

        # setting up the camera
        camera = OrbitCamera(self._img_width, self._img_height, cam_fov, cam_znear, cam_zfar)

        # setting up tensors for storing camera transforms
        camera_views_proj = torch.empty((views, 4, 4)).to(self._device)
        camera_intrs = torch.empty((views, 3, 3)).to(self._device)

        # setting up angles that will be rendered
        random_angles_number = views // 2
        step = 360 // random_angles_number
        elevations = np.random.randint(self._cam_min_elev_angle, self._cam_max_elev_angle, random_angles_number)

        j = 0
        for i in range(2):
            for azimuth, elev in zip(range(0, 360, step), elevations):
                if i == 0:
                    camera.compute_transform_orbit(0, azimuth, cam_rad)
                else:
                    camera.compute_transform_orbit(elev, azimuth, cam_rad)

                camera_views_proj[j] = camera.world_to_camera_transform
                camera_intrs[j] = camera.intrinsics
                j += 1

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
        rendered_images = [(img * 255).to(torch.uint8) for img in rendered_images]

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

    def save_rendered_images(self, images: torch.Tensor, file_name: str, path: str):
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
