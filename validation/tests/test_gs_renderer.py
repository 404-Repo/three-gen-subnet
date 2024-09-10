import inspect
import os
import sys


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/validation")

import numpy as np
import torch
from loguru import logger
from PIL import Image
from validation_lib.io.hdf5 import HDF5Loader
from validation_lib.io.ply import PlyLoader
from validation_lib.rendering.gs_camera import OrbitCamera
from validation_lib.rendering.gs_renderer import GaussianRenderer


test_data_folder = os.path.join(currentdir, "resources")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_gs_renderer_hdf5():
    hdf5_loader = HDF5Loader()
    data = hdf5_loader.from_file("monkey_pcl", test_data_folder)

    camera = OrbitCamera(1080, 1080, fov_y=49.1)
    camera.compute_transform_orbit(0, 45, 3.0)

    camera_views_proj = torch.unsqueeze(camera.world_to_camera_transform, dim=0)
    camera_intrs = torch.unsqueeze(camera.intrinsics, dim=0)

    # converting input data to tensors on GPU
    means3D = torch.tensor(data["points"], dtype=torch.float32).contiguous().squeeze().to(device)
    rotations = torch.tensor(data["rotation"], dtype=torch.float32).contiguous().squeeze().to(device)
    scales = torch.tensor(data["scale"], dtype=torch.float32).contiguous().squeeze().to(device)
    opacity = torch.tensor(data["opacities"], dtype=torch.float32).contiguous().squeeze().to(device)
    rgbs = torch.tensor(data["features_dc"], dtype=torch.float32).contiguous().squeeze().to(device)

    # preparing data to send for rendering
    gaussian_data = [means3D, rotations, scales, opacity, rgbs]

    renderer = GaussianRenderer()
    image, _, _ = renderer.render(
        camera_views_proj,
        camera_intrs,
        (camera.image_width, camera.image_height),
        camera.z_near,
        camera.z_far,
        gaussian_data,
    )

    img_np = image.detach().cpu().numpy() * 255
    # img = Image.fromarray(img_np.astype(dtype=np.uint8))
    # img.save("test_render.png")

    img_ref_file = os.path.join(test_data_folder, "test_render_hdf5.png")
    img_ref = Image.open(img_ref_file)
    try:
        assert np.allclose(img_np.astype(dtype=np.uint8), np.array(img_ref))
    except Exception as e:
        img_path = os.path.join(test_data_folder, "test_image_hdf5.png")
        img_ref.save(img_path)
        logger.warning(f"Generated image is saved, path: {img_path}")
        raise ValueError(f"Generated and stored images are not the same: {e}")


def test_gs_renderer_ply():
    ply_loader = PlyLoader()
    data = ply_loader.from_file("hamburger", test_data_folder)

    camera = OrbitCamera(1080, 1080, fov_y=49.1)
    camera.compute_transform_orbit(0, 45, 3.0)

    camera_views_proj = torch.unsqueeze(camera.world_to_camera_transform, dim=0)
    camera_intrs = torch.unsqueeze(camera.intrinsics, dim=0)

    # converting input data to tensors on GPU
    means3D = torch.tensor(data["points"], dtype=torch.float32).contiguous().squeeze().to(device)
    rotations = torch.tensor(data["rotation"], dtype=torch.float32).contiguous().squeeze().to(device)
    scales = torch.tensor(data["scale"], dtype=torch.float32).contiguous().squeeze().to(device)
    opacity = torch.tensor(data["opacities"], dtype=torch.float32).contiguous().squeeze().to(device)
    rgbs = torch.tensor(data["features_dc"], dtype=torch.float32).contiguous().squeeze().to(device)

    # preparing data to send for rendering
    gaussian_data = [means3D, rotations, scales, opacity, rgbs]

    renderer = GaussianRenderer()
    image, _, _ = renderer.render(
        camera_views_proj,
        camera_intrs,
        (camera.image_width, camera.image_height),
        camera.z_near,
        camera.z_far,
        gaussian_data,
    )

    img_np = image.detach().cpu().numpy() * 255
    # img = Image.fromarray(img_np.astype(dtype=np.uint8))
    # img.save("test_render_ply.png")

    img_ref_file = os.path.join(test_data_folder, "test_render_ply.png")
    img_ref = Image.open(img_ref_file)

    try:
        assert np.allclose(img_np.astype(dtype=np.uint8), np.array(img_ref))
    except Exception as e:
        img_path = os.path.join(test_data_folder, "test_image_ply.png")
        img_ref.save(img_path)
        logger.warning(f"Generated image is saved, path: {img_path}")
        raise ValueError(f"Generated and stored images are not the same: {e}")
