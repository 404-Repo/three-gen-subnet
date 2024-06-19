import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/validation")

import pytest
import numpy as np
from PIL import Image
from loguru import logger

from validation.rendering.gs_camera import OrbitCamera
from validation.rendering.gs_renderer import GaussianRenderer
from validation.io.hdf5 import HDF5Loader
from validation.io.ply import PlyLoader


test_data_folder = os.path.join(currentdir, "resources")


def test_gs_renderer_hdf5():
    hdf5_loader = HDF5Loader()
    data = hdf5_loader.from_file("monkey_pcl", test_data_folder)

    camera = OrbitCamera(1080, 1080, fov_y=49.1)
    camera.compute_transform_orbit(0, 45, 3.0)

    renderer = GaussianRenderer()
    image, _, _ = renderer.render(camera, data, scale_modifier=1.0)
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

    renderer = GaussianRenderer()
    image, _, _ = renderer.render(camera, data, scale_modifier=1.0)
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