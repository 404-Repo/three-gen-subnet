import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir+"/validation")

import pytest
import numpy as np
from PIL import Image

from validation.gs_camera import OrbitCamera
from validation.gs_renderer import GaussianRenderer
from validation.hdf5_loader import HDF5Loader


def test_gs_renderer():
    hdf5_loader = HDF5Loader()
    data = hdf5_loader.load_point_cloud_from_h5("monkey_pcl","./tests/resources")

    camera = OrbitCamera(1080, 1080, fov_y=49.1)
    camera.compute_transform_orbit(0, 45, 3.0)

    renderer = GaussianRenderer()
    image, _, _ = renderer.render(camera, data, scale_modifier=1.0)
    image = image.permute(1, 2, 0)
    img_np = image.detach().cpu().numpy() * 255
    # img = Image.fromarray(img_np.astype(dtype=np.uint8))
    # img.save("test_render.png")

    img_ref = Image.open("./tests/resources/test_render.png")
    assert np.allclose(img_np.astype(dtype=np.uint8), np.array(img_ref))
