import os
import sys
import inspect

import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/validation")

import pytest

from validation.rendering.rendering_pipeline import RenderingPipeline
from validation.io.hdf5 import HDF5Loader


@pytest.fixture
def h5data():
    test_data_folder = os.path.join(currentdir, "resources")

    loader = HDF5Loader()
    data_dict = loader.from_file("monkey_pcl", test_data_folder)

    return data_dict


def test_rendering_pipeline(h5data):
    data = h5data
    render = RenderingPipeline(512, 512, "gs")
    images = render.render_gaussian_splatting_views(data, 16, 3.0, data_ver=2)

    blank_image = np.ones(np.array(images[0]).shape, dtype=np.uint8) * 255
    for img in images:
        assert np.any(np.array(img) != blank_image)

    # uncomment if you want to see rendered images
    # render.save_rendered_images(images, "img", "renders")
