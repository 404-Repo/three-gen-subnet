import inspect
import os
import sys


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/validation")

import pytest
from validation_lib.io.hdf5 import HDF5Loader
from validation_lib.rendering.rendering_pipeline import RenderingPipeline
from validation_lib.validation.validation_pipeline import ValidationPipeline


@pytest.fixture
def h5data():
    test_data_folder = os.path.join(currentdir, "resources")

    loader = HDF5Loader()
    data_dict = loader.from_file("monkey_pcl", test_data_folder)

    return data_dict


def test_validator(h5data):
    prompt = "A yellow monkey"
    data = h5data
    render = RenderingPipeline(16, "gs")
    images = render.render_gaussian_splatting_views(data, 512, 512, 3.0, data_ver=2)

    validator = ValidationPipeline()
    validator.preload_model()
    score = validator.validate(images, prompt)

    assert score > 0.9
