import io
import os
import sys
import inspect
import base64

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir+"/validation")

import pytest

from validation.hdf5_loader import HDF5Loader
from validation.rendering_pipeline import RenderingPipeline
from validation.validation_pipeline import ValidationPipeline


@pytest.fixture
def h5_to_base64():
    hdf5_loader = HDF5Loader()
    data_dict = hdf5_loader.load_point_cloud_from_h5("monkey_pcl", "./tests/resources")
    buffer = hdf5_loader.pack_point_cloud_to_io_buffer(**data_dict)
    io_buffer = io.BytesIO(buffer.getbuffer())
    encoded_buffer = base64.b64encode(io_buffer.getbuffer())

    return encoded_buffer


def test_validator(h5_to_base64):
    prompt = "A yellow monkey"
    data = h5_to_base64

    render = RenderingPipeline(512, 512, "gs")
    data_ready, data_out = render.prepare_data(data)

    assert data_ready

    images = render.render_gaussian_splatting_views(data_out, 15, 3.0)
    validator = ValidationPipeline()
    validator.preload_scoring_model()
    score = validator.validate(images, prompt)

    assert score > 0.9

