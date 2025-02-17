import inspect
import os
import sys


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/validation")

import pytest
from validation_lib.io.ply import PlyLoader
from validation_lib.rendering.rendering_pipeline import RenderingPipeline
from validation_lib.validation.validation_pipeline import ValidationPipeline, ValidationResult


@pytest.fixture
def ply_data():
    test_data_folder = os.path.join(currentdir, "resources")

    loader = PlyLoader()
    data_dict = loader.from_file("hamburger", test_data_folder)

    return data_dict


def test_validator(ply_data):
    prompt = "A hamburger"
    data = ply_data
    render = RenderingPipeline(16, "gs")
    images = render.render_gaussian_splatting_views(data, 224, 224, 2.5)

    validator = ValidationPipeline()
    validator.preload_model()
    score: ValidationResult = validator.validate(images, prompt)

    assert score.final_score >= 0.7422
