import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/validation")

from pathlib import Path

import pytest
from engine.io.ply import PlyLoader
from engine.rendering.renderer import Renderer
from engine.validation_engine import ValidationEngine
from engine.data_structures import ValidationResponse, GaussianSplattingData


@pytest.fixture
def ply_data() -> GaussianSplattingData:
    current_file_path = Path(__file__).resolve()
    test_data_folder = current_file_path.parent / "resources"
    loader = PlyLoader()
    gs_data = loader.from_file("hamburger", test_data_folder.as_posix())
    return gs_data


def test_validator(ply_data):
    prompt = "A hamburger"
    gs_data = ply_data

    render = Renderer()
    images = render.render_gs(gs_data, 16, 224, 224, cam_rad=3.0, ref_bbox_size=1.0)

    validator = ValidationEngine()
    validator.load_pipelines()

    score: ValidationResponse = validator.validate_text_to_gs(prompt, images)

    assert score.final_score > 0.8
