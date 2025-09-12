import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/validation")

from pathlib import Path

import torch
from engine.io.ply import PlyLoader
from engine.rendering.renderer import Renderer
from engine.validation_engine import ValidationEngine


current_file_path = Path(__file__).resolve()
test_data_folder = current_file_path.parent / "resources"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_validation_engine():
    ply_loader = PlyLoader()
    gs_data = ply_loader.from_file("hamburger", test_data_folder.as_posix())
    gs_data = gs_data.send_to_device(device)
    prompt = "a hamburger"

    renderer = Renderer()
    images = renderer.render_gs(gs_data, 16, 518, 518, cam_rad=3.0, ref_bbox_size=1.0)

    validator = ValidationEngine()
    validator.load_pipelines()

    validation_results = validator.validate_text_to_gs(prompt, images)

    assert validation_results.score > 0.8
    assert validation_results.lpips > 0.9
    assert validation_results.ssim > 0.9
    assert validation_results.iqa > 0.8
    assert validation_results.alignment > 0.75
    print(validation_results)
