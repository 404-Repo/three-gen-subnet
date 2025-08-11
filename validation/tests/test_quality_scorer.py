import sys
import os
from pathlib import Path
import inspect

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, PARENT_DIR + "/validation")

import torch
from engine.metrics.quality_scorer import ImageQualityMetric
from engine.rendering.renderer import Renderer
from engine.io.ply import PlyLoader

current_file_path = Path(__file__).resolve()
test_data_folder = current_file_path.parent / "resources"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_quality_metric():
    ply_loader = PlyLoader()
    gs_data = ply_loader.from_file("hamburger", test_data_folder.as_posix())
    gs_data = gs_data.send_to_device(device)

    renderer = Renderer()
    images = renderer.render_gs(gs_data, 16, 518, 518)  # , cam_rad=3.0, ref_bbox_size=1.0)

    quality_metric = ImageQualityMetric()
    quality_metric.load_models()

    quality_score = quality_metric.score_images_quality(images, "mean")

    assert quality_score > 0.832

    quality_metric.unload_models()

    print(f"Quality score: {quality_score}")
