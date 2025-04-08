from pathlib import Path

import torch
from engine.metrics.quality_scorer import ImageQualityMetric
from engine.rendering.renderer import Renderer
from engine.io.ply import PlyLoader


test_data_folder = Path().cwd().resolve() / "resources"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_quality_metric():
    ply_loader = PlyLoader()
    gs_data = ply_loader.from_file("hamburger", test_data_folder.as_posix())

    renderer = Renderer()
    images = renderer.render_gs(gs_data, 16, 224, 224, cam_rad=3.0, ref_bbox_size=1.0)

    quality_metric = ImageQualityMetric()
    quality_metric.load_models()

    quality_score = quality_metric.score_images_quality(images, "mean")

    assert quality_score > 0.8

    quality_metric.unload_models()
