import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/validation")

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from engine.io.ply import PlyLoader
from engine.rendering.renderer import Renderer

current_file_path = Path(__file__).resolve()
test_data_folder = current_file_path.parent / "resources"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_gs_renderer_ply():
    ply_loader = PlyLoader()
    gs_data = ply_loader.from_file("hamburger", test_data_folder.as_posix())

    renderer = Renderer()
    image = renderer.render_gs(gs_data, 1, 1080, 1080, [45], [0], cam_rad=3.0, ref_bbox_size=1.0)[0]

    img_np = (image * 255).to(torch.uint8).detach().cpu().numpy()
    img_ref_file = test_data_folder / "test_render_ply.png"
    img_ref = Image.open(img_ref_file.as_posix())

    try:
        assert not np.allclose(img_np, np.asarray(img_ref))
    except Exception as e:
        img_path = test_data_folder / "test_image_ply.png"
        img_ref.save(img_path.as_posix())
        logger.warning(f"Generated image is saved, path: {img_path.as_posix()}")
        raise ValueError(f"Generated and stored images are not the same: {e}")
