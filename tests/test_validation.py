import base64

import h5py
import pytest

from neurons.old.protocol import TextTo3D
from validation.lib.validators import TextTo3DModelValidator


@pytest.fixture
def h5_to_base64():
    file_path = "resources/tiger_pcl.h5"
    with open(file_path, "rb") as file:
        content = file.read()
        encoded_content = base64.b64encode(content)
        encoded_content_str = encoded_content.decode("utf-8")
        return encoded_content_str


def test_validator(h5_to_base64):
    prompt = "A tiger"
    data = h5_to_base64
    t23D = TextTo3D(prompt_in=prompt, mesh_out=data)
    validator = TextTo3DModelValidator(512, 512, 10)
    validator.init_gaussian_splatting_renderer()
    scores = validator.score_response_gs_input([t23D], save_images=False, cam_rad=4)

    assert scores[0] > 0.9
