import base64

import pytest

from lib.validators import TextTo3DModelValidator


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
    validator = TextTo3DModelValidator(512, 512, 10)
    validator.init_gaussian_splatting_renderer()
    score = validator.score_response_gs_input(
        prompt, data, save_images=False, cam_rad=4
    )

    assert score > 0.9
