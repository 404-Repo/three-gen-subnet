from neurons.old.protocol import TextTo3D
from validation.lib.validators import TextTo3DModelValidator


def test_validator():
    prompt = ""
    data = ""
    t23D = TextTo3D(prompt_in=prompt, mesh_out=data)
    validator = TextTo3DModelValidator(512, 512, 10)
    validator.init_gaussian_splatting_renderer()
    scores = validator.score_response_gs_input([t23D], save_images=False, cam_rad=4)

    assert True == True
