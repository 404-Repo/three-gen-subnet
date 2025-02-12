import torch
import numpy as np
from validation_lib.validation.validation_pipeline import is_input_data_valid


def test_input_data_validity():
    data_dict = {}
    indices1 = torch.randperm(11000)[:5000]
    indices2 = torch.randperm(11000)[:9000]

    # setting up centroids
    valid_points = torch.rand((11000, 3))
    invalid_points1 = torch.zeros_like(valid_points)

    invalid_points2 = valid_points.clone()
    invalid_points2[indices1, :] = 0.0001 * valid_points[:5000, :]

    invalid_points3 = valid_points.clone()
    invalid_points3[indices2, :] = torch.zeros_like(valid_points[:9000, :])

    input_points = [valid_points, invalid_points1, invalid_points2, invalid_points3]

    # setting up scales
    valid_scales = torch.rand((11000, 3))
    invalid_scales1 = torch.zeros_like(valid_scales)

    invalid_scales2 = valid_scales.clone()
    invalid_scales2[indices1, :] = 0.0001 * valid_scales[:5000, :]

    invalid_scales3 = valid_scales.clone()
    invalid_scales3[indices2, :] = torch.zeros_like(valid_scales[:9000, :])

    input_scales = [valid_scales, invalid_scales1, invalid_scales2, invalid_scales3]

    # setting up opacities
    valid_opacities = torch.rand((11000, 1))
    invalid_opacities1 = torch.zeros_like(valid_opacities)

    invalid_opacities2 = valid_opacities.clone()
    invalid_opacities2[indices1, :] = 0.0001 * valid_opacities[:5000, :]

    invalid_opacities3 = valid_opacities.clone()
    invalid_opacities3[indices2, :] = torch.zeros_like(valid_opacities[:9000, :])

    input_opacities = [valid_opacities, invalid_opacities1, invalid_opacities2, invalid_opacities3]

    responses = []
    for points in input_points:
        for scales in input_scales:
            for opacities in input_opacities:
                data_dict["points"] = points
                data_dict["opacities"] = opacities
                data_dict["scale"] = scales
                responses.append(is_input_data_valid(data_dict))

    responses_gr_t = [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]

    for response, response_gr_t in zip(responses, responses_gr_t):
        assert response == response_gr_t
