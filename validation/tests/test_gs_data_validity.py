import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/validation")

import torch
from engine.utils.gs_data_checker_utils import is_input_data_valid
from engine.data_structures import GaussianSplattingData


def test_input_data_validity():
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
                gs_data = GaussianSplattingData(
                    points=points,
                    normals=torch.zeros_like(points),
                    opacities=opacities,
                    scales=scales,
                    features_dc=torch.zeros_like(points),
                    features_rest=torch.zeros_like(points),
                    rotations=torch.zeros(size=(points.shape[0], 4)),
                    sh_degree=torch.tensor(1),
                )
                gs_data_gpu = gs_data.send_to_device(torch.device("cuda"))
                responses.append(is_input_data_valid(gs_data_gpu))

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
