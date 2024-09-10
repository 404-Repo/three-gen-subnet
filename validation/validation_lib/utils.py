from typing import Any

import numpy as np
import torch

from .eval_sh import eval_sh


def preprocess_dream_gaussian_output(data: dict, camera_position: torch.Tensor) -> dict[str, Any]:
    """Function for converting dream gaussian project output to the format that new
    rendering pipeline can process

    Parameters
    ----------
    data: dictionary with input gaussian splatting data
    camera_position: current position of the camera

    Returns
    -------
    dictionary with processed input data
    """

    out_dict = dict()
    out_dict["points"] = data["points"]
    means3D = torch.tensor(data["points"], dtype=torch.float32).contiguous()

    rotations = torch.tensor(data["rotation"], dtype=torch.float32).contiguous()
    rotations = torch.nn.functional.normalize(rotations)
    out_dict["rotation"] = rotations.detach().cpu().numpy()

    scales = torch.tensor(data["scale"], dtype=torch.float32).contiguous()
    scales = torch.exp(scales)
    out_dict["scale"] = scales.detach().cpu().numpy()

    opacity = torch.tensor(data["opacities"], dtype=torch.float32).contiguous()
    opacity = torch.sigmoid(opacity)
    out_dict["opacities"] = opacity.detach().cpu().numpy()

    f_dc = torch.tensor(data["features_dc"], dtype=torch.float32).contiguous()
    f_rest = torch.tensor(data["features_rest"], dtype=torch.float32).contiguous()
    f_combined = torch.cat((f_dc, f_rest), dim=1)

    shs_view = f_combined.transpose(1, 2).view(-1, 3, (0 + 1) ** 2)
    dir_pp = means3D - camera_position.repeat(f_combined.shape[0], 1)
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(0, shs_view, dir_pp_normalized)
    rgbs = torch.clamp_min(sh2rgb + 0.5, 0.0)

    out_dict["features_dc"] = rgbs.detach().cpu().numpy()
    out_dict["features_rest"] = np.array([])
    out_dict["normals"] = np.array([])
    out_dict["sh_degree"] = data["sh_degree"]

    return out_dict


def sigmoid(x: np.ndarray) -> Any:
    """
    Apply the sigmoid function element-wise to the input array.

    Args:
    x (np.ndarray): Input array.

    Returns:
    np.ndarray: The output array with the sigmoid function applied element-wise.
    """
    return 1 / (1 + np.exp(-x))
