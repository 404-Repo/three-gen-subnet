import io
from pathlib import Path

import meshio
import numpy as np
import torch

from engine.data_structures import GaussianSplattingData
from engine.utils.gs_data_checker_utils import sigmoid


class PlyLoader:
    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sh_c0 = 0.28209479177387814
        self._mean_coeff_dc_features = 0.5

    def from_file(self, file_name: str, file_path: str) -> GaussianSplattingData:
        """Wrapper for _load with file_name and file_path strings"""

        fpath = Path(file_path, file_name + ".ply")
        return self._load(fpath)

    def from_buffer(self, buffer: io.BytesIO) -> GaussianSplattingData:
        """Wrapper for _load with data from bytes buffer"""

        return self._load(buffer)

    def _load(self, source: io.BytesIO | Path) -> GaussianSplattingData:
        """Function for unpacking Gaussian Splatting (GS) data from bytes buffer or file path"""

        plydata = meshio.read(source, file_format="ply")
        pdata = plydata.point_data

        points = torch.tensor(plydata.points, dtype=torch.float32)

        opacities = sigmoid(torch.tensor(pdata["opacity"], dtype=torch.float32))
        rotation = np.vstack(
            [
                pdata["rot_0"],
                pdata["rot_1"],
                pdata["rot_2"],
                pdata["rot_3"],
            ]
        ).T
        rotation_to_tensor = torch.tensor(rotation, dtype=torch.float32).contiguous()
        normalized_rotations = torch.nn.functional.normalize(rotation_to_tensor)

        scales = np.exp(np.vstack([pdata["scale_0"], pdata["scale_1"], pdata["scale_2"]]).T)
        scales = torch.tensor(scales, dtype=torch.float32)

        features_dc_arr = np.array(
            [
                self._mean_coeff_dc_features + self._sh_c0 * pdata["f_dc_0"],
                self._mean_coeff_dc_features + self._sh_c0 * pdata["f_dc_1"],
                self._mean_coeff_dc_features + self._sh_c0 * pdata["f_dc_2"],
            ]
        ).T
        features_dc = torch.tensor(features_dc_arr, dtype=torch.float32)

        normals = torch.zeros_like(points, dtype=torch.float32)
        features_rest = torch.tensor([], dtype=torch.float32)
        sh_degree = torch.tensor(0)

        return GaussianSplattingData(
            points=points,
            normals=normals,
            features_dc=features_dc,
            features_rest=features_rest,
            opacities=opacities,
            scales=scales,
            rotations=normalized_rotations,
            sh_degree=sh_degree,
        )
