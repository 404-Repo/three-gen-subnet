import io
from pathlib import Path
from typing import Any

import meshio
import numpy as np
import torch
from validation_lib.io.base import BaseLoader
from validation_lib.utils import sigmoid


class PlyLoader(BaseLoader):
    def from_file(self, file_name: str, file_path: str) -> dict[str, Any]:
        """
        Wrapper for _load with file_name and file_path strings

        Parameters
        ----------
        file_name(str): the name of the file
        file_path(str): the path to file

        Returns
        -------
        a dictionary with loaded data
        """
        fpath = Path(file_path, file_name + ".ply")
        return self._load(fpath)

    def from_buffer(self, buffer: io.BytesIO) -> dict[str, Any]:
        """
        Wrapper for _load with data from bytes buffer

        Parameters
        ----------
        buffer: data packed in the bytes buffer

        Returns
        -------
        a dictionary with loaded data
        """
        return self._load(buffer)

    def _load(self, source: io.BytesIO | Path) -> dict[str, Any]:
        """
        Function for unpacking Gaussian Splatting (GS) data from bytes buffer or file path

        Parameters
        ----------
        source (io.BytesIO|pathlib.Path): data

        Returns
        -------
        a dictionary with loaded data
        """
        plydata = meshio.read(source, file_format="ply")
        points = plydata.points
        pdata = plydata.point_data

        opacities = sigmoid(np.array(pdata["opacity"]))
        rotation = np.vstack(
            [
                pdata["rot_0"],
                pdata["rot_1"],
                pdata["rot_2"],
                pdata["rot_3"],
            ]
        ).T
        rotation_to_tensor = torch.tensor(rotation).contiguous()
        normalized_rotation = torch.nn.functional.normalize(rotation_to_tensor)

        scale = np.exp(np.vstack([pdata["scale_0"], pdata["scale_1"], pdata["scale_2"]]).T)

        SH_C0 = 0.28209479177387814
        features_dc = np.array(
            [
                0.5 + SH_C0 * pdata["f_dc_0"],
                0.5 + SH_C0 * pdata["f_dc_1"],
                0.5 + SH_C0 * pdata["f_dc_2"],
            ]
        ).T

        normals = np.zeros(points.shape)
        features_rest = np.array([])
        sh_degree = 0

        data_dict = {}
        data_dict["points"] = points
        data_dict["normals"] = normals
        data_dict["features_dc"] = features_dc
        data_dict["features_rest"] = features_rest
        data_dict["opacities"] = opacities
        data_dict["scale"] = scale
        data_dict["rotation"] = normalized_rotation
        data_dict["sh_degree"] = sh_degree

        return data_dict
