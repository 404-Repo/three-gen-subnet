import io
from pathlib import Path
from typing import Dict

import numpy as np
from plyfile import PlyData
import torch

from validation.io.base import BaseLoader
from validation.utils import sigmoid


class PlyLoader(BaseLoader):
    def from_file(self, file_name: str, file_path: str) -> Dict:
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

    def from_buffer(self, buffer: io.BytesIO) -> Dict:
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

    def _load(self, source: io.BytesIO | Path) -> Dict:
        """
        Function for unpacking Gaussian Splatting (GS) data from bytes buffer or file path

        Parameters
        ----------
        source (io.BytesIO|pathlib.Path): data

        Returns
        -------
        a dictionary with loaded data
        """
        plydata = PlyData.read(source)

        points = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
        opacities = sigmoid(np.array(plydata["vertex"]["opacity"]))

        SH_C0 = 0.28209479177387814
        features_dc = np.array(
            [
                0.5 + SH_C0 * plydata["vertex"]["f_dc_0"],
                0.5 + SH_C0 * plydata["vertex"]["f_dc_1"],
                0.5 + SH_C0 * plydata["vertex"]["f_dc_2"],
            ]
        ).T

        scale = np.exp(
            np.vstack([plydata["vertex"]["scale_0"], plydata["vertex"]["scale_1"], plydata["vertex"]["scale_2"]]).T
        )

        rotation = np.vstack(
            [
                plydata["vertex"]["rot_0"],
                plydata["vertex"]["rot_1"],
                plydata["vertex"]["rot_2"],
                plydata["vertex"]["rot_3"],
            ]
        ).T
        rotation = torch.tensor(rotation).contiguous()
        rotation = torch.nn.functional.normalize(rotation)

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
        data_dict["rotation"] = rotation
        data_dict["sh_degree"] = sh_degree

        return data_dict
