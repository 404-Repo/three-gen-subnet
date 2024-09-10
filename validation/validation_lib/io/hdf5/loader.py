import io
from pathlib import Path
from typing import Any

import h5py as h5
import numpy as np
from validation_lib.io.base import BaseLoader


class HDF5Loader(BaseLoader):
    """Class for storing Gaussian Splatting data in HDF5 file format."""

    @staticmethod
    def _get_dataset(group: h5.Group, dataset_name: str) -> np.ndarray:
        """Function for getting data from the given group using dataset name

        Parameters
        ----------
        group: HDF5 group object where the dataset is stored
        dataset_name: the name of the dataset that will be accessed and returned

        Returns
        -------
        a numpy array with requested dataset
        """

        data = group.get(dataset_name)
        return np.array(data)

    def from_file(self, file_name: str, file_path: str) -> dict[str, Any]:
        """Function for loading data from the the HDF5 file

        Parameters
        ----------
        file_name: the name of the file
        file_path: the path to file

        Returns
        -------
        a dictionary with loaded data
        """
        h5_fpath = Path(file_path) / f"{file_name}.h5"
        file = h5.File(h5_fpath, mode="r")

        points = self._get_dataset(file, "points")
        normals = self._get_dataset(file, "normals")
        features_dc = self._get_dataset(file, "features_dc")
        features_rest = self._get_dataset(file, "features_rest")
        opacities = self._get_dataset(file, "opacities")
        scale = self._get_dataset(file, "scale")
        rotation = self._get_dataset(file, "rotation")
        sh_degree = int(self._get_dataset(file, "sh_degree")[0])
        file.close()

        data_dict = {
            "points": points,
            "normals": normals,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacities": opacities,
            "scale": scale,
            "rotation": rotation,
            "sh_degree": sh_degree,
        }

        return data_dict

    def from_buffer(self, buffer: io.BytesIO) -> dict[str, Any]:
        """Function for unpacking Gaussian Splatting (GS) data from bytes buffer

        Parameters
        ----------
        buffer: data packed in the bytes buffer

        Returns
        -------
        a dictionary with loaded data
        """
        with h5.File(buffer, "r", driver="fileobj") as file:
            points = self._get_dataset(file, "points")
            normals = self._get_dataset(file, "normals")
            features_dc = self._get_dataset(file, "features_dc")
            features_rest = self._get_dataset(file, "features_rest")
            opacities = self._get_dataset(file, "opacities")
            scale = self._get_dataset(file, "scale")
            rotation = self._get_dataset(file, "rotation")
            sh_degree = int(self._get_dataset(file, "sh_degree")[0])

        data_dict = {
            "points": points,
            "normals": normals,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacities": opacities,
            "scale": scale,
            "rotation": rotation,
            "sh_degree": sh_degree,
        }

        return data_dict
