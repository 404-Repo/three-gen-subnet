from io import BytesIO
from pathlib import Path

import h5py as h5
import numpy as np
from validation_lib.io.base import BaseWriter


class HDF5Writer(BaseWriter):
    @staticmethod
    def _create_dataset(group: h5.Group, dataset_name: str, data: np.ndarray) -> h5.Group:
        """Function for creating a dataset

        Parameters
        ----------
        group: HDF5 group object where the dataset will create the dataset
        dataset_name: the name of the dataset that will be created
        data: the data that will be stored in the created dataset

        Returns
        -------
        group object with created & stored dataset
        """
        group.create_dataset(dataset_name, data.shape, data.dtype, data, compression="gzip", compression_opts=9)
        return group

    def to_buffer(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        features_dc: np.ndarray,
        features_rest: np.ndarray,
        opacities: np.ndarray,
        scale: np.ndarray,
        rotation: np.ndarray,
        sh_degree: int,
    ) -> BytesIO:
        """Function for packing Gaussian Splatting (GS) data to a bytes buffer in a compressed way

        Parameters
        ----------
        points:  numpy array with points (point cloud data)
        normals: numpy array with normals data (point cloud related) (optional)
        features_dc: numpy array with diffusion colour features (usually colours)
        features_rest: numpy array with other colour related features (optional)
        opacities: numpy array with computed opacities
        scale: numpy array with computed scales
        rotation: numpy array with computed GS rotations
        sh_degree: GS degree (integer, optional, default is 0)

        Returns
        -------
        buffer stored as bytes
        """
        buffer = BytesIO()
        with h5.File(buffer, "w", driver="fileobj") as file:
            self._create_dataset(file, "points", points)
            self._create_dataset(file, "normals", normals)
            self._create_dataset(file, "features_dc", features_dc)
            self._create_dataset(file, "features_rest", features_rest)
            self._create_dataset(file, "opacities", opacities)
            self._create_dataset(file, "scale", scale)
            self._create_dataset(file, "rotation", rotation)
            self._create_dataset(file, "sh_degree", np.array([sh_degree]))
        return buffer

    def to_file(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        features_dc: np.ndarray,
        features_rest: np.ndarray,
        opacities: np.ndarray,
        scale: np.ndarray,
        rotation: np.ndarray,
        sh_degree: int,
        h5file_name: str,
        h5file_path: str,
    ) -> None:
        """Function for saving the Gaussian Splatting (GS) data to the HDF5 file in a compressed way

        Parameters
        ----------
        points:  numpy array with points (point cloud data)
        normals: numpy array with normals data (point cloud related) (optional)
        features_dc: numpy array with diffusion colour features (usually colours)
        features_rest: numpy array with other colour related features (optional)
        opacities: numpy array with computed opacities
        scale: numpy array with computed scales
        rotation: numpy array with computed GS rotations
        sh_degree: GS degree (integer, optional, default is 0)
        h5file_name: the name of the HDF file
        h5file_path: the path where to store the HDF file

        """

        h5_fpath = Path(h5file_path) / f"{h5file_name}.h5"
        file = h5.File(h5_fpath, mode="w")

        self._create_dataset(file, "points", points)
        self._create_dataset(file, "normals", normals)
        self._create_dataset(file, "features_dc", features_dc)
        self._create_dataset(file, "features_rest", features_rest)
        self._create_dataset(file, "opacities", opacities)
        self._create_dataset(file, "scale", scale)
        self._create_dataset(file, "rotation", rotation)
        self._create_dataset(file, "sh_degree", np.array([sh_degree]))

        file.close()
