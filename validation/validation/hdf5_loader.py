import io
import os
import numpy as np
import h5py as h5


class HDF5Loader:
    """ Class for storing Gaussian Splatting data in HDF5 file format. """
    def __init__(self):
        pass

    @staticmethod
    def _create_dataset(group: h5.Group, dataset_name: str, data: np.ndarray):
        """ Function for creating a dataset

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

    @staticmethod
    def _get_dataset(group: h5.Group, dataset_name: str):
        """ Function for getting data from the given group using dataset name

        Parameters
        ----------
        group: HDF5 group object where the dataset is stored
        dataset_name: the name of the dataset that will be accessed and returned

        Returns
        -------
        a numpy array with requested dataset
        """

        data = group.get(dataset_name)
        return np.array(data, dtype=data.dtype)

    # xyz, normals, f_dc, f_rest, opacities, scale, rotation
    def save_point_cloud_to_h5(
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
    ):
        """ Function for saving the Gaussian Splatting (GS) data to the HDF5 file in a compressed way

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

        h5_fpath = os.path.join(h5file_path, h5file_name + ".h5")
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

    def load_point_cloud_from_h5(self, h5file_name: str, h5file_path: str):
        """ Function for loading data from the the HDF5 file

        Parameters
        ----------
        h5file_name: the name of the HDF file
        h5file_path: the path where to store the HDF file

        Returns
        -------
        a dictionary with loaded data
        """
        h5_fpath = os.path.join(h5file_path, h5file_name + ".h5")
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

    def pack_point_cloud_to_io_buffer(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        features_dc: np.ndarray,
        features_rest: np.ndarray,
        opacities: np.ndarray,
        scale: np.ndarray,
        rotation: np.ndarray,
        sh_degree: int,
    ):
        """ Function for packing Gaussian Splatting (GS) data to a bytes buffer in a compressed way

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
        buffer = io.BytesIO()
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

    def unpack_point_cloud_from_io_buffer(self, buffer: io.BytesIO):
        """ Function for unpacking Gaussian Splatting (GS) data from bytes buffer

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
