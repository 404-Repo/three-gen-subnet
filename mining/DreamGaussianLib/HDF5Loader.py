import io
import numpy as np
import h5py as h5


class HDF5Loader:
    def __init__(self):
        pass

    def _create_dataset(self, group: h5.Group, dataset_name: str, data: np.ndarray):
        group.create_dataset(
            dataset_name,
            data.shape,
            data.dtype,
            data,
            compression="gzip",
            compression_opts=9,
        )
        return group

    def _get_dataset(self, group: h5.Group, dataset_name: str):
        data = group.get(dataset_name)
        return np.array(data, dtype=data.dtype)

    def save_mesh_to_h5(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vnormals: np.ndarray,
        uvs: np.ndarray,
        texture: np.ndarray,
        h5file_name: str,
        h5file_path: str,
    ):
        h5_fpath = h5file_path + "/" + h5file_name + "_mesh.h5"
        file = h5.File(h5_fpath, mode="w")

        self._create_dataset(file, "vertices", vertices)
        self._create_dataset(file, "faces", faces)
        self._create_dataset(file, "uvs", uvs)
        self._create_dataset(file, "vnormals", vnormals)
        self._create_dataset(file, "texture", texture)

        file.close()

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
        h5_fpath = h5file_path + "/" + h5file_name + "_pcl.h5"
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
        h5_fpath = h5file_path + "/" + h5file_name + "_pcl.h5"
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
