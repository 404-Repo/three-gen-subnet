from typing import Any

import numpy as np
import torch


class OrbitCamera:
    """Class that defines the camera object."""

    def __init__(
        self,
        img_width: int,
        img_height: int,
        fov_y: float = 49.1,
        z_near: float = 0.01,
        z_far: float = 100,
        degrees: bool = True,
    ):
        """Constructor

        Parameters
        ----------
        img_width: the width of the camera image
        img_height: the height of the camera image
        fov_y: the field of view for the camera
        z_near: the position of the near camera plane along Z-axis
        z_far: the position of the far camera plane along Z-axis
        degrees: True if the input fov is in degrees otherwise set to False
        """

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # setting camera transform: camera -> world
        self._cam_to_world_tr = torch.eye(4, dtype=torch.float32)

        self._img_height = img_height
        self._img_width = img_width
        self._z_near = z_near
        self._z_far = z_far

        if degrees:
            self._fov_y = np.deg2rad(fov_y)
        else:
            self._fov_y = fov_y

    @property
    def camera_to_world_tr(self) -> torch.Tensor:
        """Matrix with camera to world transform"""
        return self._cam_to_world_tr

    @property
    def world_to_camera_transform(self) -> torch.Tensor:
        """Matrix with transform from world space to camera space"""
        # return torch.inverse(self._cam_to_world_tr).transpose(0, 1).to(self._device)
        R = self._cam_to_world_tr[:3, :3].transpose(0, 1)
        T = -R @ self._cam_to_world_tr[:3, 3].unsqueeze(1)
        Tr = torch.cat((R, T), dim=1)
        result = torch.eye(4)
        result[:3, :4] = Tr
        return result.to(self._device)

    @property
    def camera_position(self) -> torch.Tensor:
        """A vector with current camera position"""
        return -self._cam_to_world_tr[:3, 3]

    @property
    def tan_half_fov(self) -> Any:
        """A tan of the half value of the field of view value"""
        return np.tan(0.5 * self._fov_y)

    @property
    def fov(self) -> Any:
        """Field pof view in radians"""
        return self._fov_y

    @property
    def image_height(self) -> int:
        """The height of the camera image"""
        return self._img_height

    @property
    def image_width(self) -> int:
        """The width of the camera image"""
        return self._img_width

    @property
    def z_near(self) -> float:
        """The value of the near camera plane"""
        return self._z_near

    @property
    def z_far(self) -> float:
        """The value of the far camera plane"""
        return self._z_far

    @property
    def intrinsics(self) -> torch.Tensor:
        """Function for computing the intrinsics for the camera"""
        focal_x = self._img_width / (2 * self.tan_half_fov)
        focal_y = self._img_height / (2 * self.tan_half_fov)

        Ks = torch.eye(3)
        Ks[0, 0] = focal_x
        Ks[1, 1] = focal_y
        Ks[0, 2] = self._img_width // 2
        Ks[1, 2] = self._img_height // 2
        return Ks.to(self._device)

    def set_camera_to_world_transform(self, transform: torch.Tensor) -> None:
        """Function for setting up the camera to world transform"""
        self._cam_to_world_tr = transform

    def look_at(self, camera_pos: torch.Tensor, target_pos: torch.Tensor, opengl_conv: bool = True) -> torch.Tensor:
        """Function for computing the rotation matrix for the camera

        Parameters
        ----------
        camera_pos: current camera position in space
        target_pos: target position in space
        opengl_conv: enables opengl conversion if True, otherwise CUDA conversion will be used.

        Returns
        -------
        a torch tensor with rotation matrix
        """
        if opengl_conv:
            # camera forward aligns with +z
            forward_vector = self._safe_normalize(camera_pos - target_pos)
            up_vector = torch.tensor([0, 1, 0], dtype=torch.float32)
            right_vector = self._safe_normalize(torch.linalg.cross(up_vector, forward_vector))
            up_vector = self._safe_normalize(torch.linalg.cross(forward_vector, right_vector))
        else:
            # camera forward aligns with -z
            forward_vector = self._safe_normalize(target_pos - camera_pos)
            up_vector = torch.tensor([0, 1, 0], dtype=torch.float32)
            right_vector = self._safe_normalize(torch.linalg.cross(forward_vector, up_vector))
            up_vector = self._safe_normalize(torch.linalg.cross(right_vector, forward_vector))

        R = torch.stack((right_vector, up_vector, forward_vector), dim=1)
        return R

    def compute_transform_orbit(
        self,
        elevation: float,
        azimuth: float,
        radius: float,
        is_degree: bool = True,
        target_pos: torch.Tensor | None = None,
        opengl_conv: bool = True,
    ) -> None:
        """Function for computing orbit transform for the current camera

        Parameters
        ----------
        elevation: the elevation on a sphere in degrees/radians
        azimuth: the horizontal azimuth angle in degrees/radiansw
        radius: radius of camera orbit
        is_degree: True if the angles are in degrees
        target_pos: the position of the target (object) in space
        opengl_conv: enables opengl conversion if True, otherwise CUDA conversion will be used.

        """
        if is_degree:
            elevation = np.deg2rad(elevation)
            azimuth = np.deg2rad(azimuth)

        x = radius * np.cos(elevation) * np.sin(azimuth)
        y = -radius * np.sin(elevation)
        z = radius * np.cos(elevation) * np.cos(azimuth)

        if target_pos is None:
            target_pos = torch.zeros([3], dtype=torch.float32)

        campos = torch.tensor([x, y, z], dtype=torch.float32) + target_pos

        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = self.look_at(campos, target_pos, opengl_conv)
        T[:3, 3] = campos
        T[:3, 1:3] *= -1

        self._cam_to_world_tr = T

    @staticmethod
    def _length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
        """Function for computing the length of the input vector

        Parameters
        ----------
        x: a vector for which a length will be computed
        eps: accuracy of the output data

        Returns
        -------
        a float value equal to the length of the vector
        """
        return torch.sqrt(torch.clamp(torch.dot(x, x), min=eps))

    def _safe_normalize(self, x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
        """Function for computing the normalization of the input vector

        Parameters
        ----------
        x: a vector that will be normalized
        eps:  accuracy of the output data

        Returns
        -------
        a normalized vector
        """
        return x / self._length(x, eps)
