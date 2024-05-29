import torch
import numpy as np


class OrbitCamera():
    """ Class that defines the camera object. """
    def __init__(self, img_width: int,
                 img_height: int,
                 fov_y: float = 49.1,
                 z_near: float = 0.01,
                 z_far: float = 100,
                 degrees: bool = True):
        """ Constructor

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
    def camera_to_world_tr(self):
        """ Matrix with camera to world transform """
        return self._cam_to_world_tr

    @property
    def world_view_transform(self):
        """ Matrix with transform from camera space to world space """
        return torch.inverse(self._cam_to_world_tr).transpose(0, 1).to(self._device)

    @property
    def full_projection_transform(self):
        """ Matrix with full camera projection transform """
        return self.world_view_transform @ self.get_projection_matrix()

    @property
    def camera_position(self):
        """ A vector with current camera position """
        return -self._cam_to_world_tr[:3, 3]

    @property
    def tan_half_fov(self):
        """ A tan of the half value of the field of view value """
        return np.tan(0.5 * self._fov_y)

    @property
    def fov(self):
        """ Field pof view in radians """
        return self._fov_y

    @property
    def image_height(self):
        """ The height of the camera image """
        return self._img_height

    @property
    def image_width(self):
        """  The width of the camera image """
        return self._img_width

    @property
    def z_near(self):
        """ The value of the near camera plane """
        return self._z_near

    @property
    def z_far(self):
        """ The value of the far camera plane """
        return self._z_far

    def get_projection_matrix(self):
        """ Function for computing the projection matrix for the camera """
        tan_half_fov = self.tan_half_fov

        P = torch.zeros(4, 4)
        P[0, 0] = 1.0 / tan_half_fov
        P[1, 1] = 1.0 / tan_half_fov
        P[2, 2] = (self._z_far + self._z_near) / (self._z_far - self._z_near)
        P[3, 2] = -(self._z_far * self._z_near) / (self._z_far - self._z_near)
        P[2, 3] = 1

        return P.to(self._device)

    def get_intrinsics(self):
        """ Function for computing the intrinsics for the camera """
        focal = self._img_height / (2 * self.tan_half_fov)
        return np.array([focal, focal, self._img_width // 2, self._img_height // 2], dtype=np.float32)

    def set_camera_to_world_transform(self, transform: torch.tensor):
        """ Function for setting up the camera to world transform """
        self._cam_to_world_tr = transform

    def look_at(self, camera_pos: torch.Tensor, target_pos: torch.Tensor, opengl_conv: bool = True):
        """ Function for computing the rotation matrix for the camera

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

    def compute_transform_orbit(self, elevation: float,
                                azimuth: float,
                                radius: float,
                                is_degree: bool = True,
                                target_pos: torch.Tensor = None,
                                opengl_conv: bool = True):
        """ Function for computing orbit transform for the current camera

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
    def _length(x: torch.Tensor, eps: float = 1e-20):
        """ Function for computing the length of the input vector

        Parameters
        ----------
        x: a vector for which a length will be computed
        eps: accuracy of the output data

        Returns
        -------
        a float value equal to the length of the vector
        """
        return torch.sqrt(torch.clamp(torch.dot(x, x), min=eps))

    def _safe_normalize(self, x: torch.Tensor, eps: float = 1e-20):
        """ Function for computing the normalization of the input vector

        Parameters
        ----------
        x: a vector that will be normalized
        eps:  accuracy of the output data

        Returns
        -------
        a normalized vector
        """
        return x / self._length(x, eps)
