import inspect
import os
import sys


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/validation")

import numpy as np
import torch
from validation_lib.rendering.gs_camera import OrbitCamera


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_camera_properties():
    camera = OrbitCamera(256, 128, 60, 0.01, 100)

    assert camera.image_width == 256
    assert camera.image_height == 128
    assert camera.fov == np.deg2rad(60)
    assert camera.z_far == 100
    assert camera.z_near == 0.01
    assert camera.tan_half_fov == np.tan(0.5 * np.deg2rad(60))

    Ks = camera.intrinsics
    focal_x = 256 / (2 * np.tan(0.5 * np.deg2rad(60)))
    focal_y = 128 / (2 * np.tan(0.5 * np.deg2rad(60)))
    xc = 256 // 2
    yc = 128 // 2

    Ks_ref = torch.tensor([[focal_x, 0, xc], [0, focal_y, yc], [0, 0, 1]], dtype=torch.float32).to(device)

    assert torch.allclose(Ks, Ks_ref)


def test_camera_orbiting():
    camera = OrbitCamera(256, 256, 49.1, 0.01, 100)

    camera_pos = torch.tensor([1, 1, 0], dtype=torch.float32)
    target_pos = torch.tensor([0, 0, 0], dtype=torch.float32)
    R_gl_ref = torch.tensor([[0, -0.7071, 0.7071], [0, 0.7071, 0.7071], [-1, 0, 0]], dtype=torch.float32)
    R = camera.look_at(camera_pos, target_pos, opengl_conv=True)
    assert torch.allclose(R, R_gl_ref)

    R_cu_ref = torch.tensor(
        [[0.0000, -0.7071, -0.7071], [0.0000, 0.7071, -0.7071], [-1.0000, 0.0000, 0.0000]], dtype=torch.float32
    )
    R_cu = camera.look_at(camera_pos, target_pos, opengl_conv=False)
    assert torch.allclose(R_cu, R_cu_ref)

    camera.compute_transform_orbit(0, 0, 1)
    camera_to_world1 = camera.camera_to_world_tr
    camera_tr1_ref = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]], dtype=torch.float32)
    assert torch.allclose(camera_to_world1, camera_tr1_ref)

    camera.compute_transform_orbit(30, 45, 1)
    camera_world2 = camera.camera_to_world_tr
    camera_tr2_ref = torch.tensor(
        [
            [0.7071, -0.3536, -0.6124, 0.6124],
            [0.0000, -0.8660, 0.5000, -0.5000],
            [-0.7071, -0.3536, -0.6124, 0.6124],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(camera_world2, camera_tr2_ref, atol=1e-4)
