import numpy as np
import torch

from validation.gs_camera import OrbitCamera


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_camera_properties():
    camera = OrbitCamera(256, 128, 60, 0.01, 100)

    assert camera.image_width == 256
    assert camera.image_height == 128
    assert camera.fov == np.deg2rad(60)
    assert camera.z_far == 100
    assert camera.z_near == 0.01
    assert camera.tan_half_fov == np.tan(0.5 * np.deg2rad(60))


def test_camera_projection_matrix():
    camera = OrbitCamera(256, 256, 90, 0.01, 100)
    proj_mat = camera.get_projection_matrix()

    P_ref = torch.zeros((4, 4), dtype=torch.float32).to(device)
    P_ref[0, 0] = 1.0
    P_ref[1, 1] = 1.0
    P_ref[2, 2] = (100.0 + 0.01) / (100.0 - 0.01)
    P_ref[3, 2] = -(100.0 * 0.01) / (100.0 - 0.01)
    P_ref[2, 3] = 1

    assert torch.allclose(proj_mat, P_ref)


def test_camera_orbiting():
    camera = OrbitCamera(256, 256, 49.1, 0.01, 100)

    camera_pos = torch.tensor([1, 1, 0], dtype=torch.float32)
    target_pos = torch.tensor([0, 0, 0], dtype=torch.float32)
    R_gl_ref = torch.tensor([[0, -0.7071, 0.7071],
                             [0, 0.7071, 0.7071],
                             [-1, 0, 0]], dtype=torch.float32)
    R = camera.look_at(camera_pos, target_pos, opengl_conv=True)
    assert torch.allclose(R, R_gl_ref)

    R_cu_ref = torch.tensor([[0.0000, -0.7071, -0.7071],
                            [0.0000,  0.7071, -0.7071],
                            [-1.0000,  0.0000,  0.0000]], dtype=torch.float32)
    R_cu = camera.look_at(camera_pos, target_pos, opengl_conv=False)
    assert torch.allclose(R_cu, R_cu_ref)

    camera.compute_transform_orbit(0, 0, 1)
    camera_to_world1 = camera.camera_to_world_tr
    camera_tr1_ref = torch.tensor([[1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, -1, 1],
                                  [0, 0, 0, 1]], dtype=torch.float32)
    assert torch.allclose(camera_to_world1, camera_tr1_ref)

    camera.compute_transform_orbit(30, 45, 1)
    camera_world2 = camera.camera_to_world_tr
    camera_tr2_ref = torch.tensor([[0.7071, -0.3536, -0.6124, 0.6124],
                                  [0.0000, -0.8660,  0.5000, -0.5000],
                                  [-0.7071, -0.3536, -0.6124, 0.6124],
                                  [0.0000, 0.0000, 0.0000, 1.0000]], dtype=torch.float32)
    assert torch.allclose(camera_world2, camera_tr2_ref, atol=1e-4)
