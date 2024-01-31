import base64
import io
from traceback import print_exception
from typing import Any, Callable

import bittensor as bt
import PIL
import pyrender
import clip
import torch
import trimesh
import numpy as np
from pyrender import RenderFlags

from neurons import protocol


class Validate3DModels:
    def __init__(
        self, model: torch.nn.Module, preprocess: Callable[[PIL.Image], torch.Tensor]
    ):
        self.model = model
        self.preprocess = preprocess


class Renderer:
    def __init__(self):
        self.r = pyrender.OffscreenRenderer(
            viewport_width=512, viewport_height=512, point_size=1.0
        )

        w = 512
        h = 512
        f = 0.5 * w / np.tan(0.5 * 55 * np.pi / 180.0)
        cx = 0.5 * w
        cy = 0.5 * h
        znear = 0.01
        zfar = 100
        self.pc = pyrender.IntrinsicsCamera(
            fx=f, fy=f, cx=cx, cy=cy, znear=znear, zfar=zfar
        )


def score_responses(
    prompt: str,
    synapses: list[protocol.TextTo3D],
    device: torch.device,
    models: Validate3DModels,
) -> torch.Tensor:
    scores = np.zeros(len(synapses), dtype=float)
    renderer = Renderer()

    try:
        prompt_features = _get_prompt_features(prompt, device, models)
        for i, synapse in enumerate(synapses):
            if synapse.mesh_out is None:
                continue

            mesh = base64.b64decode(synapse.mesh_out)

            images = _render_images(mesh, renderer)
            scores[i] = _score_images(images, device, models, prompt_features)

            # import matplotlib.pyplot as plt
            #
            # for x in range(4):
            #     plt.imshow(images[x])
            #     plt.savefig(f'image{x}.png')
    except Exception as e:
        bt.logging.exception(f"Error during scroring: {e}")
        bt.logging.debug(print_exception(type(e), e, e.__traceback__))
    finally:
        renderer.r.delete()  # It's important to free the resources

    return torch.tensor(scores, dtype=torch.float32)


def load_models(device: torch.device, cache_dir: str) -> Validate3DModels:
    download_root = f"{cache_dir}/clip"
    model, preprocess = clip.load(
        "ViT-B/32", device=device, jit=False, download_root=download_root
    )
    return Validate3DModels(model, preprocess)


def _get_prompt_features(
    prompt: str, device: torch.device, models: Validate3DModels
) -> torch.Tensor:
    text = clip.tokenize([prompt]).to(device)
    prompt_features = models.model.encode_text(text)
    return prompt_features


def _score_images(
    images: list[Any],
    device: torch.device,
    models: Validate3DModels,
    prompt_features: torch.Tensor,
) -> float:
    with torch.no_grad():
        dists = []
        for img in images:
            img_tensor = (
                models.preprocess(PIL.Image.fromarray(img)).unsqueeze(0).to(device)
            )
            image_features = models.model.encode_image(img_tensor)
            dist = torch.nn.functional.cosine_similarity(
                prompt_features, image_features, dim=1
            )
            dist_norm = dist.item()
            dists.append(dist_norm)

        # Taking the mean similarity across images
        return float(np.mean(dists))


def _render_images(mesh_bytes: bytes, renderer: Renderer, views=4) -> list[np.ndarray]:
    mesh = trimesh.load(io.BytesIO(mesh_bytes), file_type="ply")
    _normalize(mesh)

    flags = RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL

    images = []
    frame = 0
    step = 360 // views
    for azimd in range(0, 360, step):
        if isinstance(mesh, trimesh.Trimesh):
            scene = trimesh.Scene()
            scene.add_geometry(mesh)
            scene = pyrender.Scene.from_trimesh_scene(scene)
        else:
            # trimeshScene = fuze_trimesh
            scene = pyrender.Scene.from_trimesh_scene(mesh)

        camera_matrix = np.eye(4)

        dist = 1.0
        elev = 0
        azim = azimd / 180 * np.pi
        # azim = 0
        x = dist * np.cos(elev) * np.sin(azim)
        y = dist * np.sin(elev)
        z = dist * np.cos(elev) * np.cos(azim)

        camera_matrix[0, 3] = x
        camera_matrix[1, 3] = y
        camera_matrix[2, 3] = z

        camera_matrix_out = np.eye(4)
        camera_matrix_out[0, 3] = x
        camera_matrix_out[1, 3] = z
        camera_matrix_out[2, 3] = y

        camera_matrix[0, 0] = np.cos(azim)
        camera_matrix[2, 2] = np.cos(azim)
        camera_matrix[0, 2] = np.sin(azim)
        camera_matrix[2, 0] = -np.sin(azim)

        camera_matrix_out[0, 0] = np.cos(azim)
        camera_matrix_out[1, 1] = np.cos(azim)
        camera_matrix_out[0, 1] = np.sin(azim)
        camera_matrix_out[1, 0] = -np.sin(azim)

        point_l = pyrender.PointLight(color=np.ones(3), intensity=3.0)

        nc = pyrender.Node(camera=renderer.pc, matrix=camera_matrix)
        scene.add_node(nc)
        ncl = pyrender.Node(light=point_l, matrix=camera_matrix)
        scene.add_node(ncl)

        color, depth = renderer.r.render(scene, flags=flags)

        images.append(color)

        frame += 1

    return images


def _normalize(tri_mesh: trimesh.base.Trimesh):
    vert_np = np.asarray(tri_mesh.vertices)
    minv = vert_np.min(0)
    maxv = vert_np.max(0)

    half = (minv + maxv) / 2

    scale = maxv - minv
    scale = scale.max()
    scale = 1 / scale

    vert_np = vert_np - half
    vert_np = vert_np * scale
    np.copyto(tri_mesh.vertices, vert_np)
