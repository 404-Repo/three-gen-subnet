import base64
import io
import os

import bittensor as bt
import numpy as np
import PIL
import protocol
import torch
import trimesh
import vedo
from transformers import CLIPModel, CLIPProcessor


class ValidateTextTo3DModel:
    def __init__(self, img_width: int, img_height: int, views: int, device: torch.device):
        self._device = device
        self._model = None
        self._preprocess = None
        self._renderer = None
        self.__views = views
        self._img_width = img_width
        self._img_height = img_height

        self._negative_prompts = [
            "empty",
            "nothing",
            "false",
            "wrong",
            "negative",
            "not quite right",
        ]
        self._false_neg_thres = 0.4

        self._camera_params = {
            "pos": (0, 0, 0),
            "focalPoint": (0, 0, 0),
            "viewup": (0, 1, 0),
            "clipping_range": (0, 100),
        }

        self._preload_clip_model()

    def _preload_clip_model(self):
        bt.logging.info("Preloading CLIP model for validation.")

        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        bt.logging.info("CLIP models preloaded.")

    def score_responses(
        self,
        synapses: list[protocol.TextTo3D],
        cam_rad=3.0,
        cam_elev=0,
        save_images: bool = False,
    ):
        bt.logging.info("Start scoring the response.")

        scores = np.zeros(len(synapses), dtype=float)

        for synapse, i in zip(synapses, range(len(synapses))):
            if synapse.mesh_out is None:
                continue

            mesh_bytes = base64.b64decode(synapse.mesh_out)
            mesh = trimesh.load(io.BytesIO(mesh_bytes), file_type="ply")
            self._normalize(mesh)

            vedo_mesh = vedo.Mesh((mesh.vertices, mesh.faces))

            rendered_images = []
            step = 360 // self.__views

            for azimd in range(0, 360, step):
                azim = azimd / 180 * np.pi
                x = cam_rad * np.cos(cam_elev) * np.sin(azim)
                y = cam_rad * np.sin(cam_elev)
                z = cam_rad * np.cos(cam_elev) * np.cos(azim)

                self._camera_params["pos"] = (x, y, z)
                img = vedo.show(
                    vedo_mesh,
                    camera=self._camera_params,
                    interactive=False,
                    offscreen=True,
                ).screenshot(asarray=True)
                rendered_images.append(img)

            if save_images:
                for j, im in enumerate(rendered_images):
                    img = PIL.Image.fromarray(im)
                    img.save(os.path.curdir + "/images/img" + str(j) + "_" + str(i) + ".png")

            scores[i] = self._score_images(rendered_images, synapse.prompt_in)

        bt.logging.info("Scoring completed.")
        return scores

    def _score_images(self, images: list, prompt: str):
        dists = []
        self._negative_prompts.append(prompt)
        prompts = self._negative_prompts
        for img in images:
            inputs = self._preprocess(text=prompts, images=[img], return_tensors="pt", padding=True)
            results = self._model(**inputs)
            logits_per_image = results["logits_per_image"]  # this is the image-text similarity score
            probs = (
                logits_per_image.softmax(dim=1).detach().numpy()
            )  # we can take the softmax to get the label probabilities
            dists.append(probs[0][-1])

        dists = np.sort(dists)
        count_false_detection = np.sum(dists < self._false_neg_thres)
        if count_false_detection < len(dists):
            dists = dists[dists > self._false_neg_thres]

        del self._negative_prompts[-1]

        return np.mean(dists)

    def _normalize(self, tri_mesh: trimesh.base.Trimesh):
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
