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

"""
Validation mechanism (general overview).

Input: generated mesh by the ML project (in our case Shap-E);
       format:
               buffer = io.BytesIO()
               test_mesh.export(buffer, file_type='ply')
               encoded_buffer = base64.b64encode(buffer.getbuffer()) 
       prompt that was used for generating the mesh (python str type)

Output: the score between 0 and 1 that defines the correlation between the prompt and the mesh.
        0 means no correlation, 1 means 100% match, everything in between just tells the user how far/close is the 
        generated mesh from/to the prompt. (score, python float type)

The algorithm for validation is as follows (general overview):
- Input for the evaluation algorithm is an encoded mesh object & prompt that was used for its generarion
- We want to evaluate provided prompt against several other prompts that will be always negative. We need a set of prompts as only in this case open clip will be able to estimate scores correctly.
- We render several views of the input mesh and store rendered images in the array for further processing using open clip.
- We embed both prompts array and rendred images to the same high dimensional space. Only in this case open clip will be able to figure out similarities and assign corresponding scores.
- After scoring the input we will store the score for our input prompt per rendered view and output the average score to the user.
"""


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

    """
    Function for preloading open clip model from transformers package. 
    It preloads model and preprocessing functions for both input prompts and images
    """

    def _preload_clip_model(self):
        bt.logging.info("Preloading CLIP model for validation.")

        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        bt.logging.info("CLIP models preloaded.")

    """
    Function that calls the validation pipeline
    @param synapses: struct like data type that contains mesh to evaluate and prompt that was used for creating the mesh
    @param cam_rad: a float parameter that defines the radius for the sphere on which the camera will be placed
    @param cam_elev: a float parameter that defines the elevation of the camera
    @param save_images: a bool parameter that controls whether to save rendered images or not
    @return a list with float scores per object stored in synapses list
    """

    def score_responses(
        self,
        synapses: list[protocol.TextTo3D],
        cam_rad=3.0,
        cam_elev=0,
        save_images: bool = False,
    ):
        bt.logging.info("Start scoring the response.")

        scores = np.zeros(len(synapses), dtype=float)

        # processing input data stored in the format:
        # protocol.TextTo3D: struct
        # {
        #    mesh_out: base64.b64encode buffer with mesh data
        #    prompt_in: python str object
        # }
        for synapse, i in zip(synapses, range(len(synapses))):
            if synapse.mesh_out is None:
                continue

            # decode mesh data and convert it first to trimesh triangle mesh object
            mesh_bytes = base64.b64decode(synapse.mesh_out)
            mesh = trimesh.load(io.BytesIO(mesh_bytes), file_type="ply")

            # normalize the size of the mesh to the specified domain
            self._normalize(mesh)

            # convert trimesh to vedo mesh as we use vedo python library for rendering
            vedo_mesh = vedo.Mesh((mesh.vertices, mesh.faces))

            rendered_images = []
            step = 360 // self.__views

            # render specified amount of views for the input mesh, where step = 360 // views_number
            for azimd in range(0, 360, step):
                # compute current position of the rendering camera
                azim = azimd / 180 * np.pi
                x = cam_rad * np.cos(cam_elev) * np.sin(azim)
                y = cam_rad * np.sin(cam_elev)
                z = cam_rad * np.cos(cam_elev) * np.cos(azim)

                # update position of the rendering camera, as 'focalPoint' is (0, 0, 0) it will always point to
                # the coordinate origin where the object is by default
                self._camera_params["pos"] = (x, y, z)

                # render the image using the current camera view and store the image as a numpy array
                img = vedo.show(
                    vedo_mesh,
                    camera=self._camera_params,
                    interactive=False,
                    offscreen=True,
                ).screenshot(asarray=True)

                # store images in the list that will be evaluated at the next step
                rendered_images.append(img)

            # for debugging purposes images can be saved to the disk
            if save_images:
                for j, im in enumerate(rendered_images):
                    img = PIL.Image.fromarray(im)
                    os.makedirs(os.path.curdir + "/images", exist_ok=True)
                    img.save(os.path.curdir + "/images/img" + str(j) + "_" + str(i) + ".png")

            # score the rendered images
            scores[i] = self._score_images(rendered_images, synapse.prompt_in)

        bt.logging.info("Scoring completed.")
        return torch.tensor(scores, dtype=torch.float32)

    """
    Function for comparing the rendered images with the input prompt and provide a total score for the whole set of input images.
    @param images: a list with images stored as numpy arrays
    @param prompt: input prompt that was used for generating the input 3D object
    @return: averaged total score 
    """

    def _score_images(self, images: list, prompt: str):
        dists = []

        # we add input prompt at the end of the list with prompts that will be always evaluated as negative
        prompts = self._negative_prompts.copy()
        prompts.append(prompt)

        # loop through all input images
        for img in images:
            # preprocess input data (image and prompt) using preloaded open clip preprocessing filters
            inputs = self._preprocess(text=prompts, images=[img], return_tensors="pt", padding=True)

            # process the inputs using open clip model to get the scores
            results = self._model(**inputs)

            # this is the image-text similarity score
            logits_per_image = results["logits_per_image"]

            # we can take the softmax to get the label probabilities
            probs = logits_per_image.softmax(dim=1).detach().numpy()

            # take the score for the last prompt in the self.__negative_prompts list and store in the list of results
            dists.append(probs[0][-1])

        # filter the outlier views: we want to remove false negative scores from the final score
        dists = np.sort(dists)
        count_false_detection = np.sum(dists < self._false_neg_thres)
        if count_false_detection < len(dists):
            dists = dists[dists > self._false_neg_thres]

        # return average score
        return np.mean(dists)

    """
    Function that normalizing the size of the mesh
    @param tri_mesh: a trimesh object that represents a triangular mesh
    """

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
