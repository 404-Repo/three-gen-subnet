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
import open_clip


class ValidateTextTo3DModel:
    def __init__(self, img_width: int, img_height: int, views: int, device: torch.device):
        self._device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._renderer = None
        self.__views = views
        self._img_width = img_width
        self._img_height = img_height

        self._camera_params = {
            "pos": (0, 0, 0),
            "focalPoint": (0, 0, 0),
            "viewup": (0, 1, 0),
            "clipping_range": (0, 100),
        }

        self._preload_clip_model()

    '''
    Function for preloading open clip model from transformers package. 
    It preloads model and preprocessing functions for both input prompts and images
    '''

    def _preload_clip_model(self):
        bt.logging.info("Preloading CLIP model for validation.")
        self._model, _, self._preprocess = model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=self._device)
        self._tokenizer = open_clip.get_tokenizer('ViT-B-32')
        bt.logging.info("CLIP models preloaded.")

    '''
    Function that calls the validation pipeline
    @param synapses: struct like data type that contains mesh to evaluate and prompt that was used for creating the mesh
    @param cam_rad: a float parameter that defines the radius for the sphere on which the camera will be placed
    @param cam_elev: a float parameter that defines the elevation of the camera
    @param save_images: a bool parameter that controls whether to save rendered images or not
    @return a list with float scores per object stored in synapses list
    '''

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
        return scores

    '''
    Function that computes text prompt embeddings in a high dimensional space (same space for image embeddings) and normalize the embedding values
    @param prompt: a string with text prompt to be processed
    @return a torch tensor with normalized prompt embeddings
    '''
    def _get_prompt_features(self, prompt: str) -> torch.Tensor:
        text = self._tokenizer([prompt]).to(self._device)
        text_features = self._model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    '''
    Function that computes text prompt embeddings in a high dimensional space (same space for prompt embeddings) and normalize the embedding values
    @param image: a numpy ndarray with image data
    @return a torch tensor with normalized image embeddings
    '''
    def _get_image_features(self, image: np.ndarray):
        image = PIL.Image.fromarray(image)
        image = self._preprocess(image).unsqueeze(0).to(self._device)
        image_features = self._model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    '''
    Function for comparing the rendered images with the input prompt and provide a total score for the whole set of input images.
    @param images: a list with images stored as numpy arrays
    @param prompt: input prompt that was used for generating the input 3D object
    @return: averaged total score 
    '''
    def _score_images(self, images: list, prompt: str):
        dists = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for img in images:
                prompt_features = self._get_prompt_features(prompt)
                image_features = self._get_image_features(img)

                dist = torch.nn.functional.cosine_similarity(prompt_features, image_features, dim=1)
                dist_norm = dist.item()
                dists.append(dist_norm)

        # Taking the mean similarity across images
        return float(np.mean(dists))

    '''
    Function that normalizing the size of the mesh
    @param tri_mesh: a trimesh object that represents a triangular mesh
    '''
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


if __name__ == '__main__':
    test_mesh = trimesh.creation.torus(4, 2)
    prompt = "torus"

    buffer = io.BytesIO()
    test_mesh.export(buffer, file_type='ply')
    encoded_buffer = base64.b64encode(buffer.getbuffer())

    synapses = protocol.TextTo3D()
    synapses.prompt_in = prompt
    synapses.mesh_out = encoded_buffer

    validator = ValidateTextTo3DModel(512, 512, 10)
    scores = validator.score_responses([synapses], save_images=True)

    print(scores)
