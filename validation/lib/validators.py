import base64
import io
import os

import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

from lib.camera_utils import orbit_camera, OrbitCamera
from lib.hdf5_loader import HDF5Loader
from lib.gaussian_splatting_renderer import GSRenderer, BasicCamera


class TextTo3DModelValidator:
    def __init__(self, img_width, img_height, views: int = 10, device="cuda"):
        self.__device = torch.device(device)
        self.__model = None
        self.__preprocess = None
        self.__renderer = None
        self.__views = views
        self.__img_width = img_width
        self.__img_height = img_height

        self.__hdf5_loader = HDF5Loader()
        self.__negative_prompts = [
            "empty",
            "nothing",
            "false",
            "wrong",
            "negative",
            "not quite right",
        ]
        self.__false_neg_thres = 0.4

        self._preload_clip_model()

    def init_gaussian_splatting_renderer(self, sh_degree: int = 3, white_background: bool = True, radius: float = 1.0):
        self.__renderer = GSRenderer(sh_degree, white_background, radius)

    def score_response_gs_input(
        self,
        prompt: str,
        data: str,
        cam_rad=1.5,
        cam_elev=0,
        save_images: bool = False,
    ):
        print("[INFO] Start scoring the response.")

        orbitcam = OrbitCamera(self.__img_width, self.__img_height, r=cam_rad, fovy=49.1)

        pcl_raw = base64.b64decode(data)
        pcl_buffer = io.BytesIO(pcl_raw)
        data_dict = self.__hdf5_loader.unpack_point_cloud_from_io_buffer(pcl_buffer)

        self.__renderer.initialize(data_dict)

        rendered_images = []
        step = 360 // self.__views

        # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
        min_ver = max(min(-20, -20 - cam_elev), -30 - cam_elev)
        max_ver = min(max(20, 20 - cam_elev), 30 - cam_elev)

        for azimd in range(0, 360, step):
            ver = np.random.randint(min_ver, max_ver)

            pose = orbit_camera(cam_elev + ver, azimd, cam_rad)
            camera = BasicCamera(
                pose,
                self.__img_width,
                self.__img_height,
                orbitcam.fovy,
                orbitcam.fovx,
                orbitcam.near,
                orbitcam.far,
            )

            output_dict = self.__renderer.render(camera)
            img = output_dict["image"].permute(1, 2, 0)
            img = img.detach().cpu().numpy() * 255
            img = np.concatenate((img, 255 * np.ones((img.shape[0], img.shape[1], 1))), axis=2).astype(np.uint8)
            img = Image.fromarray(img)
            rendered_images.append(img)

        if save_images:
            for j, im in enumerate(rendered_images):
                im.save(os.path.curdir + "/images/img" + str(j) + "_" + str(i) + ".png")

        score = self._score_images(rendered_images, prompt)

        print("[INFO] Done.")
        return score

    def _preload_clip_model(self):
        print("[INFO] Preloading CLIP model for validation.")

        self.__model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.__preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        print("[INFO] Done.")

    def _score_images(self, images: list, prompt):
        dists = []
        self.__negative_prompts.append(prompt)
        prompts = self.__negative_prompts
        for img in images:
            inputs = self.__preprocess(text=prompts, images=[img], return_tensors="pt", padding=True)
            results = self.__model(**inputs)
            logits_per_image = results["logits_per_image"]  # this is the image-text similarity score
            probs = (
                logits_per_image.softmax(dim=1).detach().numpy()
            )  # we can take the softmax to get the label probabilities
            dists.append(probs[0][-1])

        dists = np.sort(dists)
        count_false_detection = np.sum(dists < self.__false_neg_thres)
        if count_false_detection < len(dists):
            dists = dists[dists > self.__false_neg_thres]

        return np.mean(dists)
