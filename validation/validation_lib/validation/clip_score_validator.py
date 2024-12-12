import gc
from time import time
from typing import Any

import open_clip
import t2v_metrics
import torch
import torch.nn.functional as F
from loguru import logger
from open_clip import CLIP
from open_clip.tokenizer import HFTokenizer
from torchvision import transforms
from validation_lib.validation.base_validator import BaseValidator


class ClipScoreValidator(BaseValidator):
    """Class that implements validation of the input images/data using CLIP-based model"""

    def __init__(self, debug: bool = False, preproc_img_size: int = 224):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._model: CLIP = None
        self._t2v_metrcics = t2v_metrics.VQAScore().to(self._device)
        self._tokenizer: HFTokenizer = None
        self._debug = debug
        self._preproc_img_size = preproc_img_size

    def validate_source_image(self, source_image: torch.Tensor, prompt: str) -> Any:
        """
        Function for validating the rendered front view of the object
        Parameters
        ----------
        validate_source_image: a rendered view of the object that can be considered as a source (preview) image
                               for generating 3D object;
        prompt: input prompt that was used for generating 3D object

        Returns
        -------
        vqa_clip_score: a float score value
        """
        t1 = time()
        vqa_clip_score = self._t2v_metrcics([source_image], [prompt])
        t2 = time()
        if self._debug:
            logger.debug(f"VQA score computation took: {(t2 - t1)} sec")
        return vqa_clip_score

    def validate_images(self, preview_image: torch.Tensor, images: list[torch.Tensor]) -> Any:
        """
        Function for validating the input data using transformers model

        Parameters
        ----------
        preview_image:
        images: a list with images (renders of the generated 3D object) stored as torch tensors on the device;

        Returns
        -------
        clip_scores: list with estimated clip scores per input image
        """

        stacked_images, _ = self.preprocess_inputs(images, "", self._preproc_img_size)
        image, _ = self.preprocess_inputs([preview_image], "", self._preproc_img_size)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            images_features = self._model.encode_image(stacked_images)
            preview_image_features = self._model.encode_image(image)
            images_features /= images_features.norm(dim=-1, keepdim=True)
            preview_image_features /= preview_image_features.norm(dim=-1, keepdim=True)
            clip_scores = images_features @ preview_image_features.T

        return clip_scores.to(torch.float32)

    def preprocess_inputs(
        self,
        images: list[torch.Tensor],
        prompt: str = "",
        image_res: int = 224,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        images: a list with images (renders of the generated 3D object) stored as torch tensors on the device;
        prompt: input prompt that was used for generation of the 3D object
        image_res: an int value defining the size of the square image

        Returns
        -------
        stacked_images: a torch tensor with stacked input images
        tokenized_prompts: a torch tensor with tokenized input prompts
        """
        stacked_images = torch.stack(images, dim=0).to(self._device) / 255.0
        stacked_images = stacked_images.permute(0, 3, 1, 2).to(torch.float16)
        stacked_images = F.interpolate(stacked_images, size=(image_res, image_res), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3
        normalize = transforms.Normalize(mean, std)
        stacked_images = normalize(stacked_images)
        tokenized_prompts = self._tokenizer(prompt)

        return stacked_images, tokenized_prompts

    def preload_model(self, model_name: str, pretrained: str = "") -> None:
        """
        Function for preloading one of the openclip models

        Parameters
        ----------
        model_name: the openclip model to preload
        pretrained: the pretrained checkpoint to use
        """

        logger.info(" Preloading OpenClip model for validation.")
        self._model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self._device
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()

        self._t2v_metrcics.preload_model("clip-flant5-xl")

        logger.info("[INFO] Done.")

    def unload_model(self) -> None:
        """Function for unloading model from the GPU VRAM"""

        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()
