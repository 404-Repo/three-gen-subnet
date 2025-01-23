import gc
from time import time
from typing import Any

import open_clip
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
        self._tokenizer: HFTokenizer = None
        self._debug = debug
        self._preproc_img_size = preproc_img_size

    def validate(self, images: list[torch.Tensor], prompt: str) -> Any:
        """
        Function for validating the input data using transformers model

        Parameters
        ----------
        images: a list with images (renders of the generated 3D object) stored as torch tensors on the device;
        prompt: input prompt that was used for generation of the 3D object
        instruction_prompt: additional instruction prompt to guide VLM model if used [optional];

        Returns
        -------
        clip_scores: list with estimated clip scores per input image
        """

        t1 = time()
        stacked_images, tokenized_prompts = self.preprocess_inputs(images, prompt, self._preproc_img_size)

        t2 = time()

        if self._debug:
            logger.debug(f"Processor time: {t2 - t1} sec")

        with torch.no_grad(), torch.amp.autocast("cuda"):
            image_features = self._model.encode_image(stacked_images)
            text_features = self._model.encode_text(tokenized_prompts)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        clip_scores = image_features @ text_features.T

        t3 = time()
        if self._debug:
            logger.debug(f"Inference time: {t3 - t2} sec")

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

        logger.info("[INFO] Done.")

    def unload_model(self) -> None:
        """Function for unloading model from the GPU VRAM"""

        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()
