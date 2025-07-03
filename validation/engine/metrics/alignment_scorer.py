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

from engine.utils.statistics_computation_utils import compute_mean, filter_outliers


class TextVSImageMetric:
    """Metric that checks the alignment of prompt vs rendered images of the input 3D data"""

    def __init__(self, verbose: bool = False) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: CLIP | None = None
        self._tokenizer: HFTokenizer | None = None
        self._verbose = verbose

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3
        self._normalize_transform = transforms.Normalize(mean, std)

    def load_model(self, model_name: str, pretrained: str = "") -> None:
        """Function that loading the model for prompt-vs-images alignment"""

        logger.info("Loading text vs image alignment model.")
        self._model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self._device
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()

    def unload_model(self) -> None:
        """Function that unloads model from the memory"""

        logger.info("Unloading text vs image alignment model.")
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()

    def preprocess_images(self, images: list[torch.Tensor], image_res: int) -> torch.Tensor:
        """Function for preprocessing input images"""

        stacked_images: torch.Tensor = torch.stack(images, dim=0).to(self._device) / 255.0
        stacked_images = stacked_images.permute(0, 3, 1, 2).to(torch.float16)
        stacked_images = F.interpolate(stacked_images, size=(image_res, image_res), mode="bicubic", align_corners=False)
        stacked_images = self._normalize_transform(stacked_images)
        return stacked_images

    def tokenize_prompt(self, prompt: str) -> Any:
        """Function for tokenization of the input prompt"""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer was not initialized!")

        return self._tokenizer(prompt)

    def score_text_alignment(
        self,
        images: list[torch.Tensor],
        prompt: str,
        img_preproc_res: int = 224,
        mean_op: str = "mean",
        use_filter_outliers: bool = False,
    ) -> float:
        """Function for computing the alignment score"""

        if self._model is None:
            raise RuntimeError("The model was not initialized!")

        t1 = time()
        preprocessed_images = self.preprocess_images(images, img_preproc_res)
        tokenized_prompt = self.tokenize_prompt(prompt)

        with torch.no_grad(), torch.amp.autocast(self._device.type):
            image_features = self._model.encode_image(preprocessed_images)
            text_features = self._model.encode_text(tokenized_prompt)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_scores = (image_features @ text_features.T).to(torch.float32)

        if use_filter_outliers:
            clip_scores = filter_outliers(clip_scores)

        clip_scores = torch.clip(clip_scores, 0, 1)
        clip_score = compute_mean(clip_scores, mean_op)

        t2 = time()
        if self._verbose:
            logger.debug(f"Text vs Image alignment score computation took: {t2 - t1} sec")

        return float(clip_score)


class ImageVSImageMetric:
    """Metric that checks the alignment of prompt-image vs rendered images of the input 3D data"""

    def __init__(self, verbose: bool = False) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: CLIP | None = None
        self._verbose = verbose

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3
        self._normalize_transform = transforms.Normalize(mean, std)

    def load_model(self, model_name: str, pretrained: str = "") -> None:
        """Function that loading the model for prompt-image-vs-images alignment"""

        logger.info("Loading text vs image alignment model.")
        self._model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self._device
        )
        self._model.eval()

    def unload_model(self) -> None:
        """Function that unloads model from the memory"""

        logger.info("Unloading image vs image alignment model.")
        del self._model
        self._model = None

        torch.cuda.empty_cache()
        gc.collect()

    def preprocess_images(self, images: list[torch.Tensor], image_res: int) -> torch.Tensor:
        """Function for preprocessing input images"""

        stacked_images: torch.Tensor = torch.stack(images, dim=0).to(self._device) / 255.0
        stacked_images = stacked_images.permute(0, 3, 1, 2).to(torch.float16)
        stacked_images = F.interpolate(stacked_images, size=(image_res, image_res), mode="bicubic", align_corners=False)
        stacked_images = self._normalize_transform(stacked_images)
        return stacked_images

    def score_image_alignment(
        self,
        prompt_image: torch.Tensor,
        images: list[torch.Tensor],
        img_preproc_res: int = 224,
        mean_op: str = "mean",
        use_filter_outliers: bool = False,
    ) -> float:
        """Function for computing the alignment score"""

        if self._model is None:
            raise RuntimeError("The model was not initialized!")

        t1 = time()
        preproc_prompt_image = self.preprocess_images([prompt_image], img_preproc_res)
        preproc_images = self.preprocess_images(images, img_preproc_res)

        with torch.no_grad(), torch.amp.autocast(self._device.type):
            images_features = self._model.encode_image(preproc_images)
            preview_image_features = self._model.encode_image(preproc_prompt_image)
            images_features /= images_features.norm(dim=-1, keepdim=True)
            preview_image_features /= preview_image_features.norm(dim=-1, keepdim=True)
        clip_scores = (images_features @ preview_image_features.T).to(torch.float32)

        if use_filter_outliers:
            clip_scores = filter_outliers(clip_scores)
        clip_score = compute_mean(clip_scores, mean_op)

        t2 = time()
        if self._verbose:
            logger.debug(f"Image vs Image alignment score computation took: {t2 - t1} sec")

        return float(clip_score)
