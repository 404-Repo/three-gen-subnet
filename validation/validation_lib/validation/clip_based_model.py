import gc
from time import time
from typing import Any

import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor, CLIPModel, CLIPProcessor


class ScoringModel:
    """ """

    def __init__(self, debug: bool = False):
        """
        Parameters
        ----------
        debug: enable/disable extra output to konsole
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._processor: CLIPProcessor = None
        self._model: CLIPModel = None
        self._debug: bool = debug

    def evaluate_image(self, images: list[Image.Image] | list[torch.Tensor], prompts: list[str]) -> Any:
        """Function for validating the input data

        Parameters
        ----------
        images: a list with rendered images defined as PIL.image
        prompts: a list of prompts that will be used for analysis

        Returns
        -------
        an estimated float score
        """

        t1 = time()
        inputs = self._processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs.to(self._device)
        t2 = time()
        if self._debug:
            logger.debug(f"Processor time: {t2 - t1} sec")

        with torch.no_grad():
            results = self._model(**inputs)

        # this is the image-text similarity score
        logits_per_image = results["logits_per_image"]

        # we can take the softmax to get the label probabilities
        probs = logits_per_image.softmax(dim=1)
        dist = probs[:, -1]
        t3 = time()
        if self._debug:
            logger.debug(f"Inference time: {t3 - t2} sec")

        return dist

    def preload_scoring_model(self, scoring_model: str = "facebook/metaclip-b16-fullcc2.5b") -> None:
        """Function for preloading the MetaClip model

        Parameters
        ----------
        scoring_model: the MetaClip model to preload and use during the validation

        """

        logger.info(" Preloading MetaClip model for validation.")

        self._processor = AutoProcessor.from_pretrained(scoring_model)
        self._model = AutoModelForZeroShotImageClassification.from_pretrained(scoring_model).to(self._device)

        logger.info("[INFO] Done.")

    def unload_model(self) -> None:
        """Function for unloading model from the GPU VRAM"""

        del self._model
        del self._processor
        self._model = None
        self._processor = None

        gc.collect()
        torch.cuda.empty_cache()
