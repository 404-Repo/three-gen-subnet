import gc
from typing import List

import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from loguru import logger
from PIL import Image


class ScoringModel:
    """ """

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._processor = None
        self._model = None

    def evaluate_image(self, images: List[Image.Image], prompts: List[str]):
        """Function for validating the input data

        Parameters
        ----------
        images: a list with rendered images defined as PIL.image
        prompts: a list of prompts that will be used for analysis

        Returns
        -------
        an estimated float score
        """

        inputs = self._processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs.to(self._device)
        results = self._model(**inputs)

        # this is the image-text similarity score
        logits_per_image = results["logits_per_image"]

        # we can take the softmax to get the label probabilities
        probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()

        dist = probs[:, -1]

        return dist

    def preload_scoring_model(self, scoring_model: str = "facebook/metaclip-b16-fullcc2.5b"):
        """Function for preloading the MetaClip model

        Parameters
        ----------
        scoring_model: the MetaClip model to preload and use during the validation

        """

        logger.info(" Preloading MetaClip model for validation.")

        self._processor = AutoProcessor.from_pretrained(scoring_model)
        self._model = AutoModelForZeroShotImageClassification.from_pretrained(scoring_model).to(self._device)

        logger.info("[INFO] Done.")

    def unload_model(self):
        """Function for unloading model from the GPU VRAM"""

        del self._model
        del self._processor
        self._model = None
        self._processor = None

        gc.collect()
        torch.cuda.empty_cache()
