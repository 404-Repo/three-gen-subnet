from typing import Any

import numpy as np
import torch
from loguru import logger
from PIL import Image
from pytod.models.knn import KNN
from validation_lib.validation.clip_based_model import ScoringModel


class ValidationPipeline:
    """Class with implementation of the validation algorithm"""

    def __init__(self, verbose: bool = False, debug: bool = False):
        """
        Parameters
        ----------
        verbose: enable/disable debugging
        """

        self._clip_processor = ScoringModel()
        self._verbose = verbose
        self._debug = debug
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._clf = KNN(device=self._device.type)

        self._negative_prompts = [
            "empty",
            "nothing",
            "false",
            "wrong",
            "negative",
            "not quite right",
        ]

    def validate(self, images: list[Image.Image], prompt: str) -> Any:
        """
        Function for validating the input data

        Parameters
        ----------
        images: a list with rendered images stored as torch.tensors, type = torch.uint8, same device
        prompt: a string with input prompt that was used for generating the 3D model

        Returns
        -------
        clip_score: a float evaluation score
        """

        clip_score = self.compute_clip_score(images, prompt)
        return clip_score

    def compute_clip_score(self, images: list[Image.Image], prompt: str) -> Any:
        """
        Function for validating the input data using clip-based model

        Parameters
        ----------
        images: a list with rendered images stored as torch.tensors, type = torch.uint8, same device
        prompt: a string with input prompt that was used for generating the 3D model

        Returns
        -------
        score: a float evaluation score
        """

        if self._verbose:
            logger.info(" Validating input data.")

        prompts = self._negative_prompts + [
            prompt,
        ]
        dists = self._clip_processor.evaluate_image(images, prompts)
        dists_sorted, _ = torch.sort(dists)
        dists_sorted = dists_sorted.reshape(-1, 1)

        # searching for the anomalies
        # Set contamination to expected proportion of outliers
        self._clf.fit(dists_sorted)
        preds = self._clf.labels_

        # removing found outliers from the input dists array
        outliers = np.where(preds == 1)[0]
        filtered_dists = np.delete(dists_sorted.cpu().detach().numpy(), outliers)
        score = np.mean(filtered_dists)

        if self._debug:
            logger.debug(f" data: {dists_sorted.T}")
            logger.debug(f" outliers: {dists_sorted[outliers].T}")
            logger.debug(f" filtered scores: {filtered_dists.T}")
            logger.debug(f" score: {score}")

        if self._verbose:
            logger.info(" Done.")

        return score

    def preload_model(self, clip_model: str = "facebook/metaclip-b16-fullcc2.5b") -> None:
        """
        Function for preloading the scoring model (clip-based) for rendered image analysis

        Parameters
        ----------
        clip_model: the name of the CLIP-based model to preload and use during the validation
        """
        if self._verbose:
            logger.info(" Preloading model for scoring.")

        self._clip_processor.preload_scoring_model(clip_model)

        if self._verbose:
            logger.info(" Done.")
