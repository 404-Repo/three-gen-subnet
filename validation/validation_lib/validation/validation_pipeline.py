from typing import Any

import numpy as np
import torch
from loguru import logger
from pytod.models.knn import KNN
from validation_lib.validation.clip_score_validator import ClipScoreValidator
from validation_lib.validation.metric_utils import MetricUtils


class ValidationPipeline:
    """Class with implementation of the validation algorithm"""

    def __init__(self, debug: bool = False):
        """
        Parameters
        ----------
        verbose: enable/disable debugging
        """

        self._clip_validator = ClipScoreValidator()
        self._debug = debug
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)

        self._metrics_utils = MetricUtils()
        self._clf = KNN(device=self._device.type)

    def validate(self, images: list[torch.Tensor], prompt: str) -> tuple[Any, Any, Any, Any]:
        """
        Function for validating the input data

        Parameters
        ----------
        images: a list with rendered images stored as torch.tensors, type = torch.uint8, same device
        prompt: a string with input prompt that was used for generating the 3D model

        Returns
        -------
        final_score: combined score
        clip_score: clip similarity score
        ssim: structure similarity index score
        lpips: perceptive similarity score
        """

        return self.compute_clip_score(images, prompt)

    def compute_clip_score(self, images: list[torch.Tensor], prompt: str) -> tuple[Any, Any, Any, Any]:
        """
        Function for validating the input data using clip-based model

        Parameters
        ----------
        images: a list with rendered images stored as torch.tensors, type = torch.uint8, same device
        prompt: a string with input prompt that was used for generating the 3D model

        Returns
        -------
        final_score: combined score
        clip_score: clip similarity score
        ssim: structure similarity index score
        lpips: perceptive similarity score
        """

        dists = self._clip_validator.validate(images, prompt)
        dists_sorted, _ = torch.sort(dists)

        # normalizing clip score to range [0, 1]
        dists_sorted = dists_sorted.reshape(-1, 1) / 0.45

        # searching for the anomalies
        # Set contamination to expected proportion of outliers
        self._clf.fit(dists_sorted)
        preds = self._clf.labels_

        # removing found outliers from the input dists array
        outliers = np.where(preds == 1)[0]
        filtered_dists = np.delete(dists_sorted.cpu().detach().numpy(), outliers)

        clip_score = np.exp(np.log(filtered_dists).mean())
        ssim_score, _ = self._metrics_utils.compute_ssim_across_views(images)
        lpips_score, _ = self._metrics_utils.compute_lpips_score(images)

        final_score = (
            (self._metrics_utils.sigmoid_function(clip_score, 12, 0.6))
            * (self._metrics_utils.sigmoid_function(ssim_score, 45, 0.82))
            * self._metrics_utils.sigmoid_function((1 - lpips_score), 28, 0.81)
        )

        if self._debug:
            logger.debug(f" filtered scores: {filtered_dists.T}")
            logger.debug(f" outliers: {dists_sorted[outliers].T} \n")
            logger.debug(f" ssim score: {ssim_score}")
            logger.debug(f" lpips score: {lpips_score}")
            logger.debug(f" clip score: {clip_score}")
            logger.debug(f" final score: {final_score}")

        return final_score, clip_score, ssim_score, lpips_score

    def preload_model(self, clip_model: str = "ViT-B-16-quickgelu", preload: str = "metaclip_fullcc") -> None:
        """
        Function for preloading the scoring model (clip-based) for rendered images analysis

        Parameters
        ----------
        clip_model: the name of the CLIP-based model to preload and use during the validation;
        preload: model checkpoint, should be used only with openclip [optional]
        """

        self._clip_validator.preload_model(clip_model, preload)

    def unload_model(self) -> None:
        """Function for unloading validation model from the VRAM."""

        self._clip_validator.unload_model()
