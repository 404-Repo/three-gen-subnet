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

    def validate(
        self, source_images: list[torch.Tensor], images: list[torch.Tensor], prompt: str
    ) -> tuple[Any, Any, Any, Any, Any]:
        """
        Function for validating the input data

        Parameters
        ----------
        source_image: a torch tensor with source image (rendered from a selected view)
        images: a list with rendered images stored as torch.tensors, type = torch.uint8, same device
        prompt: a string with input prompt that was used for generating the 3D model

        Returns
        -------
        final_score: combined score
        vqa_score: combined score
        clip_score: clip similarity score
        ssim: structure similarity index score
        lpips: perceptive similarity score
        """

        # compute two clip scores
        vqa_score, clip_score = self.compute_clip_score(source_images, images, prompt)

        # compute other metrics
        # ssim score
        _, ssim_scores = self._metrics_utils.compute_ssim_across_views(images)
        self._clf.fit(torch.tensor(ssim_scores).reshape(-1, 1))
        preds = self._clf.labels_
        outliers = np.where(preds == 1)[0]
        filtered_ssim = np.delete(ssim_scores, outliers)
        ssim_score = np.exp(np.log(filtered_ssim).mean())

        # lpips score
        _, lpips_scores = self._metrics_utils.compute_lpips_score(images)
        self._clf.fit(torch.tensor(lpips_scores).reshape(-1, 1))
        preds = self._clf.labels_
        outliers = np.where(preds == 1)[0]
        filtered_lpips = np.delete(lpips_scores, outliers)
        lpips_score = np.exp(np.log(filtered_lpips).mean())
        lpips_score = 1 - lpips_score

        # compute final score
        final_score = (
            0.5 * vqa_score
            + 0.4 * clip_score
            + 0.05 * self._metrics_utils.sigmoid_function(ssim_score, 35, 0.83)
            + 0.05 * lpips_score * self._metrics_utils.sigmoid_function(lpips_score, 30, 0.7)
        )

        if self._debug:
            logger.debug(f" ssim score: {ssim_score}")
            logger.debug(f" lpips score: {lpips_score}")
            logger.debug(f" vqa score, prev.image: {vqa_score}")
            logger.debug(f" clip score: {clip_score}")
            logger.debug(f" final score: {final_score}")

        return final_score, vqa_score, clip_score, ssim_score, lpips_score

    def compute_clip_score(
        self, source_image: list[torch.Tensor], images: list[torch.Tensor], prompt: str
    ) -> tuple[Any, Any]:
        """
        Function for validating the input data using clip-based model

        Parameters
        ----------
        source_image: a torch tensor with source image (rendered from a selected view)
        images: a list with rendered images stored as torch.tensors, type = torch.uint8, same device
        prompt: a string with input prompt that was used for generating the 3D model

        Returns
        -------
        vqa_score: vqa score computed for source image + prompt
        clip_score_final: clip score computed for source image + rendered images
        """
        # clip score
        vqa_score = self._clip_validator.validate_source_image(source_image[0], prompt)
        clip_score = self._clip_validator.validate_images(source_image[1], images)

        # filtering outliers
        filtered_scores = self.filter_outliers(clip_score)
        clip_score_final = torch.exp(torch.log(filtered_scores).mean())

        return vqa_score, clip_score_final

    def filter_outliers(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Function for filtering the outliers
        Parameters
        ----------
        input_data: an input torch tensor that will be processed

        Returns
        -------
        filtered_input: a torch tensor with filtered values
        """
        # normalizing clip scores to range [0, 1] and sorting
        input_sorted, _ = torch.sort(input_data)
        input_sorted = input_sorted.reshape(-1, 1)

        # searching for the anomalies
        self._clf.fit(input_sorted)
        preds = self._clf.labels_
        outliers = np.where(preds == 1)[0]
        filtered_input = np.delete(input_sorted.cpu().detach().numpy(), outliers)

        return torch.tensor(filtered_input).to(self._device)

    def preload_model(self, clip_model: str = "ViT-B-16-SigLIP", preload: str = "webli") -> None:
        """
        Function for preloading the scoring model (clip-based) for rendered images analysis
        metaclip: ViT-B-16-quickgelu, metaclip_fullcc

        Parameters
        ----------
        clip_model: the name of the CLIP-based model to preload and use during the validation;
        preload: model checkpoint, should be used only with openclip [optional]
        """

        self._clip_validator.preload_model(clip_model, preload)

    def unload_model(self) -> None:
        """Function for unloading validation model from the VRAM."""

        self._clip_validator.unload_model()
