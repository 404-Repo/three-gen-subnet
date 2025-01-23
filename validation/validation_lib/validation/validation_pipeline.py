import math
from typing import Any

import numpy as np
import torch
from loguru import logger
from PromptIQA.prompt_iqa_pipeline import PromptIQAPipeline
from pydantic import BaseModel
from pytod.models.knn import KNN
from validation_lib.validation.clip_score_validator import ClipScoreValidator
from validation_lib.validation.vqascore_validator import VQAScoreValidator
from validation_lib.validation.metric_utils import MetricUtils


class ValidationResult(BaseModel):
    final_score: float  # combined score
    quality_score: float  # prompy-iqa output
    clip_score: float  # clip similarity score
    ssim_score: float  # structure similarity index score
    lpips_score: float  # perceptive similarity score
    sharpness_score: float  # laplacian variance score
    vqa_score: float # vqascore


class ValidationPipeline:
    """Class with implementation of the validation algorithm"""

    def __init__(self, debug: bool = False):
        """
        Parameters
        ----------
        verbose: enable/disable debugging
        """
        self._clip_validator = ClipScoreValidator()
        self._quality_validator = PromptIQAPipeline()
        self._vqa_validator = VQAScoreValidator()
        self._debug = debug
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)

        self._metrics_utils = MetricUtils()
        self._clf = KNN(device=self._device.type)

    def validate(self, source_images: list[torch.Tensor], images: list[torch.Tensor], prompt: str) -> ValidationResult:
        """
        Function for validating the input data

        Parameters
        ----------
        source_images: a list of torch tensors with the source image (rendered from a selected view)
        images: a list with rendered images stored as torch.tensors, type = torch.uint8, same device
        prompt: a string with input prompt that was used for generating the 3D model
        """

        # compute two clip scores
        clip_score = self.compute_clip_score(images, prompt)
        quality_scores = []
        for img in source_images:
            quality_score = self._quality_validator.compute_quality(img)
            quality_scores.append(quality_score.detach().cpu().numpy()[0])
        quality_score = np.array(quality_scores).mean()

        # normalization to [0, 1]
        clip_score = clip_score / 0.35

        vqa_scores = self._vqa_validator.validate(images, prompt)
        vqa_score = vqa_scores.mean()

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

        # laplacian variance
        sharpness_score = self._metrics_utils.compute_laplacian_variance(images)

        updated_quality_score = min(1.0, math.exp(2 * quality_score - 1.5) - 0.45)

        # compute final score
        final_score = (
            0.75 * updated_quality_score
            + 0.2 * clip_score
            + 0.025 * self._metrics_utils.sigmoid_function(ssim_score, 35, 0.83)
            + 0.025 * lpips_score * self._metrics_utils.sigmoid_function(lpips_score, 30, 0.7)
        ).detach().cpu().numpy()

        if clip_score < 0.3:
            final_score = 0.0

        if self._debug:
            logger.debug(f" ssim score: {ssim_score}")
            logger.debug(f" lpips score: {lpips_score}")
            logger.debug(f" clip score: {clip_score}")
            logger.debug(f" vqa score: {vqa_score}")
            logger.debug(f" quality score: {quality_score}")
            logger.debug(f" sharpness score: {sharpness_score}")
            logger.debug(f" final score: {final_score}")

        return ValidationResult(
            final_score=float(final_score),
            quality_score=float(quality_score),
            clip_score=float(clip_score.detach().cpu().numpy()),
            ssim_score=float(ssim_score),
            lpips_score=float(lpips_score),
            sharpness_score=float(sharpness_score.detach().cpu().numpy()),
            vqa_score=vqa_score
        )

    def compute_clip_score(self, images: list[torch.Tensor], prompt: str) -> Any:
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

        clip_score = self._clip_validator.validate(images, prompt)

        # filtering outliers
        filtered_scores = self.filter_outliers(clip_score).clip(0, 1)
        clip_score_final = torch.exp(torch.log(filtered_scores).mean())

        return clip_score_final

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

    def preload_model(self, clip_model: str = "convnext_large_d", preload: str = "laion2b_s26b_b102k_augreg") -> None:
        """
        Function for preloading the scoring model (clip-based) for rendered images analysis
        metaclip: ViT-B-16-quickgelu, metaclip_fullcc

        Parameters
        ----------
        clip_model: the name of the CLIP-based model to preload and use during the validation;
        preload: model checkpoint, should be used only with openclip [optional]
        """

        self._clip_validator.preload_model(clip_model, preload)
        self._vqa_validator.preload_model("llava-v1.5-7b")
        self._quality_validator.load_pipeline()

    def unload_model(self) -> None:
        """Function for unloading validation model from the VRAM."""

        self._clip_validator.unload_model()
