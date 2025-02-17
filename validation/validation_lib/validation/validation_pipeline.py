from typing import Any

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from pytod.models.knn import KNN
from validation_lib.memory import enough_gpu_mem_available
from validation_lib.validation.clip_score_validator import ClipScoreValidator
from validation_lib.validation.combined_models_validator import CombinedModelQualityValidator
from validation_lib.validation.metric_utils import MetricUtils


def is_input_data_valid(data_dict: dict) -> bool:
    if not enough_gpu_mem_available(data_dict):
        return False

    means3d_size = data_dict["points"].shape
    if means3d_size[0] < 7000:
        return False

    zero_opacity_epsilon = 1e-3
    zero_opacity_count = torch.sum(torch.tensor(data_dict["opacities"]) < zero_opacity_epsilon).item()
    total_opacity_count = data_dict["opacities"].shape[0]
    zero_opacity_percentage = 100 * zero_opacity_count / total_opacity_count
    if zero_opacity_percentage > 80:
        return False

    zero_scales_epsilon = 0.05
    zero_scales_count = torch.sum(torch.all(torch.tensor(data_dict["scale"]) < zero_scales_epsilon)).item()
    total_scales_count = data_dict["scale"].shape[0]
    zero_scales_percentage = 100 * zero_scales_count / total_scales_count

    if zero_scales_percentage > 80:
        return False

    return True


class ValidationResult(BaseModel):
    final_score: float  # combined score
    combined_quality_score: float  # (non-normalized) combined models predictor - score
    clip_score: float  # clip similarity scores
    ssim_score: float  # structure similarity index score
    lpips_score: float  # perceptive similarity score


class ValidationPipeline:
    """Class with implementation of the validation algorithm"""

    def __init__(self, debug: bool = False):
        """
        Parameters
        ----------
        verbose: enable/disable debugging
        """
        self._clip_validator = ClipScoreValidator()
        self._combined_model_quality_validator = CombinedModelQualityValidator()
        self._debug = debug
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)

        self._metrics_utils = MetricUtils()
        self._clf = KNN(device=self._device.type)

    def validate(self, images: list[torch.Tensor], prompt: str) -> ValidationResult:
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
        # raw_quality_score = self._quality_validator.validate(images)
        combined_quality_score = self._combined_model_quality_validator.validate(images)

        # normalization to [0, 1]
        clip_score = clip_score / 0.35

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
            (
                0.75 * combined_quality_score
                + 0.2 * clip_score
                + 0.025 * self._metrics_utils.sigmoid_function(ssim_score, 35, 0.83)
                + 0.025 * lpips_score * self._metrics_utils.sigmoid_function(lpips_score, 30, 0.7)
            )
            .detach()
            .cpu()
            .numpy()
        )

        if clip_score < 0.3:
            final_score = 0.0

        if self._debug:
            logger.debug(f" ssim score: {ssim_score}")
            logger.debug(f" lpips score: {lpips_score}")
            logger.debug(f" clip score: {clip_score}")
            logger.debug(f" combined models quality score: {combined_quality_score}")
            logger.debug(f" final score: {final_score}")

        return ValidationResult(
            final_score=float(final_score),
            combined_quality_score=float(combined_quality_score),
            clip_score=float(clip_score.detach().cpu().numpy()),
            ssim_score=float(ssim_score),
            lpips_score=float(lpips_score),
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
        # self._quality_validator.preload_model("")

    def unload_model(self) -> None:
        """Function for unloading validation model from the VRAM."""

        self._clip_validator.unload_model()
        # self._quality_validator.unload_model()
