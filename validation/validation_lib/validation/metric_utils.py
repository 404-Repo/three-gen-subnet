from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class MetricUtils:
    """Class with methods that provides extra information for validation."""

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)

        self._lpips_model = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self._device)
        self._ssim_score_cuda = StructuralSimilarityIndexMeasure(data_range=1.0).to(self._device)
        self._images_order = [1, 4, 10, 11, 9, 8, 11, 12, 3, 12, 15, 8, 15, 5, 9, 12]

    def compute_ssim_across_views(self, images: list[torch.Tensor]) -> tuple[float, np.ndarray]:
        """
        Function for computing structural similarity score between input images

        Parameters
        ----------
        images: list of images defined as torch.tensor

        Returns
        -------
        ssim_score_mean: an averaged ssim score
        ssim_scores: a list with ssim scores
        """

        ssim_scores = []
        for i, j in zip(range(len(images)), self._images_order, strict=False):
            image1 = images[i].unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            image2 = images[j].unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            ssim_score = self._ssim_score_cuda(image1, image2)
            ssim_scores.append(ssim_score.detach().cpu().numpy())

        ssim_score_mean = np.exp(np.log(ssim_scores).mean())
        return ssim_score_mean, np.array(ssim_scores)

    def compute_lpips_score(self, images: list[torch.Tensor]) -> tuple[float, np.ndarray]:
        """
        Function for computing Learned Perceptual Image Patch Similarity (LPIPS) score.
        It captures perceptual differences that are not captured by traditional metrics like PSNR or SSIM,
        focusing more on how humans perceive image similarity.

        Parameters
        ----------
        images: list of images defined as torch.tensor

        Returns
        -------
        lpips_score_mean: an averaged lpips score
        lpips_scores: a list with lpips scores
        """

        lpips_scores = []

        for i, j in zip(range(len(images)), self._images_order, strict=False):
            image1 = images[i].unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            image1 = torch.clip(F.interpolate(image1, size=(256, 256), mode="bicubic", align_corners=False), 0, 1)
            image2 = images[j].unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            image2 = torch.clip(F.interpolate(image2, size=(256, 256), mode="bicubic", align_corners=False), 0, 1)
            lpips_score = self._lpips_model(image1, image2).detach().cpu().numpy()
            lpips_scores.append(lpips_score.flatten())

        lpips_score_mean = np.exp(np.log(lpips_scores).mean())
        return lpips_score_mean, np.array(lpips_scores).T

    @staticmethod
    def sigmoid_function(x: float, slope: float, x_shift: float) -> Any:
        """
        Function for remapping input data using sigmoid function

        Parameters
        ----------
        x: input value that will be remapped using sigmoid function
        slope: a float parameter that controls the slope of the sigmoid function
        x_shift: a float paramter that shifts the sigmoid curve along X axis

        Returns
        -------
        a remapped float value
        """
        return 1.0 / (1.0 + np.exp(-slope * (x - x_shift)))
