from typing import Any

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure


class MetricUtils:
    """Class with methods that provides extra information for validation."""

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lpips_model = lpips.LPIPS(net="alex").to(self._device)
        self._ssim_score_cuda = StructuralSimilarityIndexMeasure(data_range=1.0).to(self._device)

    def compute_ssim_across_views(self, images: list[torch.Tensor]) -> tuple[float, list[torch.Tensor]]:
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
        for i in range(len(images) - 1):
            image1 = images[i].unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            image2 = images[i + 1].unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            ssim_score = self._ssim_score_cuda(image1, image2)
            ssim_scores.append(ssim_score.detach().cpu().numpy())

        ssim_score_mean = np.exp(np.log(ssim_scores).mean())
        return ssim_score_mean, ssim_scores

    def compute_lpips_score(self, images: list[torch.Tensor]) -> tuple[float, list[torch.Tensor]]:
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
        for i in range(len(images) - 1):
            image1 = images[i].unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            image1 = F.interpolate(image1, size=(256, 256), mode="bicubic", align_corners=False)
            image2 = images[i + 1].unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            image2 = F.interpolate(image2, size=(256, 256), mode="bicubic", align_corners=False)

            with torch.no_grad():
                lpips_score = self._lpips_model(image1, image2).detach().cpu().numpy()
                lpips_scores.append(lpips_score)

        lpips_score_mean = np.exp(np.log(lpips_scores).mean())
        return lpips_score_mean, lpips_scores

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
