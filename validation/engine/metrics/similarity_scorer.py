import gc

import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from engine.utils.statistics_computation_utils import compute_mean, filter_outliers


class SimilarityMetrics:
    """Metric that computes two similarity scores: SSIM and LPIPS"""

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lpips_metric: LearnedPerceptualImagePatchSimilarity | None = None
        self._ssim_metric: StructuralSimilarityIndexMeasure | None = None

    def load_models(self) -> None:
        """Function for loading models / pipelines"""

        self._lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self._device).eval()
        self._ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(self._device).eval()

    def unload_models(self) -> None:
        """Function for unloading models/pipelines"""

        del self._lpips_metric
        del self._ssim_metric

        self._lpips_metric = None
        self._ssim_metric = None

        torch.cuda.empty_cache()
        gc.collect()

    def score_lpips_similarity(
        self, images: list[torch.Tensor], mean_op: str = "mean", use_filter_outliers: bool = False
    ) -> float:
        """
        Function for computing Learned Perceptual Image Patch Similarity (LPIPS) score.
        It captures perceptual differences that are not captured by traditional metrics like PSNR or SSIM,
        focusing more on how humans perceive image similarity.
        """

        if self._lpips_metric is None:
            raise RuntimeError("The lpips pipeline was not initialized.")

        with torch.no_grad():
            proc = [
                torch.clip(
                    F.interpolate(
                        img.unsqueeze(0).permute(0, 3, 1, 2).to(self._device) / 255,
                        (256, 256),
                        mode="bicubic",
                        align_corners=False,
                    ),
                    0,
                    1,
                ).half()
                for img in images
            ]

            x1 = torch.cat(proc, 0)  # pair (i)
            x2 = torch.cat(proc[1:] + proc[:1], 0)  # pair (i+1)

            lpips_scores_torch = self._lpips_metric(x1, x2).squeeze().detach().cpu()

        return 1 - float(lpips_scores_torch)

    def score_ssim_similarity(
        self, images: list[torch.Tensor], mean_op: str = "mean", use_filter_outliers: bool = False
    ) -> float:
        """Function for computing structural similarity score between input images"""

        if self._ssim_metric is None:
            raise RuntimeError("The ssim pipeline was not initialized.")

        with torch.no_grad():
            # Batch process all images for efficiency
            proc = [img.unsqueeze(0).permute(0, 3, 1, 2).to(self._device) / 255.0 for img in images]

            x1 = torch.cat(proc, 0)  # pair (i)
            x2 = torch.cat(proc[1:] + proc[:1], 0)  # pair (i+1)

            ssim_scores_torch = self._ssim_metric(x1, x2).squeeze().detach().cpu()

        if use_filter_outliers:
            ssim_scores_torch = filter_outliers(ssim_scores_torch)
        ssim_score_mean = compute_mean(ssim_scores_torch, mean_op)

        return float(ssim_score_mean)
