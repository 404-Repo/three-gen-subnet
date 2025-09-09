import torch
from loguru import logger

from engine.data_structures import ValidationResponse
from engine.metrics.alignment_scorer import ImageVSImageMetric, TextVSImageMetric
from engine.metrics.quality_scorer import ImageQualityMetric
from engine.metrics.similarity_scorer import SimilarityMetrics
from engine.utils.gs_data_checker_utils import sigmoid


class ValidationEngine:
    """Class that handles all validation metrics"""

    def __init__(self, verbose: bool = False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)

        self._verbose = verbose
        self._image_quality_metric = ImageQualityMetric()
        self._text_vs_image_metric = TextVSImageMetric()
        self._image_vs_image_metric = ImageVSImageMetric()
        self._similarity_metric = SimilarityMetrics()

    def load_pipelines(self) -> None:
        """Function for loading all pipelines (metrics) that are used within the engine"""

        self._image_quality_metric.load_models()
        self._text_vs_image_metric.load_model("convnext_large_d", "laion2b_s26b_b102k_augreg")
        self._image_vs_image_metric.load_model("convnext_large_d", "laion2b_s26b_b102k_augreg")
        self._similarity_metric.load_models()

    def unload_pipelines(self) -> None:
        """Function for unloading all pipelines (metrics) from the memory"""

        self._image_quality_metric.unload_models()
        self._text_vs_image_metric.unload_model()
        self._image_vs_image_metric.unload_model()
        self._similarity_metric.unload_models()

    def _compute_image_based_metrics(self, images: list[torch.Tensor], mean_op: str) -> tuple[float, float, float]:
        """Function that computes image based metrics"""

        combined_quality_score = self._image_quality_metric.score_images_quality(images, "mean", False)
        lpips_score = self._similarity_metric.score_lpips_similarity(images, "geometric_mean", True)
        ssim_score = self._similarity_metric.score_ssim_similarity(images, "geometric_mean", True)
        return combined_quality_score, lpips_score, ssim_score

    def _compute_final_score(self, validation_results: ValidationResponse) -> ValidationResponse:
        """Function that combines all metrics' scores and combine them in a single final score"""

        if validation_results.alignment < 0.3:
            final_score = 0.0
        else:
            final_score = float(
                0.75 * validation_results.iqa
                + 0.2 * validation_results.alignment
                + 0.025 * sigmoid(torch.tensor(validation_results.ssim), 35, 0.83)
                + 0.025 * validation_results.lpips * sigmoid(torch.tensor(validation_results.lpips), 30, 0.7)
            )
        validation_results.score = final_score

        if self._verbose:
            logger.debug(f" ssim score: {validation_results.ssim}")
            logger.debug(f" lpips score: {validation_results.lpips}")
            logger.debug(f" clip score: {validation_results.alignment}")
            logger.debug(f" image quality assessment (iqa) score: {validation_results.iqa}")
            logger.debug(f" final score: {validation_results.score}")

        return validation_results

    def validate_image_to_gs(
        self, prompt_image: torch.Tensor, images: list[torch.Tensor], mean_op: str = ""
    ) -> ValidationResponse:
        """Function that validates the input 3D data generated using provided prompt-image"""

        alignment_score = self._image_vs_image_metric.score_image_alignment(
            prompt_image, images, mean_op="geometric_mean", use_filter_outliers=True
        )
        combined_quality_score, lpips_score, ssim_score = self._compute_image_based_metrics(images, mean_op)

        validation_results = ValidationResponse(
            score=0,
            iqa=combined_quality_score,
            alignment=alignment_score,
            ssim=ssim_score,
            lpips=lpips_score,
        )

        return self._compute_final_score(validation_results)

    def validate_text_to_gs(self, prompt: str, images: list[torch.Tensor], mean_op: str = "") -> ValidationResponse:
        """Function that validates the input 3D data generated using provided prompt"""

        alignment_score = self._text_vs_image_metric.score_text_alignment(
            images, prompt, mean_op="geometric_mean", use_filter_outliers=True
        )
        combined_quality_score, lpips_score, ssim_score = self._compute_image_based_metrics(images, mean_op)

        validation_results = ValidationResponse(
            score=0,
            iqa=combined_quality_score,
            alignment=alignment_score / 0.35,  # artificial normalization for current clip version
            ssim=ssim_score,
            lpips=lpips_score,
        )
        return self._compute_final_score(validation_results)
