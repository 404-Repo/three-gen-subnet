import gc

import torch
from loguru import logger

from engine.models.quality_model import QualityClassifierModel
from engine.utils.statistics_computation_utils import compute_mean, filter_outliers


class ImageQualityMetric:
    """Metric that measures the quality of the rendered images using DinoNet"""

    def __init__(self, verbose: bool = False) -> None:
        self._quality_classifier_model: QualityClassifierModel = QualityClassifierModel()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._verbose = verbose

    def load_models(
        self, repo_id: str = "404-Gen/validation", quality_scorer_model: str = "quality_scorer.pth"
    ) -> None:
        """Function for loading DinoNet quality model

        Args:
            repo_id: Hugging Face repository ID
            quality_scorer_model: Name of the quality scorer model
        """

        # Load DinoNet quality classifier
        self._quality_classifier_model.load_model(repo_id, quality_scorer_model)

        if self._verbose:
            logger.info("DinoNet quality model loaded successfully")

    def unload_models(self) -> None:
        """Function for unloading DinoNet model"""

        self._quality_classifier_model.unload_model()

        torch.cuda.empty_cache()
        gc.collect()

    def score_images_quality(
        self, images: list[torch.Tensor], mean_op: str = "mean", use_filter_outliers: bool = False
    ) -> float:
        """Function for computing quality score using DinoNet

        Args:
            images: List of torch tensors representing images
            mean_op: Type of mean operation to apply
            use_filter_outliers: Whether to filter outliers before computing mean

        Returns:
            float: Combined quality score for all images
        """

        if self._quality_classifier_model._model is None:
            raise RuntimeError("DinoNet quality model has not been loaded!")

        # Get DinoNet quality scores for all images
        final_scores = self._quality_classifier_model.score(list(images))

        if use_filter_outliers:
            final_scores = filter_outliers(final_scores)

        final_score = compute_mean(final_scores, mean_op)

        if self._verbose:
            logger.debug(f"DinoNet quality scores: {final_score}")

        return float(final_score)
