import gc

import numpy as np
import sklearn
import torch
from huggingface_hub import hf_hub_download
from joblib import load

from engine.models.aethtetic_model import AestheticsPredictorModel
from engine.models.quality_model import QualityClassifierModel
from engine.utils.statistics_computation_utils import compute_mean, filter_outliers


class ImageQualityMetric:
    """Metric that measures the quality of the rendered images"""

    def __init__(self, verbose: bool = False) -> None:
        self._quality_classifier_model: QualityClassifierModel = QualityClassifierModel()
        self._aesthetics_predictor_model: AestheticsPredictorModel = AestheticsPredictorModel()
        self._polynomial_pipeline_model: sklearn.pipeline.Pipeline | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._verbose = verbose

    def load_models(
        self,
        repo_id: str = "404-Gen/validation",
        quality_classifier_model: str = "score_based_classifier_params.pth",
        aesthetics_predictor_model: str = "aesthetic_predictor.pth",
        polynomial_pipeline_model: str = "poly_fit.joblib",
    ) -> None:
        """Function for loading all models"""

        self._quality_classifier_model.load_model(repo_id, quality_classifier_model)
        self._aesthetics_predictor_model.load_model(repo_id, aesthetics_predictor_model)
        self._polynomial_pipeline_model = load(hf_hub_download(repo_id, polynomial_pipeline_model))

    def unload_models(self) -> None:
        """Function for unloading all models"""

        self._quality_classifier_model.unload_model()
        self._aesthetics_predictor_model.unload_model()
        self._polynomial_pipeline_model = None

        torch.cuda.empty_cache()
        gc.collect()

    def score_images_quality(
        self, images: list[torch.Tensor], mean_op: str = "mean", use_filter_outliers: bool = False
    ) -> float:
        """Function for computing quality score of the input data"""

        if self._quality_classifier_model is None:
            raise RuntimeError("Quality Classifier model has not been loaded!")
        elif self._aesthetics_predictor_model is None:
            raise RuntimeError("Aesthetic Predictor model has not been loaded!")
        elif self._polynomial_pipeline_model is None:
            raise RuntimeError("Polynomial pipeline model has not been loaded!")

        classifier_validator_predictions = self._quality_classifier_model.score(list(images)).squeeze()
        aesthetic_validator_predictions = self._aesthetics_predictor_model.score(list(images)).squeeze()
        X = np.column_stack((classifier_validator_predictions, aesthetic_validator_predictions))
        combined_score_v1 = self._polynomial_pipeline_model.predict(X.reshape(1, -1))
        final_scores = torch.tensor(combined_score_v1).squeeze().clip(max=1.0)

        if use_filter_outliers:
            final_scores = filter_outliers(torch.tensor(final_scores))
        final_score = compute_mean(final_scores, mean_op)

        return float(final_score)
