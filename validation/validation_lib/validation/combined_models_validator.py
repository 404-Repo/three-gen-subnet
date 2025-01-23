import numpy as np
import torch
from huggingface_hub import hf_hub_download
from joblib import load

from .models.combined_models import AestheticModel, ClassifierModel


# set these model paths
REPO_ID = "404-Gen/validation"
CLASSIFIER_MODEL = "score_based_classifier.pth"
AESTHETIC_PREDICTOR_MODEL = "aesthetic_predictor.pth"
POLYNOMIAL_MODEL = "poly_fit.joblib"


class CombinedModelQualityValidator:
    """Class with implementation of the validation algorithm"""

    def __init__(self, debug: bool = False):
        """
        Parameters
        ----------
        verbose: enable/disable debugging
        """
        self._classifier_validator = ClassifierModel(REPO_ID, CLASSIFIER_MODEL)
        self._aesthetics_validator = AestheticModel(REPO_ID, AESTHETIC_PREDICTOR_MODEL)
        self._polynomial_pipeline = load(hf_hub_download(REPO_ID, POLYNOMIAL_MODEL))
        self._debug = debug
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._validator_type = "combined_score"
        torch.set_default_device(self._device)

    def validate(self, images: list[torch.Tensor]) -> float:
        """
        Function for validating the input data

        Parameters
        ----------
        images: a list with rendered images stored as torch.tensors, type = torch.uint8, same device
        """
        classifier_validator_predictions = self._classifier_validator.score(list(images)).squeeze()
        aesthetic_validator_predictions = self._aesthetics_validator.score(list(images)).squeeze()
        X = np.column_stack((classifier_validator_predictions, aesthetic_validator_predictions))
        combined_score_v1_temp = self._polynomial_pipeline.predict(X.reshape(1, -1))
        combined_score_v1 = float(np.array(combined_score_v1_temp).squeeze().clip(max=1.0).mean())
        return combined_score_v1
