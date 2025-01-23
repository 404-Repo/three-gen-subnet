import gc
from typing import Any

import numpy as np
import torch
from validation_lib.validation.base_validator import BaseValidator
import t2v_metrics


class VQAScoreValidator(BaseValidator):
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._model = None

    def validate(self, images: list[torch.Tensor], prompt: str) -> Any:
        """
        Function for validating the input data using transformers model

        Parameters
        ----------
        images: a list with images (renders of the generated 3D object) stored as torch tensors on the device;
        prompt: input prompt that was used for generation of the 3D object

        Returns
        -------
        clip_scores: list with estimated clip scores per input image
        """
        view_images = [images[0], images[3], images[7], images[11]]
        clip_scores = []
        score = self._model.forward(view_images, [prompt])
        clip_scores.append(score.detach().cpu().numpy())

        return np.array(clip_scores)

    def preload_model(self, model_name: str="", pretrained: str = "") -> None:
        """
        Function for preloading multimodal language model in GPU memory

        Parameters
        ----------
        model_name: model id to load (HF model_id or one of the models supported by openclip)
        pretrained: pretrained model checkpoint [optional, applicable if openclip is in use]
        """

        cache_dir = t2v_metrics.constants.HF_CACHE_DIR
        self._model = t2v_metrics.get_score_model(model=model_name, device=self._device.type, cache_dir=cache_dir)

    def unload_model(self) -> None:
        """Function for unloading the model"""
        del self._model
        torch.cuda.empty_cache()
        gc.collect()
        self._model = None

