from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, images: list[torch.Tensor], prompt: str) -> Any:
        """
        Function for validating the input data using transformers model

        Parameters
        ----------
        images: a list with images (renders of the generated 3D object) stored as torch tensors on the device;
        prompt: input prompt that was used for generation of the 3D object
        instruction_prompt: additional instruction prompt to guide VLM model if used [optional];

        Returns
        -------
        clip_scores: list with estimated clip scores per input image
        """

    @abstractmethod
    def preload_model(self, model_name: str, pretrained: str = "") -> None:
        """
        Function for preloading multimodal language model in GPU memory

        Parameters
        ----------
        model_name: model id to load (HF model_id or one of the models supported by openclip)
        pretrained: pretrained model checkpoint [optional, applicable if openclip is in use]
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Function for unloading the model"""
        pass
