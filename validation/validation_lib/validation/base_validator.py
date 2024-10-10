from abc import ABC, abstractmethod

import torch


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, images: list[torch.Tensor], prompt: str, instruction_prompt: str = "") -> torch.Tensor:
        """
        Function that validates input images against input prompt that was used for generation of the 3D model;

        Parameters
        ----------
        images: a list with images (renders of the generated 3D object) stored as torch tensors on the device;
        prompt_list: a list of prompts or a single prompt defines as string/s;
        instruction_prompt: additional instruction prompt to guide VLM model if used [optional];

        Returns
        -------
        a score stored as a torch.Tensor
        """
        pass

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
