from abc import ABC, abstractmethod

import torch


class BaseValidator(ABC):
    @abstractmethod
    def validate_images(self, preview_image: torch.Tensor, images: list[torch.Tensor]) -> torch.Tensor:
        """
        Function that validates input images against input prompt that was used for generation of the 3D model;

        Parameters
        ----------
        preview_image:
        images: a list with images (renders of the generated 3D object) stored as torch tensors on the device;

        Returns
        -------
        a score stored as a torch.Tensor
        """
        pass

    @abstractmethod
    def validate_source_image(self, source_image: torch.Tensor, prompt: str) -> torch.Tensor:
        """
        Function for validating the rendered front view of the object
        Parameters
        ----------
        validate_source_image: a rendered view of the object that can be considered as a source (preview)
                               image for generating 3D object;
        prompt: input prompt that was used for generating 3D object

        Returns
        -------
        vqa_clip_score: a float score value
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
