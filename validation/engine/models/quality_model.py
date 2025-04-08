import gc
from typing import Any

import clip
import numpy as np
import torch
import torch.nn as nn
from clip.model import CLIP
from huggingface_hub import hf_hub_download
from loguru import logger
from torchvision import transforms


class QualityClassifierModel:
    """
    A binary classifier model that uses CLIP embeddings for image classification.
    This model loads a pre-trained CLIP model and adds a binary classification head
    on top of it for specific image classification tasks.
    """

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model: CLIP | None = None
        self._model: CLIPBinaryClassifier | None = None
        self._model_path = ""
        self._checkpoint: list | tuple | dict | None = None
        self._state_dict: dict = {}

    def load_model(self, repo_id: str, filename: str, clip_model: str = "ViT-B/32") -> None:
        """Function for loading model"""

        self.clip_model, _ = clip.load(clip_model, device=self._device)
        self._model = CLIPBinaryClassifier()
        self._model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self._checkpoint = torch.load(self._model_path, map_location=self._device)

        try:
            # Handle different checkpoint formats
            if isinstance(self._checkpoint, (list | tuple)):
                if len(self._checkpoint) > 0:
                    self._state_dict = self._checkpoint[0]
                    self._model.load_state_dict(self._state_dict)
                    logger.info("Loaded model state from first element of sequence")
                else:
                    raise ValueError("Loaded checkpoint is an empty sequence")
            elif isinstance(self._checkpoint, dict):
                self._model.load_state_dict(self._checkpoint)
                logger.info("Loaded model state directly from checkpoint")
            else:
                raise TypeError(f"Unexpected checkpoint type: {type(self._checkpoint)}")

        except Exception as e:
            logger.info(f"Error loading checkpoint: {e}")
            raise
        logger.info(f"Classifier loaded to device {self._device}")
        self._model.eval()

    def unload_model(self) -> None:
        """Function for unloading model"""

        del self._model
        del self.clip_model
        del self._checkpoint

        self._model = None
        self.clip_model = None
        self._checkpoint = None

        torch.cuda.empty_cache()
        gc.collect()

    def score(self, images: list[torch.Tensor]) -> np.ndarray:
        """Function for Generation of classification scores for a batch of images"""

        if self._model is None:
            raise RuntimeError("The model has not been loaded!")

        image_features = self.preprocess_inputs(images)

        with torch.no_grad():
            scores = self._model(image_features)
        return np.array(scores.cpu().detach().numpy())

    def preprocess_inputs(self, images: list[torch.Tensor]) -> Any:
        """Preprocess images for input to the model"""
        if self.clip_model is None:
            raise RuntimeError("The quality model was not initialized!")

        normalized_images: torch.Tensor

        stacked_images = torch.stack(images, dim=0).to(self._device) / 255.0
        stacked_images = stacked_images.permute(0, 3, 1, 2).to(torch.float16)

        # Define normalization parameters
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3

        normalize = transforms.Normalize(mean, std)
        normalized_images = normalize(stacked_images)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(normalized_images.to(normalized_images.device))

        # Normalize features and apply classification head
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        image_features = image_features.to(torch.float32)
        return image_features


class CLIPBinaryClassifier(nn.Module):
    """
    Binary classifier that uses CLIP embeddings as input features.

    Args:
        clip_model (nn.Module): Pre-trained CLIP model to use as feature extractor.
            The model should have an encode_image method that outputs 512-dimensional features.

    Attributes:
        clip_model (nn.Module): Frozen CLIP model for feature extraction
        classifier (nn.Sequential): Classification head for binary prediction
    """

    def __init__(self) -> None:
        super().__init__()
        # Define classification head architecture
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # First linear layer reduces dimensionality
            nn.ReLU(),  # Non-linear activation
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(256, 1),  # Output layer for binary classification
            nn.Sigmoid(),  # Sigmoid activation for [0,1] output
        )

        # Initialize linear layers using Kaiming initialization
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                # Initialize weights using Kaiming normalization
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # Initialize biases to zero
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, image_features_normed: torch.Tensor) -> Any:
        """
        Forward pass of the model.

        Args:
            image_features_normed (torch.Tensor): Batch of input image normalized features.
                Expected shape: (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Binary predictions in range [0,1].
                Shape: (batch_size, 1)
        """

        # Pass through classification head
        return self.classifier(image_features_normed)
