import warnings
from typing import Any

import clip
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from loguru import logger
from PIL import Image
from torch import Tensor
from torchvision import transforms


warnings.filterwarnings("ignore", category=FutureWarning)


class ClassifierModel:
    """
    A binary classifier model that uses CLIP embeddings for image classification.

    This model loads a pre-trained CLIP model and adds a binary classification head
    on top of it for specific image classification tasks.

    Attributes:
        _device (torch.device): The device (CPU/GPU) where the model runs
        clip_model (nn.Module): Pre-trained CLIP model
        _model (CLIPBinaryClassifier): The complete classification model
    """

    def __init__(self, repo_id: str, filename: str) -> None:
        # def __init__(self, MODEL_PATH_Classifier: str | Path) -> None:
        """
        Initialize the classifier model.

        Args:
            MODEL_PATH_Classifier: Path to the pre-trained classifier checkpoint

        Raises:
            ValueError: If the checkpoint is empty
            TypeError: If the checkpoint format is unexpected
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # torch.set_default_device(self._device)
        self.clip_model, _ = clip.load("ViT-B/32", device=self._device)
        self._model = CLIPBinaryClassifier(self.clip_model)
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

        self._model.eval()

    def score(self, images: list[torch.Tensor]) -> np.ndarray:
        """
        Generate classification scores for a batch of images.

        Args:
            images: List of image tensors to be classified

        Returns:
            numpy.ndarray: Classification scores for each image
        """
        stacked_images = self.preprocess_inputs(images)
        with torch.no_grad():
            scores = self._model(stacked_images)
        return np.array(scores.cpu().detach().numpy())

    def preprocess_inputs(
        self,
        images: list[torch.Tensor],
        image_res: int = 224,
    ) -> torch.Tensor:
        """
        Preprocess images for input to the model.

        Args:
            images: List of image tensors to preprocess
            image_res: Target resolution for the images (default: 224)

        Returns:
            torch.Tensor: Preprocessed and normalized image tensor batch
        """
        normalized_iamges: Tensor

        stacked_images = torch.stack(images, dim=0).to(self._device) / 255.0
        stacked_images = stacked_images.permute(0, 3, 1, 2).to(torch.float16)

        # Define normalization parameters
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 3
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 3

        normalize = transforms.Normalize(mean, std)
        normalized_iamges = normalize(stacked_images)
        return normalized_iamges


class CLIPBinaryClassifier(nn.Module):
    """
    Binary classifier built on top of CLIP embeddings.

    This model uses a pre-trained CLIP model for feature extraction and adds
    a binary classification head for specific tasks.
    """

    def __init__(self, clip_model: nn.Module) -> None:
        """
        Initialize the binary classifier.

        Args:
            clip_model: Pre-trained CLIP model for feature extraction
        """
        super().__init__()
        self._classifier_score: Tensor
        self.clip_model = clip_model
        # Freeze CLIP parameters to prevent fine-tuning
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Binary classification head architecture
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1), nn.Sigmoid()
        )

        # Initialize weights using Kaiming initialization
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            images: Batch of preprocessed images

        Returns:
            torch.Tensor: Binary classification scores
        """
        # Extract features using CLIP
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images.to(images.device))

        # Normalize features and apply classification head
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        image_features = image_features.to(torch.float32)
        self._classifier_score = self.classifier(image_features)
        return self._classifier_score


class AestheticModel:
    """
    Model for predicting aesthetic scores of images using CLIP embeddings.

    This model uses a pre-trained CLIP model for feature extraction and a custom MLP
    for aesthetic score prediction.
    """

    def __init__(self, repo_id: str, filename: str) -> None:
        """
        Initialize the aesthetic scoring model.

        Args:
            MODEL_PATH_Aesthetics: Path to the pre-trained aesthetic model checkpoint
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        self._model = MLP.from_pretrained(repo_id, filename)
        self._model.to(self._device)
        self._model.eval()
        self._clip_model, self._preprocess = clip.load("ViT-L/14", device=self._device)

    def score(self, images: list[torch.Tensor]) -> np.ndarray:
        """
        Generate aesthetic scores for a batch of images.

        Args:
            images: List of image tensors to be scored

        Returns:
            numpy.ndarray: Aesthetic scores for each image
        """
        model_scores: list[float] = []
        for img in images:
            np_image = img.cpu().detach().numpy()
            pil_image = Image.fromarray(np_image, "RGB")
            image = self._preprocess(pil_image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                image_features = self._clip_model.encode_image(image)

            im_emb_arr = self.normalized(image_features.cpu().detach().numpy())
            img_score = self._model(torch.from_numpy(im_emb_arr).to(self._device).type(torch.cuda.FloatTensor))
            model_scores.append(img_score.cpu().detach().numpy())

        return self.custom_sigmoid_function(np.array(model_scores), 0.01, 0.0)

    @staticmethod
    def normalized(a: np.ndarray, axis: int = -1, order: int = 2) -> np.ndarray:
        """
        Normalize the input array using L2 normalization.

        Args:
            a: Input array to normalize
            axis: Axis along which to normalize
            order: Order of the normalization

        Returns:
            numpy.ndarray: Normalized array
        """
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        normalized: np.ndarray = a / np.expand_dims(l2, axis)
        return normalized

    @staticmethod
    def custom_sigmoid_function(x: np.ndarray, slope: float, x_shift: float) -> np.ndarray:
        """
        Apply a custom sigmoid function to remap input values.

        Args:
            x: Input values to remap
            slope: Controls the steepness of the sigmoid curve
            x_shift: Shifts the sigmoid curve along the x-axis

        Returns:
            numpy.ndarray: Remapped values
        """
        sigmoid: np.ndarray = 1.0 / (1.0 + np.exp(-slope * (x - x_shift)))
        return sigmoid


class MLP(pl.LightningModule):
    """
    Multi-Layer Perceptron model for aesthetic score prediction.

    A PyTorch Lightning module implementing a deep neural network for
    predicting aesthetic scores from image embeddings.
    """

    def __init__(self, xcol: str = "emb", ycol: str = "avg_rating") -> None:
        """
        Initialize the MLP model.

        Args:
            input_size: Dimension of input features
            xcol: Name of the input column in the dataset
            ycol: Name of the target column in the dataset
        """
        super().__init__()
        self.xcol = xcol
        self.ycol = ycol

        # Define network architecture
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> Any:
        """
        Forward pass of the MLP.

        Args:
            x: Input tensor of image embeddings

        Returns:
            torch.Tensor: Predicted aesthetic scores
        """
        return self.layers(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Training step for the MLP.

        Args:
            batch: Dictionary containing input and target values
            batch_idx: Index of the current batch

        Returns:
            torch.Tensor: Training loss
        """
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        return F.mse_loss(x_hat, y)

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Validation step for the MLP.

        Args:
            batch: Dictionary containing input and target values
            batch_idx: Index of the current batch

        Returns:
            torch.Tensor: Validation loss
        """
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        return F.mse_loss(x_hat, y)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer instance
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    @classmethod
    def from_pretrained(cls, repo_id: str, filename: str) -> "MLP":
        model = cls()  # Creates an instance using cls
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return model
