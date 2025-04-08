import gc
from typing import Any

import clip
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.model import CLIP
from huggingface_hub import hf_hub_download
from loguru import logger
from torchvision import transforms
from torchvision.transforms import InterpolationMode as im


class AestheticsPredictorModel:
    """
    Model for predicting aesthetic scores of images using CLIP embeddings.
    This model uses a pre-trained CLIP model for feature extraction and a custom MLP
    for aesthetic score prediction.
    """

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: MLP | None = None
        self._clip_model: CLIP | None = None
        self._imgsize = 224
        self._preprocess = transforms.Compose(
            [
                transforms.Resize(self._imgsize, interpolation=im.BICUBIC),
                transforms.CenterCrop(self._imgsize),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

    def load_model(self, repo_id: str, filename: str, clip_model: str = "ViT-L/14") -> None:
        """Function for loading models"""

        self._clip_model, _ = clip.load(clip_model, device=self._device)
        self._model = MLP.from_pretrained(repo_id, filename)
        self._model.to(self._device)
        self._model.eval()
        logger.info(f"Aesthetic Predictor loaded to device {self._device}")

    def unload_model(self) -> None:
        """Function for unloading model"""

        del self._clip_model
        del self._model
        self._model = None
        self._clip_model = None

        torch.cuda.empty_cache()
        gc.collect()

    def score(self, images: list[torch.Tensor]) -> np.ndarray:
        """Function for generation of the aesthetic scores for a batch of images"""

        if self._clip_model is None:
            raise RuntimeError("The clip model has not been loaded.")
        elif self._model is None:
            raise RuntimeError("The aesthetic predictor model has not been loaded.")

        model_scores: torch.Tensor = torch.zeros(len(images))

        for i, img in enumerate(images):
            image = img.to(torch.float16) / 255.0
            image = image.permute(2, 0, 1)
            image = self._preprocess(image)

            with torch.no_grad():
                image_features = self._clip_model.encode_image(image.unsqueeze(0))

            im_emb_arr = self.normalized(image_features).squeeze()
            img_score = self._model(im_emb_arr.to(torch.float32))
            model_scores[i] = img_score

        score = self.custom_sigmoid(model_scores, 0.01, 0.0)
        out: np.ndarray = score.cpu().detach().numpy()
        return out

    @staticmethod
    def normalized(a: torch.Tensor, axis: int = -1, order: int = 2) -> torch.Tensor:
        """Normalize the input tensor using L2 normalization"""

        l2 = torch.norm(a, p=order, dim=axis, keepdim=True)
        l2 = torch.where(l2 == 0, torch.tensor(1.0, device=a.device, dtype=a.dtype), l2)
        return torch.Tensor(a / l2)

    @staticmethod
    def custom_sigmoid(x: torch.Tensor, slope: float = 1.0, shift: float = 0.0) -> torch.Tensor:
        """Custom sigmoid function with adjustable slope and shift"""

        return torch.Tensor(1 / (1 + torch.exp(-slope * (x - shift))))


class MLP(pl.LightningModule):
    """
    Multi-Layer Perceptron model for aesthetic score prediction.
    A PyTorch Lightning module implementing a deep neural network for
    predicting aesthetic scores from image embeddings.
    """

    def __init__(self, xcol: str = "emb", ycol: str = "avg_rating") -> None:
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
        """Forward pass of the MLP"""

        return self.layers(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step for the MLP"""

        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        return F.mse_loss(x_hat, y)

    def validation_step(self, batch: dict, batch_idx: int) -> Any:
        """Validation step for the MLP"""

        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        return F.mse_loss(x_hat, y)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training"""

        return torch.optim.Adam(self.parameters(), lr=1e-3)

    @classmethod
    def from_pretrained(cls, repo_id: str, filename: str) -> "MLP":
        """Function for loading the model checkpoint"""

        model = cls()  # Creates an instance using cls
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return model
