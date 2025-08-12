import gc
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from loguru import logger

from engine.utils.gs_data_checker_utils import sigmoid


class QualityClassifierModel:
    """
    A quality classifier model that uses DinoNet for image quality assessment.
    This model loads a pre-trained DinoNet model and uses it to predict image quality scores.
    """

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: nn.Module | None = None
        self._model_path = ""
        self._emb_dim = 256
        self._model_name = "dinov2_vits14"
        self._image_size = 518
        self._norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self._device).view(3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225], device=self._device).view(3, 1, 1)

    def load_model(self, repo_id: str, quality_scorer_model: str) -> None:
        """Function for loading DinoNet model

        Args:
            repo_id: Hugging Face repository ID
            quality_scorer_model: Name of the quality scorer model
        """
        # Use default DinoNet parameters
        if repo_id is None:
            raise ValueError("Repo ID is required")
        if quality_scorer_model is None:
            raise ValueError("Quality scorer model is required")

        # Load model weights
        backbone = torch.hub.load("facebookresearch/dinov2", self._model_name, pretrained=True)  # nosec B614
        model = DINOv2Net(backbone, emb_dim=self._emb_dim)
        self._model_path = hf_hub_download(
            repo_id=repo_id, revision="4438c19183d7b13f56cd9ce2ce08964bc072533b", filename=quality_scorer_model
        )
        self._model_state = torch.load(self._model_path, map_location=self._device, weights_only=True)  # nosec B614

        # Load full model weights
        model.load_state_dict(self._model_state)
        model.eval().to(self._device)
        self._model = model

        logger.info(f"DinoNet quality scorer loaded to device {self._device}")
        self._model.eval()

    def unload_model(self) -> None:
        """Function for unloading model"""

        if self._model is not None:
            del self._model
            self._model = None

        torch.cuda.empty_cache()
        gc.collect()

    def score(self, images: list[torch.Tensor]) -> torch.Tensor:
        """Function for generation of quality scores for a batch of images

        Args:
            images: List of torch tensors representing images

        Returns:
            torch.Tensor: Quality scores for each image (raw sigmoid outputs)
        """

        if self._model is None:
            raise RuntimeError("The model has not been loaded!")

        processed_images = self.preprocess_inputs(images)

        with torch.no_grad():
            scores = torch.empty(len(images), device=self._device)
            i = 0
            for img_tensor in processed_images:
                # Add batch dimension and move to device
                x = img_tensor.unsqueeze(0).to(self._device)

                # Get embedding and score from DinoNet
                _, score_logit = self._model(x)
                # Return raw sigmoid output
                score = sigmoid(score_logit)
                scores[i] = score
                i += 1

        return scores

    def preprocess_inputs(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        """Preprocess images for input to the DinoNet model

        Args:
            images: List of torch tensors in format (H, W, C) with values 0-255

        Returns:
            List of preprocessed torch tensors ready for DinoNet
        """
        processed_images = []

        for img_tensor in images:
            # Apply tensor transforms directly (avoid PIL conversion)
            processed_tensor = self._tensor_transform(img_tensor)
            processed_images.append(processed_tensor)

        return processed_images

    def _tensor_transform(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Efficient tensor-based transforms without PIL conversion"""
        # Convert (H,W,C) to (C,H,W) and normalize to [0,1]
        x = img_tensor.to(self._device).permute(2, 0, 1).float() / 255.0

        # Resize
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0),
            size=(self._image_size, self._image_size),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        ).squeeze(0)

        # Production-safe type narrowing
        if self._norm_mean is None or self._norm_std is None:
            raise RuntimeError("Normalization tensors not initialized. Model must be loaded first.")
        return (x - self._norm_mean) / self._norm_std


class DINOv2Net(nn.Module):
    """
    Wraps a frozen / finetuned DINOv2 backbone with:
      • an embedding head  (for metric learning / triplet loss)
      • a scoring head     (single logit for BCEWithLogitsLoss)
    """

    def __init__(self, backbone: Any, emb_dim: int = 256) -> None:
        super().__init__()
        self.backbone = backbone  # Vision transformer from DINOv2
        feat_dim = backbone.embed_dim  # 384, 768, 1024 … depending on variant

        # Two small heads
        self.emb_head = nn.Linear(feat_dim, emb_dim)
        self.score_head = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          • normalized embedding   (B, emb_dim)
          • raw score logits       (B, 1)
        """
        feats = self.backbone(x)  # (B, feat_dim)
        emb = nn.functional.normalize(self.emb_head(feats), p=2, dim=-1)  # L2-normalize
        score = self.score_head(feats).squeeze(1)  # (B,)
        return emb, score
