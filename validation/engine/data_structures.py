import torch
from pydantic import BaseModel, ConfigDict, Field, AliasChoices


class GaussianSplattingData(BaseModel):
    points: torch.Tensor  # gaussian centroids
    normals: torch.Tensor  # normals if provided
    features_dc: torch.Tensor  # colour information stored as RGB values
    features_rest: torch.Tensor  # optional attribute field, not being in use
    opacities: torch.Tensor  # opacity value for every gaussian splat presented
    scales: torch.Tensor  # scale values per gaussian splat
    rotations: torch.Tensor  # rotation quaternion for every gaussian splat
    sh_degree: torch.Tensor  # degree of the spherical harmonics

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow PyTorch tensors

    def send_to_device(self, device: torch.device) -> "GaussianSplattingData":
        """Moves all tensors in the instance to the specified device."""
        return GaussianSplattingData(
            points=self.points.to(device),
            normals=self.normals.to(device),
            features_dc=self.features_dc.to(device),
            features_rest=self.features_rest.to(device),
            opacities=self.opacities.to(device),
            scales=self.scales.to(device),
            rotations=self.rotations.to(device),
            sh_degree=self.sh_degree.to(device),
        )


class ValidationRequest(BaseModel):
    prompt: str | None = Field(
        default=None, max_length=1024, description="Text prompt used to generate the 3D assets or images"
    )
    prompt_image: str | None = Field(
        default=None, description="Base64-encoded image data for 3D model generation"
    )

    data: str = Field(
        max_length=500 * 1024 * 1024,
        description="Generated 3D asset data (mesh, textures, etc.) in base64 or binary format",
    )
    compression: int = Field(
        default=0, description="Asset compression level: 0 for uncompressed data, 2 for SPZ compressed format"
    )

    generate_single_preview: bool = Field(
        default=False,
        validation_alias=AliasChoices('generate_preview', 'generate_single_preview'),
        description="Generate a single front-facing render preview of the 3D asset",
    )
    generate_grid_preview: bool = Field(
        default=False,
        description="Generate a 2x2 grid preview showing multiple viewing angles (front, side, top, perspective)",
    )
    preview_score_threshold: float = Field(
        default=0.6, description="Minimum validation score required to generate and return preview images (0.0-1.0)"
    )


class ValidationResponse(BaseModel):
    score: float = Field(default=0.0, description="Final validation score. Combines all Metrics, 0.0-1.0")
    iqa: float = Field(default=0.0, description="Image Quality Assessment score")
    alignment: float = Field(
        default=0.0, serialization_alias="alignment_score", description="Semantic alignment score."
    )
    ssim: float = Field(default=0.0, description="Structural Similarity Index (SSIM).")
    lpips: float = Field(default=0.0, description="Learned Perceptual Image Patch Similarity (LPIPS).")

    preview: str | None = Field(default=None, description="Base64-encoded PNG of single front-facing render")
    grid_preview: str | None = Field(
        default=None, description="Base64-encoded PNG of 2x2 grid showing multiple angles/views"
    )


class RenderRequest(BaseModel):
    prompt: str | None = Field(
        default=None, max_length=1024, description="Text prompt used to generate the 3D assets or images"
    )
    data: str = Field(
        max_length=500 * 1024 * 1024,
        description="Generated 3D asset data (mesh, textures, etc.) in base64 or binary format",
    )
    compression: int = Field(
        default=0, description="Asset compression level: 0 for uncompressed data, 2 for SPZ compressed format"
    )


class TimeStat(BaseModel):
    loading_data_time: float = Field(default=0.0, description="Loading data time")
    image_rendering_time: float = Field(default=0.0, description="Image rendering time")
    validation_time: float = Field(default=0.0, description="Validation time")
    total_time: float = Field(default=0.0, description="Total time of server processing")
