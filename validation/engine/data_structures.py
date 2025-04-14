import torch
from pydantic import BaseModel, ConfigDict, Field


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


class ValidationResult(BaseModel):
    final_score: float  # combined score
    combined_quality_score: float  # (non-normalized) combined models predictor - score
    alignment_score: float  # clip similarity scores
    ssim_score: float  # structure similarity index score
    lpips_score: float  # perceptive similarity score
    validation_time: float | None = None  # time that validation took


class RequestData(BaseModel):
    prompt: str | None = Field(default=None, max_length=1024, description="Prompt used to generate assets")
    prompt_image: str | None = Field(default=None, description="Prompt-image used to generate assets")
    data: str = Field(max_length=500 * 1024 * 1024, description="Generated assets")
    compression: int = Field(default=0, description="Experimental feature")
    generate_preview: bool = Field(default=False, description="Optional. Pass to render and return a preview")
    preview_score_threshold: float = Field(default=0.8, description="Minimal score to return a preview")


class ResponseData(BaseModel):
    score: float = Field(default=0.0, description="Validation score, from 0.0 to 1.0")
    iqa: float = Field(default=0.0, description="Aesthetic Predictor (quality) score")
    alignment_score: float = Field(
        default=0.0, description="prompt vs rendered images or prompt-image vs rendered images score."
    )
    ssim: float = Field(default=0.0, description="Structure similarity score")
    lpips: float = Field(default=0.0, description="Perceptive similarity score")
    preview: str | None = Field(default=None, description="Optional. Preview image, base64 encoded PNG")


class TimeStat(BaseModel):
    loading_data_time: float = Field(default=0.0, description="Loading data time")
    image_rendering_time: float = Field(default=0.0, description="Image rendering time")
    validation_time: float = Field(default=0.0, description="Validation time")
    total_time: float = Field(default=0.0, description="Total time of server processing")


class ValidationResultData(BaseModel):
    response_data: ResponseData  # Response data with validation result
    time_stat: TimeStat | None = None  # Performance statistics of the validation
