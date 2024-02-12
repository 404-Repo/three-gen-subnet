import base64
import io
import typing
import torch
from traceback import print_exception

import bittensor as bt
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config, GaussianDiffusion
from shap_e.models.transmitter.base import Transmitter
from shap_e.util.notebooks import decode_latent_mesh

from neurons import protocol


class TextTo3DModels:
    """
    Class wrapper for the components required to turn text prompts into 3D models.

    Attributes:
        transmitter: Transmitter object for processing latent representations.
        neural_model: A PyTorch neural network model for generating latents from text.
        diffusion_model: GaussianDiffusion object that handles the diffusion process.
    """

    def __init__(
        self,
        transmitter: Transmitter,
        neural_model: torch.nn.Module,
        diffusion_model: GaussianDiffusion,
    ):
        self.transmitter = transmitter
        self.neural_model = neural_model
        self.diffusion_model = diffusion_model


def forward(synapse: protocol.TextTo3D, models: TextTo3DModels) -> protocol.TextTo3D:
    """
    Processes the given task by converting the input text prompt to a 3D model.

    Args:
        synapse: Task object containing the text prompt.
        models: TextTo3DModels object containing the required model components.

    Returns:
        The updated task, after the text to 3D conversion process.
    """
    try:
        synapse.mesh_out = base64.b64encode(text_to_3d(synapse.prompt_in, models))
    except Exception as e:
        bt.logging.exception(f"Error during mining: {e}")
        bt.logging.debug(print_exception(type(e), e, e.__traceback__))
    return synapse


def load_models(device: torch.device, cache_dir: str) -> TextTo3DModels:
    """
    Load the neural network and diffusion models required to generate 3D models.

    Args:
        device: The torch.device object where the model should be loaded to.
        cache_dir: The directory where models are cached.

    Returns:
        TextTo3DModels object containing the initialized models.
    """
    model_cache_dir = f"{cache_dir}/shap_e_model_cache"

    transmitter_instance = load_model(
        "transmitter", device=device, cache_dir=model_cache_dir
    )
    text_to_3d_model = load_model("text300M", device=device, cache_dir=model_cache_dir)
    config = load_config("diffusion", cache_dir=model_cache_dir)
    diffusion_instance = diffusion_from_config(config)

    return TextTo3DModels(
        typing.cast(Transmitter, transmitter_instance),
        typing.cast(torch.nn.Module, text_to_3d_model),
        diffusion_instance,
    )


def text_to_3d(prompt: str, models: TextTo3DModels) -> bytes:
    """
    Converts the given text prompt into a 3D model representation in bytes.

    Args:
        prompt: The text prompt to be converted into a 3D model.
        models: TextTo3DModels object containing the required model components.

    Returns:
        A byte representation of the generated 3D mesh model.
    """
    batch_size = 1

    # Sample latents from the provided neural network model and diffusion process
    latents = sample_latents(
        batch_size=batch_size,
        model=models.neural_model,
        diffusion=models.diffusion_model,
        guidance_scale=15,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    # Convert the generated latent vectors into a 3D model
    # and serialize it into a binary string using a BytesIO buffer
    buffer = io.BytesIO()
    for i, latent in enumerate(latents):
        mesh = decode_latent_mesh(models.transmitter, latent).tri_mesh()

        # Reorder the vertex coordinates
        mesh.verts = mesh.verts[:, [0, 2, 1]]
        mesh.write_ply(buffer)

    return buffer.getvalue()
