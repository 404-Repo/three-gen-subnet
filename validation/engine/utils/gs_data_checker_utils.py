import sys
from typing import Any

import torch
from loguru import logger
from PIL import Image

from engine.data_structures import GaussianSplattingData


def sigmoid(x: torch.Tensor, slope: float = 1.0, x_shift: float = 0.0) -> Any:
    """Function for remapping input data using sigmoid function"""

    return 1.0 / (1.0 + torch.exp(-slope * (x - x_shift)))


def enough_gpu_mem_available(gs_data: GaussianSplattingData, verbose: bool = True) -> bool:
    """Function that checks whether the input data will fit in the GPU Memory"""

    _, gpu_memory_total = torch.cuda.mem_get_info()
    gpu_available_memory = gpu_memory_total - torch.cuda.memory_allocated()

    total_memory_bytes = 0
    total_memory_bytes += sys.getsizeof(gs_data.points)
    total_memory_bytes += sys.getsizeof(gs_data.scales)
    total_memory_bytes += sys.getsizeof(gs_data.normals)
    total_memory_bytes += sys.getsizeof(gs_data.rotations)
    total_memory_bytes += sys.getsizeof(gs_data.features_dc)
    total_memory_bytes += sys.getsizeof(gs_data.features_rest)
    total_memory_bytes += sys.getsizeof(gs_data.opacities)
    total_memory_bytes += sys.getsizeof(gs_data.sh_degree)

    if total_memory_bytes < gpu_available_memory:
        if verbose:
            logger.info(f" Total VRAM available: {gpu_available_memory / 1024 ** 3} Gb")
            logger.info(f" Total VRAM allocated: {torch.cuda.memory_allocated() / 1024 ** 3} Gb")
            logger.info(f" Total data size to load to VRAM: {total_memory_bytes / 1024 ** 3} Gb")
        return True

    if verbose:
        logger.warning(f" Total VRAM available: {gpu_available_memory / 1024 ** 3} Gb")
        logger.warning(f" Total VRAM allocated: {torch.cuda.memory_allocated()/1024 ** 3} Gb")
        logger.warning(f" Total data size to load to VRAM: {total_memory_bytes / 1024 ** 3} Gb")
        logger.warning(" Input data size exceeds the available VRAM free memory!")
        logger.warning(" Input data will not be further processed.\n")

    return False


def is_input_data_valid(gs_data: GaussianSplattingData, verbose: bool = True) -> bool:
    """Function that checks if the input data valid according to specified constraints"""

    if not enough_gpu_mem_available(gs_data, verbose):
        return False

    means3d_size = gs_data.points.shape
    if means3d_size[0] < 7000:
        return False

    centroids = gs_data.points
    centroids_checker = torch.ones((centroids.shape[0], 1))
    for i in range(3):
        if torch.allclose(gs_data.points[i], centroids_checker):
            return False

    zero_opacity_epsilon = 1e-3
    zero_opacity_count = torch.sum(gs_data.opacities < zero_opacity_epsilon).item()
    total_opacity_count = gs_data.opacities.shape[0]
    zero_opacity_percentage = 100 * zero_opacity_count / total_opacity_count
    if zero_opacity_percentage > 80:
        return False

    rotation_checker = torch.tensor([1, 0, 0, 0], dtype=gs_data.rotations.dtype).to(gs_data.rotations.device)
    if torch.allclose(gs_data.rotations, rotation_checker):
        return False

    zero_scales_epsilon = 0.05
    zero_scales_count = torch.sum(torch.all(gs_data.scales < zero_scales_epsilon)).item()
    total_scales_count = gs_data.scales.shape[0]
    zero_scales_percentage = 100 * zero_scales_count / total_scales_count

    if zero_scales_percentage > 80:
        return False

    return True


def add_white_background(image: Image.Image) -> Image.Image:
    # Create white background with same size
    white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))

    # Composite the image onto white background
    return Image.alpha_composite(white_bg, image.convert("RGBA"))
