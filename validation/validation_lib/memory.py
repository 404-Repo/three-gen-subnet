import sys

import torch
from loguru import logger


def enough_gpu_mem_available(data: dict) -> bool:
    """Function that checks whether the input data will fit in the GPU Memory

    Parameters
    ----------
    data_dict: raw input data that was received by the validator
    gpu_available_memory: the amount of memory that is currently available in the GPU

    Returns
    -------
    True - if the input data fits in the GPU memory
    False - otherwise
    """
    _, gpu_memory_total = torch.cuda.mem_get_info()
    gpu_available_memory = gpu_memory_total - torch.cuda.memory_allocated()

    total_memory_bytes = 0
    for _, d in data.items():
        total_memory_bytes += sys.getsizeof(d)

    if total_memory_bytes < gpu_available_memory:
        logger.info(f" Total VRAM available: {gpu_available_memory / 1024 ** 3} Gb")
        logger.info(f" Total VRAM allocated: {torch.cuda.memory_allocated() / 1024 ** 3} Gb")
        logger.info(f" Total data size to load to VRAM: {total_memory_bytes / 1024 ** 3} Gb")
        return True

    logger.warning(f" Total VRAM available: {gpu_available_memory / 1024 ** 3} Gb")
    logger.warning(f" Total VRAM allocated: {torch.cuda.memory_allocated()/1024 ** 3} Gb")
    logger.warning(f" Total data size to load to VRAM: {total_memory_bytes / 1024 ** 3} Gb")
    logger.warning(" Input data size exceeds the available VRAM free memory!")
    logger.warning(" Input data will not be further processed.\n")

    return False
