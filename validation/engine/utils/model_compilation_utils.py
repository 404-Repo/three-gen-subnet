"""
GPU-adaptive compilation utilities for PyTorch models.

This module provides functions to detect GPU capabilities and configure
torch.compile settings optimally for different hardware configurations.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from loguru import logger


def configure_torch_for_gpu() -> str:
    """Detect GPU and configure Triton/PyTorch settings adaptively

    Returns:
        str: Recommended torch.compile mode based on GPU capabilities
    """
    if not torch.cuda.is_available():
        return "default"

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    logger.info(f"Detected GPU: {gpu_name} ({gpu_memory_gb:.1f}GB)")

    # GPU-specific configurations based on shared memory capabilities
    gpu_configs = {
        "L40": {"shared_mem_kb": 164, "compile_mode": "max-autotune"},
        "L40S": {"shared_mem_kb": 164, "compile_mode": "max-autotune"},
        "RTX A6000": {"shared_mem_kb": 100, "compile_mode": "reduce-overhead"},
        "RTX 6000 Ada": {"shared_mem_kb": 100, "compile_mode": "reduce-overhead"},
        "RTX 4090": {"shared_mem_kb": 100, "compile_mode": "reduce-overhead"},
    }

    # Detect GPU type
    detected_gpu = None
    for gpu_type in gpu_configs.keys():
        if gpu_type.replace(" ", "").lower() in gpu_name.replace(" ", "").lower():
            detected_gpu = gpu_type
            break

    if detected_gpu:
        config = gpu_configs[detected_gpu]

        # Configure Triton based on GPU capabilities
        shared_mem_kb = config["shared_mem_kb"]
        if isinstance(shared_mem_kb, int | float) and shared_mem_kb < 120:  # Conservative threshold
            # Reduce block sizes for lower shared memory GPUs
            try:
                # Try different ways to access Triton config
                inductor_config = getattr(torch, "_inductor", None)
                if inductor_config and hasattr(inductor_config, "config"):
                    config_obj = inductor_config.config
                    if hasattr(config_obj, "triton"):
                        triton_config = config_obj.triton
                        if hasattr(triton_config, "max_block_size"):
                            triton_config.max_block_size = 64
                        if hasattr(triton_config, "max_num_stages"):
                            triton_config.max_num_stages = 2
                elif hasattr(torch._inductor, "config"):
                    # Alternative config paths for different PyTorch versions
                    if hasattr(config_obj, "max_autotune_gemm_search_space"):
                        config_obj.max_autotune_gemm_search_space = 4
                    if hasattr(config_obj, "coordinate_descent_tuning"):
                        config_obj.coordinate_descent_tuning = True
                logger.info(f"Configured conservative Triton settings for {detected_gpu}")
            except AttributeError as e:
                logger.warning(f"Could not configure Triton settings: {e}. Using compile mode only.")

            compile_mode = "reduce-overhead"  # Less aggressive optimization
        else:
            # Higher-end GPUs can handle more aggressive settings
            compile_mode = str(config["compile_mode"])
            logger.info(f"Using optimized settings for {detected_gpu}")

        return compile_mode

    # Fallback for unknown GPUs - use conservative settings
    logger.warning(f"Unknown GPU type: {gpu_name}. Using conservative settings.")
    try:
        inductor_config = getattr(torch, "_inductor", None)
        if inductor_config and hasattr(inductor_config, "config"):
            config_obj = inductor_config.config
            if hasattr(config_obj, "triton"):
                triton_config = config_obj.triton
                if hasattr(triton_config, "max_block_size"):
                    triton_config.max_block_size = 64
                if hasattr(triton_config, "max_num_stages"):
                    triton_config.max_num_stages = 2
    except AttributeError:
        logger.warning("Could not configure Triton settings. Using compile mode only.")

    return "reduce-overhead"


def safe_compile_model(model: nn.Module, preferred_mode: str) -> nn.Module | Callable[..., Any]:
    """Try compilation with fallback strategies

    Args:
        model: The model to compile
        preferred_mode: Preferred compilation mode based on GPU detection

    Returns:
        Compiled model or original model if compilation fails
    """
    strategies: list[dict[str, str | None]] = [
        {"mode": preferred_mode, "desc": f"GPU-optimized ({preferred_mode})"},
        {"mode": "reduce-overhead", "desc": "Balanced optimization"},
        {"mode": "default", "desc": "Conservative optimization"},
        {"mode": None, "desc": "No compilation (eager mode)"},
    ]

    for strategy in strategies:
        try:
            mode = strategy["mode"]
            desc = strategy["desc"]
            if mode:
                compiled_model = torch.compile(model, mode=mode, dynamic=False, fullgraph=False)
                logger.info(f"Successfully compiled with {desc}")
                return compiled_model
            else:
                logger.warning("Using eager mode - no compilation")
                return model
        except Exception as e:
            desc = strategy["desc"]
            logger.warning(f"Compilation failed with {desc}: {e}")
            continue

    return model  # Fallback to original model
