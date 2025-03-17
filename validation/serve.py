import argparse
import gc
import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import time

import numpy as np
import pybase64
import pyspz
import torch
import uvicorn
import zstandard
from engine.data_structures import GaussianSplattingData, RequestData, ResponseData, ValidationResult
from engine.io.ply import PlyLoader
from engine.rendering.renderer import Renderer
from engine.utils.gs_data_checker_utils import is_input_data_valid
from engine.validation_engine import ValidationEngine
from fastapi import FastAPI
from loguru import logger
from PIL import Image


VERSION = "2.0.0"


def get_args() -> tuple[argparse.Namespace, list[str]]:
    """Function for handling input arguments related to running the server"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10006)
    args, extras = parser.parse_known_args()
    return args, extras


app = FastAPI()
args, _ = get_args()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Function for initializing all pipelines"""
    # Startup logic
    app.state.validator = ValidationEngine()
    app.state.validator.load_pipelines()
    app.state.zstd_decompressor = zstandard.ZstdDecompressor()
    app.state.renderer = Renderer()
    app.state.ply_data_loader = PlyLoader()
    gc.collect()
    torch.cuda.empty_cache()
    yield


app.router.lifespan_context = lifespan


def _prepare_input_data(assets: bytes) -> tuple[GaussianSplattingData | None, list[torch.Tensor]]:
    """Function for preparing input data for further processing"""

    # Loading input data
    t1 = time()
    pcl_buffer = io.BytesIO(assets)
    gs_data: GaussianSplattingData = app.state.ply_data_loader.from_buffer(pcl_buffer)
    t2 = time()
    logger.info(f"Loading data took: {t2 - t1} sec.")

    # Check required memory
    if not is_input_data_valid(gs_data):
        return None, []

    # Render images for validation
    gs_data_gpu = gs_data.send_to_device(app.state.validator.device)
    images = app.state.renderer.render_gs(gs_data_gpu, 16, 224, 224)
    t3 = time()
    logger.info(f"Image Rendering took: {t3 - t2} sec.")
    return gs_data_gpu, images


def _render_preview_image(
    gs_data: GaussianSplattingData, validation_score: float, preview_score_threshold: float
) -> str | None:
    """Function for rendering preview image of the input gs data"""
    if validation_score > preview_score_threshold:
        buffered = io.BytesIO()
        rendered_image = app.state.renderer.render_gs(gs_data, 1, 512, 512, [25.0], [-10.0])[0]
        preview_image = Image.fromarray(rendered_image.detach().cpu().numpy())
        preview_image.save(buffered, format="PNG")
        encoded_preview = pybase64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        encoded_preview = None
    return encoded_preview


def _validate_text_vs_image(
    prompt: str,
    images: list[torch.Tensor],
) -> ValidationResult:
    """Function for validation of the data that was generated using provided prompt"""

    # Validate input GS data by assessing rendered images
    t1 = time()
    val_res: ValidationResult = app.state.validator.validate_text_to_gs(prompt, images)
    logger.info(f" Score: {val_res.final_score}. Prompt: {prompt}")
    logger.info(f"Validation took: {time() - t1} sec.")
    return val_res


def _validate_image_vs_image(
    prompt_image: torch.Tensor,
    images: list[torch.Tensor],
) -> ValidationResult:
    """Function for validation of the data that was generated using prompt-image"""
    t1 = time()
    val_res: ValidationResult = app.state.validator.validate_image_to_gs(prompt_image, images)
    logger.info(f" Score: {val_res.final_score}. Prompt: provided image.")
    logger.info(f" Validation took: {time() - t1} sec.")
    return val_res


def _finalize_results(
    validation_results: ValidationResult,
    gs_data: GaussianSplattingData,
    generate_preview: bool,
    preview_score_threshold: float,
) -> ResponseData:
    """Function that finalize results"""
    if generate_preview:
        encoded_preview = _render_preview_image(gs_data, validation_results.final_score, preview_score_threshold)
    else:
        encoded_preview = None

    return ResponseData(
        score=validation_results.final_score,
        iqa=validation_results.combined_quality_score,
        alignment_score=validation_results.alignment_score,
        ssim=validation_results.ssim_score,
        lpips=validation_results.lpips_score,
        preview=encoded_preview,
    )


def _cleanup() -> None:
    """Function for cleaning up the memory"""
    t1 = time()
    torch.cuda.empty_cache()
    gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()
    logger.info(f"Cache purge took: {time() - t1} sec. VRAM Memory: {gpu_memory_free} / {gpu_memory_total}")


def decode_assets(request: RequestData, zstd_decomp: zstandard.ZstdDecompressor) -> bytes:
    t1 = time()
    assets = pybase64.b64decode(request.data, validate=True)
    t2 = time()
    logger.info(
        f"Assets decoded. Size: {len(request.data)} -> {len(assets)}. "
        f"Time taken: {t2 - t1:.2f} sec. Prompt: {request.prompt}."
    )

    if request.compression == 1:  # Experimental. Zstd compression.
        compressed_size = len(assets)
        assets = zstd_decomp.decompress(assets)
        logger.info(
            f"Decompressed. Size: {compressed_size} -> {len(assets)}. "
            f"Time taken: {time() - t2:.2f} sec. Prompt: {request.prompt}."
        )
    elif request.compression == 2:  # Experimental. SPZ compression.
        compressed_size = len(assets)
        assets = pyspz.decompress(assets, include_normals=False)
        logger.info(
            f"Decompressed. Size: {compressed_size} -> {len(assets)}. "
            f"Time taken: {time() - t2:.2f} sec. Prompt: {request.prompt}."
        )

    return assets


@app.get("/version/", response_model=str)
async def version() -> str:
    """
    Returns current endpoint version.
    """
    return str(VERSION)


@app.post("/validate_txt_to_3d_ply/", response_model=ResponseData)
async def validate_txt_to_3d_ply(request: RequestData) -> ResponseData:
    """
    Validates the input prompt and PLY data to produce scores.

    Parameters:
    - request (RequestData): An instance of RequestData containing the input prompt and data.

    Returns:
    - ResponseData: An instance of ResponseData containing the scores generated from the validation_lib process.

    """
    try:
        assets = decode_assets(request, zstd_decomp=app.state.zstd_decompressor)
        gs_data, gs_rendered_images = _prepare_input_data(assets)
        if gs_data and request.prompt is not None:
            validation_results = _validate_text_vs_image(request.prompt, gs_rendered_images)
            response = _finalize_results(
                validation_results, gs_data, request.generate_preview, request.preview_score_threshold
            )
        else:
            response = ResponseData(score=0.0)
    except Exception as e:
        logger.exception(e)
        response = ResponseData(score=0.0)
    finally:
        _cleanup()

    return response


@app.post("/validate_img_to_3d_ply/", response_model=ResponseData)
async def validate_img_to_3d_ply(request: RequestData) -> ResponseData:
    """
    Validates the input prompt and PLY data to produce scores.

    Parameters:
    - request (RequestData): An instance of RequestData containing the input prompt and data.

    Returns:
    - ResponseData: An instance of ResponseData containing the scores generated from the validation_lib process.

    """
    try:
        assets = decode_assets(request, zstd_decomp=app.state.zstd_decompressor)
        gs_data, gs_rendered_images = _prepare_input_data(assets)
        if gs_data and request.prompt_image:
            image_data = pybase64.b64decode(request.prompt_image)
            prompt_image = Image.open(io.BytesIO(image_data))
            torch_prompt_image = torch.tensor(np.asarray(prompt_image))
            validation_results = _validate_image_vs_image(torch_prompt_image, gs_rendered_images)
            response = _finalize_results(
                validation_results, gs_data, request.generate_preview, request.preview_score_threshold
            )
        else:
            response = ResponseData(score=0.0)
    except Exception as e:
        logger.exception(e)
        response = ResponseData(score=0.0)
    finally:
        _cleanup()

    return response


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, backlog=256)
