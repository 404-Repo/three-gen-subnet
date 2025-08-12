import argparse
import asyncio
import gc
import io
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from time import time

import numpy as np
import pybase64
import pyspz
import torch
import uvicorn
from engine.data_structures import (
    GaussianSplattingData,
    RenderRequest,
    TimeStat,
    ValidationRequest,
    ValidationResponse,
    ValidationResult,
)
from engine.io.ply import PlyLoader
from engine.rendering.renderer import Renderer
from engine.utils.gs_data_checker_utils import is_input_data_valid
from engine.validation_engine import ValidationEngine
from fastapi import FastAPI, HTTPException
from loguru import logger
from PIL import Image
from starlette.responses import StreamingResponse


VERSION = "2.2.0"


def get_args() -> tuple[argparse.Namespace, list[str]]:
    """Function for handling input arguments related to running the server"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10006)
    args, extras = parser.parse_known_args()
    return args, extras


app = FastAPI()
args, _ = get_args()

# Set max_workers=1 to serialize GPU requests and prevent inefficient parallel execution.
# GPU workloads don't parallelize well - multiple concurrent requests compete for the same
# GPU resources, causing each request to take longer (2 requests = 2x total time).
# By queuing requests sequentially, we ensure the first request completes as quickly as
# possible. This is especially important for miners with fixed cooldowns, as serialized
# processing spreads requests evenly over time rather than bunching completions together.
executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Function for initializing all pipelines"""
    # Startup logic
    app.state.validator = ValidationEngine()
    app.state.validator.load_pipelines()
    app.state.renderer = Renderer()
    app.state.ply_data_loader = PlyLoader()
    gc.collect()
    torch.cuda.empty_cache()
    yield


app.router.lifespan_context = lifespan


def _decode_assets(request: ValidationRequest | RenderRequest) -> bytes:
    t1 = time()
    assets = pybase64.b64decode(request.data, validate=True)
    t2 = time()
    logger.info(
        f"Assets decoded. Size: {len(request.data)} -> {len(assets)}. "
        f"Time taken: {t2 - t1:.2f} sec. Prompt: {request.prompt}."
    )

    if request.compression == 2:  # SPZ compression.
        compressed_size = len(assets)
        assets = pyspz.decompress(assets, include_normals=False)
        logger.info(
            f"Decompressed. Size: {compressed_size} -> {len(assets)}. "
            f"Time taken: {time() - t2:.2f} sec. Prompt: {request.prompt}."
        )

    return assets


def prepare_input_data(
    assets: bytes,
    renderer: Renderer,
    ply_data_loader: PlyLoader,
    validator: ValidationEngine,
    render_views_number: int = 16,
    render_img_width: int = 518,
    render_img_height: int = 518,
    render_theta_angles: list[float] | None = None,
) -> tuple[GaussianSplattingData | None, list[torch.Tensor], TimeStat]:
    """Function for preparing input data for further processing"""

    time_stat = TimeStat()

    # Loading input data
    t1 = time()
    pcl_buffer = io.BytesIO(assets)
    gs_data: GaussianSplattingData = ply_data_loader.from_buffer(pcl_buffer)
    t2 = time()
    time_stat.loading_data_time = t2 - t1
    logger.info(f"Loading data took: {time_stat.loading_data_time} sec.")

    # Check required memory
    if not is_input_data_valid(gs_data):
        return None, [], time_stat

    # Render images for validation
    gs_data_gpu = gs_data.send_to_device(validator.device)
    images = renderer.render_gs(
        gs_data_gpu,
        views_number=render_views_number,
        img_width=render_img_width,
        img_height=render_img_height,
        theta_angles=render_theta_angles,
    )
    t3 = time()
    time_stat.image_rendering_time = t3 - t2
    logger.info(f"Image Rendering took: {time_stat.image_rendering_time} sec.")
    return gs_data_gpu, images, time_stat


def _render_preview_image(
    gs_data: GaussianSplattingData, validation_score: float, preview_score_threshold: float, renderer: Renderer
) -> str | None:
    """Function for rendering preview image of the input gs data"""
    if validation_score > preview_score_threshold:
        buffered = io.BytesIO()
        rendered_image = renderer.render_gs(gs_data, 1, 512, 512, [25.0], [-10.0])[0]
        preview_image = Image.fromarray(rendered_image.detach().cpu().numpy())
        preview_image.save(buffered, format="PNG")
        encoded_preview = pybase64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        encoded_preview = None
    return encoded_preview


def _validate_text_vs_image(
    prompt: str,
    images: list[torch.Tensor],
    validator: ValidationEngine,
) -> ValidationResult:
    """Function for validation of the data that was generated using provided prompt"""

    t1 = time()
    val_res: ValidationResult = validator.validate_text_to_gs(prompt, images)
    logger.info(f"Score: {val_res.final_score}. Prompt: {prompt}")
    logger.info(f"Validation took: {time() - t1:6f} sec.")
    return val_res


def _validate_image_vs_image(
    prompt_image: torch.Tensor,
    images: list[torch.Tensor],
    validator: ValidationEngine,
) -> ValidationResult:
    """Function for validation of the data that was generated using prompt-image"""

    t1 = time()
    val_res: ValidationResult = validator.validate_image_to_gs(prompt_image, images)
    logger.info(f" Score: {val_res.final_score}. Prompt: provided image.")
    logger.info(f" Validation took: {time() - t1} sec.")
    return val_res


def _finalize_results(
    validation_results: ValidationResult,
    gs_data: GaussianSplattingData,
    generate_preview: bool,
    preview_score_threshold: float,
    renderer: Renderer,
) -> ValidationResponse:
    """Function that finalizes results"""
    if generate_preview:
        encoded_preview = _render_preview_image(
            gs_data, validation_results.final_score, preview_score_threshold, renderer
        )
    else:
        encoded_preview = None

    return ValidationResponse(
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


def decode_and_validate_txt(
    request: ValidationRequest,
    ply_data_loader: PlyLoader,
    renderer: Renderer,
    validator: ValidationEngine,
) -> tuple[ValidationResponse, TimeStat]:
    t1 = time()
    assets = _decode_assets(request)
    gs_data, gs_rendered_images, time_stat = prepare_input_data(
        assets,
        renderer,
        ply_data_loader,
        validator,
        render_views_number=16,
        render_img_width=518,
        render_img_height=518,
    )

    if gs_data is not None and request.prompt is not None:
        t2 = time()
        validation_result = _validate_text_vs_image(request.prompt, gs_rendered_images, validator)
        time_stat.validation_time = time() - t2

        response = _finalize_results(
            validation_result,
            gs_data,
            request.generate_preview,
            request.preview_score_threshold,
            renderer,
        )
        time_stat.total_time = time() - t1
    else:
        response = ValidationResponse(score=0.0)
    return response, time_stat


@app.post("/validate_txt_to_3d_ply/", response_model=ValidationResponse)
async def validate_txt_to_3d_ply(request: ValidationRequest) -> ValidationResponse:
    """
    Validates the input prompt and PLY data to produce scores.
    """
    try:
        loop = asyncio.get_running_loop()
        response, time_stat = await loop.run_in_executor(
            executor,
            decode_and_validate_txt,
            request,
            app.state.ply_data_loader,
            app.state.renderer,
            app.state.validator,
        )
    except Exception as e:
        logger.exception(e)
        response = ValidationResponse(score=0.0)
    finally:
        _cleanup()

    return response


def decode_and_validate_img(
    request: ValidationRequest,
    ply_data_loader: PlyLoader,
    renderer: Renderer,
    validator: ValidationEngine,
) -> tuple[ValidationResponse, TimeStat]:
    t1 = time()
    assets = _decode_assets(request)
    gs_data, gs_rendered_images, time_stat = prepare_input_data(assets, renderer, ply_data_loader, validator)

    if gs_data is not None and request.prompt_image:
        t2 = time()
        image_data = pybase64.b64decode(request.prompt_image)
        prompt_image = Image.open(io.BytesIO(image_data))
        torch_prompt_image = torch.tensor(np.asarray(prompt_image))
        validation_results = _validate_image_vs_image(torch_prompt_image, gs_rendered_images, validator)
        time_stat.validation_time = t2 - time()

        response = _finalize_results(
            validation_results,
            gs_data,
            request.generate_preview,
            request.preview_score_threshold,
            renderer,
        )
        time_stat.total_time = time() - t1
    else:
        response = ValidationResponse(score=0.0)
    return response, time_stat


@app.post("/validate_img_to_3d_ply/", response_model=ValidationResponse)
async def validate_img_to_3d_ply(request: ValidationRequest) -> ValidationResponse:
    """
    Validates the input prompt and PLY data to produce scores.
    """
    try:
        loop = asyncio.get_running_loop()
        response, time_stat = await loop.run_in_executor(
            executor,
            decode_and_validate_img,
            request,
            app.state.ply_data_loader,
            app.state.renderer,
            app.state.validator,
        )
    except Exception as e:
        logger.exception(e)
        response = ValidationResponse(score=0.0)
    finally:
        _cleanup()
    return response


def combine_images4(
    images: list[torch.Tensor], img_width: int, img_height: int, gap: int, resize_factor: float
) -> Image.Image:
    row_width = img_width * 2 + gap
    column_height = img_height * 2 + gap

    combined_image = Image.new("RGB", (row_width, column_height), color="black")

    pil_images = [Image.fromarray(img.detach().cpu().numpy()) for img in images]

    combined_image.paste(pil_images[0], (0, 0))
    combined_image.paste(pil_images[1], (img_width + gap, 0))
    combined_image.paste(pil_images[2], (0, img_height + gap))
    combined_image.paste(pil_images[3], (img_width + gap, img_height + gap))

    w, h = combined_image.size
    if resize_factor != 1.0:
        combined_image = combined_image.resize(
            (int(w * resize_factor), int(h * resize_factor)), Image.Resampling.LANCZOS
        )

    return combined_image


@app.post("/render_duel_view/")
async def render_duel_view(request: RenderRequest) -> StreamingResponse:
    try:
        assets = _decode_assets(request)
        loop = asyncio.get_running_loop()
        gs_data, gs_rendered_images, _ = await loop.run_in_executor(
            executor,
            prepare_input_data,
            assets,
            app.state.renderer,
            app.state.ply_data_loader,
            app.state.validator,
            4,
            512,
            512,
            [20.0, 120.0, 220.0, 310.0],
        )
        if not gs_data:
            raise RuntimeError("Invalid splats data")

        if len(gs_rendered_images) != 4:
            raise RuntimeError(f"Failed to generate 4 views, only {len(gs_rendered_images)} views are generated")

        final_image = combine_images4(
            images=gs_rendered_images, img_width=512, img_height=512, gap=5, resize_factor=1.0
        )

        img_byte_arr = io.BytesIO()
        final_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        _cleanup()


@app.get("/version/", response_model=str)
async def version() -> str:
    """
    Returns current endpoint version.
    """
    return str(VERSION)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, backlog=256)
