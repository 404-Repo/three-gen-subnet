import argparse
import base64
import gc
import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import time

import torch
import uvicorn
from application.metrics import Metrics
from fastapi import FastAPI
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from validation_lib.io.base import BaseLoader
from validation_lib.io.ply import PlyLoader
from validation_lib.memory import enough_gpu_mem_available
from validation_lib.rendering.rendering_pipeline import RenderingPipeline
from validation_lib.validation.validation_pipeline import ValidationPipeline, ValidationResult


VERSION = "1.11.0"
SHARPNESS_THRESHOLD = 620.0


def get_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10006)
    args, extras = parser.parse_known_args()
    return args, extras


app = FastAPI()
args, _ = get_args()


class RequestData(BaseModel):
    prompt: str = Field(max_length=1024, description="Prompt used to generate assets")
    data: str = Field(max_length=500 * 1024 * 1024, description="Generated assets")
    generate_preview: bool = Field(default=False, description="Optional. Pass to render and return a preview")
    preview_score_threshold: float = Field(default=0.8, description="Minimal score to return a preview")


class ResponseData(BaseModel):
    score: float = Field(default=0.0, description="Validation score, from 0.0 to 1.0")
    vqa: float = Field(default=0.0, description="VQA score")
    clip: float = Field(default=0.0, description="Metaclip similarity score")
    ssim: float = Field(default=0.0, description="Structure similarity score")
    lpips: float = Field(default=0.0, description="Perceptive similarity score")
    sharpness: float = Field(default=0.0, description="Laplacian variance (sharpness) score")
    preview: str | None = Field(default=None, description="Optional. Preview image, base64 encoded PNG")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # Startup logic
    app.state.validator = ValidationPipeline()
    app.state.validator.preload_model()
    app.state.metrics = Metrics()
    gc.collect()
    torch.cuda.empty_cache()

    yield

    gc.collect()
    torch.cuda.empty_cache()


app.router.lifespan_context = lifespan


def _validate(request: RequestData, loader: BaseLoader) -> ResponseData:
    logger.info(f"Validating started. Prompt: {request.prompt}")

    t1 = time()

    # Load data
    pcl_raw = base64.b64decode(request.data, validate=True)
    pcl_buffer = io.BytesIO(pcl_raw)
    data_dict = loader.from_buffer(pcl_buffer)
    t2 = time()

    logger.info(f"Loading data took: {t2 - t1} sec.")

    # Check required memory
    if not enough_gpu_mem_available(data_dict):
        return ResponseData(score=0.0)

    # Render images
    renderer = RenderingPipeline(16, mode="gs")
    images = renderer.render_gaussian_splatting_views(data_dict, 512, 512, 2.5)
    preview_image_input0 = renderer.render_preview_image(data_dict, 512, 512, 25.0, -10.0, cam_rad=2.5)
    preview_image_input1 = renderer.render_preview_image(data_dict, 512, 512, 0.0, 0.0, cam_rad=2.5)

    t3 = time()
    logger.info(f"Image Rendering took: {t3 - t2} sec.")

    # Validate images
    val_res: ValidationResult = app.state.validator.validate(
        [preview_image_input0, preview_image_input1], images, request.prompt
    )
    logger.info(f" Score: {val_res.final_score}. Prompt: {request.prompt}")

    t4 = time()
    logger.info(f"Validation took: {t4 - t3} sec. Total time: {t4 - t1} sec.")

    # Sharpness check
    if val_res.sharpness_score < SHARPNESS_THRESHOLD:
        logger.info(
            f"Sharpness score ({val_res.sharpness_score:.1f}) too low. Resetting the score. "
            f"Prompt: {request.prompt}"
        )
        val_res.final_score = 0.0

    if request.generate_preview and val_res.final_score > request.preview_score_threshold:
        buffered = io.BytesIO()
        rendered_image = renderer.render_preview_image(data_dict, 512, 512, 25.0, -10.0, cam_rad=2.5)
        preview_image = Image.fromarray(rendered_image.detach().cpu().numpy())
        preview_image.save(buffered, format="PNG")
        encoded_preview = base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        encoded_preview = None

    app.state.metrics.update(val_res.final_score)

    return ResponseData(
        score=val_res.final_score,
        vqa=val_res.vqa_score,
        clip=val_res.clip_score,
        ssim=val_res.ssim_score,
        lpips=val_res.lpips_score,
        sharpness=val_res.sharpness_score,
        preview=encoded_preview,
    )


def _cleanup() -> None:
    t1 = time()

    # gc.collect()
    torch.cuda.empty_cache()
    gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()

    t2 = time()
    logger.info(f"Cache purge took: {t2 - t1} sec. VRAM Memory: {gpu_memory_free} / {gpu_memory_total}")


@app.post("/validate_ply/", response_model=ResponseData)
async def validate_ply(request: RequestData) -> ResponseData:
    """
    Validates the input prompt and PLY data to produce scores.

    Parameters:
    - request (RequestData): An instance of RequestData containing the input prompt and data.

    Returns:
    - ResponseData: An instance of ResponseData containing the scores generated from the validation_lib process.

    """
    try:
        response = _validate(request, PlyLoader())
    except Exception as e:
        logger.exception(e)
        response = ResponseData(score=0.0)
    finally:
        _cleanup()

    return response


@app.get("/version/", response_model=str)
async def version() -> str:
    """
    Returns current endpoint version.
    """
    return str(VERSION)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, backlog=256)
