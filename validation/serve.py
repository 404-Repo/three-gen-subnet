import argparse
import base64
import gc
import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field
from validation_lib.io.base import BaseLoader
from validation_lib.io.hdf5 import HDF5Loader
from validation_lib.io.ply import PlyLoader
from validation_lib.memory import enough_gpu_mem_available
from validation_lib.rendering.rendering_pipeline import RenderingPipeline
from validation_lib.validation.validation_pipeline import ValidationPipeline


VERSION = "1.7.1"


def get_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10006)
    args, extras = parser.parse_known_args()
    return args, extras


app = FastAPI()
args, _ = get_args()


class RequestData(BaseModel):
    prompt: str = Field(max_length=1024)
    data: str = Field(max_length=500 * 1024 * 1024)
    data_ver: int = Field(default=0)
    # 0 - Dream Gaussian native format (default value)
    # 1+ - Preparation for PLY support


class ResponseData(BaseModel):
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # Startup logic
    app.state.validator = ValidationPipeline()
    app.state.validator.preload_model()

    yield

    gc.collect()
    torch.cuda.empty_cache()


app.router.lifespan_context = lifespan


def _validate(prompt: str, data: str, data_ver: int, loader: BaseLoader) -> Any:
    logger.info(f"Validating started. Prompt: {prompt}")

    t1 = time()

    # Load data
    pcl_raw = base64.b64decode(data)
    pcl_buffer = io.BytesIO(pcl_raw)
    data_dict = loader.from_buffer(pcl_buffer)
    t2 = time()

    logger.info(f"Loading data took: {t2 - t1} sec.")

    # Check required memory
    if not enough_gpu_mem_available(data_dict):
        return 0.0

    # Render images
    renderer = RenderingPipeline(16, mode="gs")
    images = renderer.render_gaussian_splatting_views(data_dict, 512, 512, 3.5, data_ver=data_ver)

    t3 = time()
    logger.info(f"Image Rendering took: {t3 - t2} sec.")

    # Validate images
    score = app.state.validator.validate(images, prompt)
    logger.info(f" Score: {score}. Prompt: {prompt}")

    t4 = time()
    logger.info(f"Validation took: {t4 - t3} sec. Total time: {t4 - t1} sec.")

    return score


def _cleanup() -> None:
    t1 = time()

    # gc.collect()
    torch.cuda.empty_cache()
    gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()

    t2 = time()
    logger.info(f"Cache purge took: {t2 - t1} sec. VRAM Memory: {gpu_memory_free} / {gpu_memory_total}")


@app.post("/validate/", response_model=ResponseData)
async def validate(request: RequestData) -> ResponseData:
    """
    Validates the input prompt and HDF5 data to produce scores.

    Parameters:
    - request (RequestData): An instance of RequestData containing the input prompt and data.

    Returns:
    - ResponseData: An instance of ResponseData containing the scores generated from the validation_lib process.

    """
    prompt = request.prompt
    data = request.data
    version = int(request.data_ver)
    loader = HDF5Loader()
    score = 0.0

    try:
        score = _validate(prompt, data, version, loader)
    except Exception as e:
        logger.exception(e)
    finally:
        _cleanup()

    return ResponseData(score=score)


@app.post("/validate_ply/", response_model=ResponseData)
async def validate_ply(request: RequestData) -> ResponseData:
    """
    Validates the input prompt and PLY data to produce scores.

    Parameters:
    - request (RequestData): An instance of RequestData containing the input prompt and data.

    Returns:
    - ResponseData: An instance of ResponseData containing the scores generated from the validation_lib process.

    """
    prompt = request.prompt
    data = request.data
    loader = PlyLoader()
    score = 0.0

    try:
        score = _validate(prompt, data, 256, loader)
    except Exception as e:
        logger.exception(e)
    finally:
        _cleanup()

    return ResponseData(score=score)


@app.get("/version/", response_model=str)
async def version() -> str:
    """
    Returns current endpoint version.
    """
    return str(VERSION)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, backlog=256)
