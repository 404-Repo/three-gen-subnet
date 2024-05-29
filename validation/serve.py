import argparse
import gc
from time import time

from loguru import logger
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr
import torch
import uvicorn

from validation.validation_pipeline import ValidationPipeline
from validation.rendering_pipeline import RenderingPipeline

VERSION = "1.3.0"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    args, extras = parser.parse_known_args()
    return args, extras


app = FastAPI()
args, _ = get_args()


class RequestData(BaseModel):
    prompt: constr(max_length=1024)
    data: constr(max_length=100 * 1024 * 1024)
    # 0 - dream graussian prj data
    # 1 - LGM / other prj data
    data_ver: int = 0


class ResponseData(BaseModel):
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    app.state.validator = ValidationPipeline()
    app.state.validator.preload_scoring_model()

    yield

    gc.collect()
    torch.cuda.empty_cache()


app.router.lifespan_context = lifespan


@app.post("/validate/", response_model=ResponseData)
async def validate(request: RequestData) -> ResponseData:
    """
    Validates the input prompt and data to produce scores.

    Parameters:
    - request (RequestData): An instance of RequestData containing the input prompt and data.

    Returns:
    - ResponseData: An instance of ResponseData containing the scores generated from the validation process.

    """
    logger.info(f" Start validating the input 3D data.")
    logger.info(f" Input prompt: {request.prompt}")

    t1 = time()

    try:
        renderer = RenderingPipeline(512, 512, mode="gs")
        data_ready, data_out = renderer.prepare_data(request.data)
        if data_ready:
            images = renderer.render_gaussian_splatting_views(data_out, 15, 4.0, data_ver=request.data_ver)
            score = app.state.validator.validate(images, request.prompt)
        else:
            score = 0

        logger.info(f" Score: {score}. Prompt: {request.prompt}")
        error = None
    except Exception as e:
        logger.error(f" Validation failed with: {e}.")
        error = e
        score = 0.0

    t2 = time()
    logger.info(f" Validation took: {t2 - t1} sec")

    gc.collect()
    torch.cuda.empty_cache()
    gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()

    t3 = time()
    logger.info(f" Garbage collection took: {t3 - t2} sec. VRAM Memory: {gpu_memory_free} / {gpu_memory_total}")

    if error is not None:
        raise HTTPException(status_code=500, detail=str(error))

    return ResponseData(score=score)


@app.get("/version/", response_model=str)
async def version() -> str:
    """
    Returns current endpoint version.
    """
    return str(VERSION)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, backlog=256)