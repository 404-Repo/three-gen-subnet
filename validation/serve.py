import argparse
import gc
from time import time

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, constr
import torch
import uvicorn

from lib.validation_pipeline import Validator
from lib.rendering_pipeline import Renderer

VERSION = "1.0.0"


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


class ResponseData(BaseModel):
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    app.state.validator = Validator()
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
    print(f"[INFO] Start validating the input 3D data. Data size: {len(request.data)}")
    print(f"[INFO] Input prompt: {request.prompt}")
    t1 = time()

    try:
        renderer = Renderer(512, 512)
        renderer.init_gaussian_splatting_renderer()
        images = renderer.render_gaussian_splatting_views(request.data, 10, 5.0)
        score = app.state.validator.validate(images, request.prompt)
    except Exception as e:
        print(f"[ERROR] Validation failed with: {e}")
        score = 0.0

    t2 = time()
    print(f"[INFO] Score: {score}")
    print(f"[INFO] Validation took: {t2 - t1} sec")

    gc.collect()
    torch.cuda.empty_cache()

    t3 = time()
    print(f"[INFO] Garbage collection took: {t3 - t2} sec")

    return ResponseData(score=score)


@app.get("/version/", response_model=str)
async def version() -> str:
    """
    Returns current endpoint version.
    """
    return str(VERSION)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, backlog=256)
