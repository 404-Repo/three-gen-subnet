import argparse
import base64
import io
from time import time
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import numpy as np


from lib.validation_pipeline import Validator
from lib.rendering_pipeline import Renderer
from lib.hdf5_loader import HDF5Loader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    args, extras = parser.parse_known_args()
    return args, extras


app = FastAPI()
args, _ = get_args()


class RequestData(BaseModel):
    prompt: str
    data: str


class ResponseData(BaseModel):
    score: float


@app.on_event("startup")
def startup_event() -> None:
    app.state.validator = Validator()
    app.state.validator.preload_scoring_model()


@app.post("/validate/", response_model=ResponseData)
async def validate(request: RequestData) -> ResponseData:
    """
    Validates the input prompt and data to produce scores.

    Parameters:
    - request (RequestData): An instance of RequestData containing the input prompt and data.

    Returns:
    - ResponseData: An instance of ResponseData containing the scores generated from the validation process.

    """

    print("[INFO] Start validating the input 3D data.")
    print(f"[INFO] Input prompt: {request.prompt}")
    t1 = time()

    renderer = Renderer(512, 512)
    result = renderer.init_gaussian_splatting_renderer(request.data)
    if result:
        images = renderer.render_gaussian_splatting_views(10, 5.0)
        score = app.state.validator.validate(images, request.prompt)
    else:
        score = 0

    t2 = time()
    print(f"[INFO] Score: {score}")
    print(f"[INFO] Validation took: {t2 - t1} sec")

    return ResponseData(score=score)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
