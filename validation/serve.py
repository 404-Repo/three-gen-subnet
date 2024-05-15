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


def check_memory_footprint(data: str, memory_limit: int):
    """ Function that checks whether the input data will fit in the GPU VRAM

    Parameters
    ----------
    pcl_raw - raw input data that was received by the validator
    memory_limit - the amount of memory that is currently available in the GPU

    Returns True if the size of the input data can be fit in the VRAM, otherwise return False
    -------

    """
    # unpack data
    pcl_raw = base64.b64decode(data)
    pcl_buffer = io.BytesIO(pcl_raw)

    # unpack data to dictionary
    hdf5_loader = HDF5Loader()
    data_dict = hdf5_loader.unpack_point_cloud_from_io_buffer(pcl_buffer)
    data_arr = [np.array(data_dict["points"]),
                np.array(data_dict["normals"]),
                np.array(data_dict["features_dc"]),
                np.array(data_dict["features_rest"]),
                np.array(data_dict["opacities"])]

    total_memory_bytes = 0
    for d in data_arr:
        total_memory_bytes += d.nbytes

    if total_memory_bytes <= memory_limit:
        return True
    else:
        print("\n[INFO] Total VRAM available: ", memory_limit/int(1e+9), " Gb")
        print("[INFO] Total data size to load to VRAM: ", total_memory_bytes/int(1e+9), " Gb")
        print("[INFO] Input data size exceeds the available VRAM free memory!")
        print("[INFO] Input data will not be further processed.\n")
        return False


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

    gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info(0)
    result = check_memory_footprint(request.data, gpu_memory_free)

    if result:
        print("[INFO] Start validating the input 3D data.")
        print(f"[INFO] Input prompt: {request.prompt}")
        t1 = time()

        renderer = Renderer(512, 512)
        renderer.init_gaussian_splatting_renderer()
        images = renderer.render_gaussian_splatting_views(request.data, 10, 5.0)
        score = app.state.validator.validate(images, request.prompt)

        t2 = time()
        print(f"[INFO] Score: {score}")
        print(f"[INFO] Validation took: {t2 - t1} sec")

        return ResponseData(score=score)
    else:
        return ResponseData(score=0)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
