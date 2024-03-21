import argparse
from time import time
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from lib.validators import TextTo3DModelValidator


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

    validator = TextTo3DModelValidator(512, 512, 10)
    validator.init_gaussian_splatting_renderer()
    score = validator.score_response_gs_input(request.prompt, request.data, save_images=False, cam_rad=6)

    t2 = time()
    print(f"[INFO] Score: {score}")
    print(f"[INFO] Validation took: {t2 - t1} sec")

    return ResponseData(score=score)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
