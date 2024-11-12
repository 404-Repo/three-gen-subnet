from io import BytesIO

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
import argparse
import base64
from time import time

from omegaconf import OmegaConf
from loguru import logger

from DreamGaussianLib import ModelsPreLoader
from DreamGaussianLib.GaussianProcessor import GaussianProcessor
from video_utils import VideoUtils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument("--config", default="configs/text_mv.yaml")
    return parser.parse_args()


args = get_args()
app = FastAPI()


def get_config() -> OmegaConf:
    config = OmegaConf.load(args.config)
    return config


# def get_models(config: OmegaConf = Depends(get_config)):
#     return ModelsPreLoader.preload_model(config, "cuda")


@app.on_event("startup")
def startup_event() -> None:
    config = get_config()
    app.state.models = ModelsPreLoader.preload_model(config, "cuda")


@app.post("/generate/")
async def generate(
    prompt: str = Form(),
    opt: OmegaConf = Depends(get_config),
    #models: list = Depends(get_models),
) -> Response:
    t0 = time()
    gaussian_processor = GaussianProcessor(opt, prompt)
    gaussian_processor.train(app.state.models, opt.iters)
    t1 = time()
    logger.info(f" Generation took: {(t1 - t0) / 60.0} min")

    buffer = BytesIO()
    gaussian_processor.get_gs_model().save_ply(buffer)
    buffer.seek(0)
    buffer = base64.b64encode(buffer.getbuffer()).decode("utf-8")
    t2 = time()
    logger.info(f" Saving and encoding took: {(t2 - t1) / 60.0} min")

    return Response(buffer, media_type="application/octet-stream")


@app.post("/generate_video/")
async def generate_video(
    prompt: str = Form(),
    video_res: int = Form(1088),
    opt: OmegaConf = Depends(get_config),
    #models: list = Depends(get_models),
):
    t1 = time()
    gaussian_processor = GaussianProcessor(opt, prompt)
    gaussian_processor.train(app.state.models, opt.iters)
    processed_data = gaussian_processor.get_gs_data(return_rgb_colors=True)

    logger.info(f" It took: {(time() - t1) / 60.0} min")
    logger.info("Generating video.")

    t2 = time()
    video_utils = VideoUtils(video_res, video_res, 5, 5, 10, -30, 10)
    buffer, _ = video_utils.render_video(
        processed_data[0],
        None,
        processed_data[4],
        None,
        processed_data[3],
        processed_data[2],
        processed_data[1],
        None
    )
    logger.info(f" It took: {(time() - t2) / 60.0} min")

    return StreamingResponse(content=buffer, media_type="video/mp4")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
