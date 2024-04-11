from io import BytesIO

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
import argparse
import base64
from time import time

from omegaconf import OmegaConf

from DreamGaussianLib import GaussianProcessor, ModelsPreLoader, HDF5Loader
from utils.video_utils import VideoUtils


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


def get_models(config: OmegaConf = Depends(get_config)):
    return ModelsPreLoader.preload_model(config, "cuda")


@app.post("/generate/")
async def generate(
    prompt: str = Form(),
    config: OmegaConf = Depends(get_config),
    models: list = Depends(get_models),
):
    buffer = await _generate(models, config, prompt)
    buffer = base64.b64encode(buffer.getbuffer()).decode("utf-8")
    return Response(content=buffer, media_type="application/octet-stream")


async def _generate(models: list, opt: OmegaConf, prompt: str) -> BytesIO:
    start_time = time()
    gaussian_processor = GaussianProcessor.GaussianProcessor(opt, prompt)
    processed_data = gaussian_processor.train(models, opt.iters)
    hdf5_loader = HDF5Loader.HDF5Loader()
    buffer = hdf5_loader.pack_point_cloud_to_io_buffer(*processed_data)
    print(f"[INFO] It took: {(time() - start_time) / 60.0} min")
    return buffer


@app.post("/generate_raw/")
async def generate_raw(
    prompt: str = Form(),
    opt: OmegaConf = Depends(get_config),
    models: list = Depends(get_models),
):
    buffer = await _generate(models, opt, prompt)
    return Response(content=buffer.getvalue(), media_type="application/octet-stream")


@app.post("/generate_model/")
async def generate_model(
    prompt: str = Form(),
    opt: OmegaConf = Depends(get_config),
    models: list = Depends(get_models),
) -> Response:
    start_time = time()
    gaussian_processor = GaussianProcessor.GaussianProcessor(opt, prompt)
    gaussian_processor.train(models, opt.iters)
    print(f"[INFO] It took: {(time() - start_time) / 60.0} min")

    buffer = BytesIO()
    gaussian_processor.get_gs_model().save_ply(buffer)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/octet-stream")


@app.post("/generate_video/")
async def generate_video(
    prompt: str = Form(),
    video_res: int = Form(1088),
    opt: OmegaConf = Depends(get_config),
    models: list = Depends(get_models),
):
    start_time = time()
    gaussian_processor = GaussianProcessor.GaussianProcessor(opt, prompt)
    processed_data = gaussian_processor.train(models, opt.iters)
    print(f"[INFO] It took: {(time() - start_time) / 60.0} min")

    video_utils = VideoUtils(video_res, video_res, 5, 5, 10, -30, 10)
    buffer = video_utils.render_video(*processed_data)

    return StreamingResponse(content=buffer, media_type="video/mp4")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
