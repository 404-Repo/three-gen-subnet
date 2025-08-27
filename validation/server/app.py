import argparse
import asyncio
import gc
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
from engine.data_structures import RenderRequest, TimeStat, ValidationRequest, ValidationResponse
from engine.io.ply import PlyLoader
from engine.rendering.renderer import Renderer
from engine.validation_engine import ValidationEngine
from fastapi import APIRouter, FastAPI, Request, Response
from loguru import logger
from starlette.datastructures import State
from starlette.responses import StreamingResponse

from server.pipeline import decode_and_render_grid, decode_and_validate_img, decode_and_validate_txt


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    return parser.parse_args()


args: argparse.Namespace = get_args()

# Set max_workers=1 to serialize GPU requests and prevent inefficient parallel execution.
# GPU workloads don't parallelize well - multiple concurrent requests compete for the same
# GPU resources, causing each request to take longer (2 requests = 2x total time).
# By queuing requests sequentially, we ensure the first request completes as quickly as
# possible. This is especially important for miners with fixed cooldowns, as serialized
# processing spreads requests evenly over time rather than bunching completions together.
executor = ThreadPoolExecutor(max_workers=1)


class MyFastAPI(FastAPI):
    state: State
    router: APIRouter
    version: str


@asynccontextmanager
async def lifespan(app: MyFastAPI) -> AsyncIterator[None]:
    app.state.validator = ValidationEngine()
    app.state.validator.load_pipelines()

    app.state.renderer = Renderer()
    app.state.ply_data_loader = PlyLoader()

    cleanup()
    gc.collect()

    yield

    cleanup()


app = MyFastAPI(title="404--GEN Validation Service", version="2.3.1")


@app.middleware("http")
async def add_request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    request_id = str(uuid.uuid4())[:4]
    request.state.rid = request_id
    with logger.contextualize(rid=request_id):
        response = await call_next(request)
    return response


app.router.lifespan_context = lifespan


@app.post("/validate_txt_to_3d_ply/", response_model=ValidationResponse)
async def validate_txt_to_3d_ply(request: ValidationRequest) -> ValidationResponse:
    """
    Validates text-to-3D generation results by scoring how well the 3D asset matches the input prompt.
    """
    return await execute_validation(
        fn=decode_and_validate_txt,
        request=request,
        ply_data_loader=app.state.ply_data_loader,
        renderer=app.state.renderer,
        engine=app.state.validator,
    )


@app.post("/validate_img_to_3d_ply/", response_model=ValidationResponse)
async def validate_img_to_3d_ply(request: ValidationRequest) -> ValidationResponse:
    """
    Validates 2D-to-3D generation results by scoring how well the 3D asset matches the input prompt.
    """
    return await execute_validation(
        fn=decode_and_validate_img,
        request=request,
        ply_data_loader=app.state.ply_data_loader,
        renderer=app.state.renderer,
        engine=app.state.validator,
    )


@app.post("/render_duel_view/")
async def render_duel_view(request: RenderRequest) -> StreamingResponse:
    loop = asyncio.get_running_loop()
    buffer = await loop.run_in_executor(
        executor,
        decode_and_render_grid,
        request,
        app.state.ply_data_loader,
        app.state.renderer,
        app.state.validator,
    )
    return StreamingResponse(buffer, media_type="image/png")


@app.get("/version/", response_model=str)
async def version() -> str:
    """
    Returns current endpoint version.
    """
    return app.version


async def execute_validation(
    fn: Callable[[ValidationRequest, PlyLoader, Renderer, ValidationEngine], tuple[ValidationResponse, TimeStat]],
    request: ValidationRequest,
    ply_data_loader: PlyLoader,
    renderer: Renderer,
    engine: ValidationEngine,
) -> ValidationResponse:
    """Execute validation function in thread pool executor to avoid blocking the event loop."""
    try:
        loop = asyncio.get_running_loop()
        response, time_stat = await loop.run_in_executor(
            executor,
            fn,
            request,
            ply_data_loader,
            renderer,
            engine,
        )
    except Exception as e:
        logger.exception(e)
        response = ValidationResponse(score=0.0)
    finally:
        cleanup()
    return response


def cleanup() -> None:
    """Function for cleaning up the memory"""
    t1 = time.time()
    torch.cuda.empty_cache()
    gpu_memory_free, gpu_memory_total = torch.cuda.mem_get_info()
    logger.info(f"Cache purge took: {time.time() - t1} sec. VRAM Memory: {gpu_memory_free} / {gpu_memory_total}")
