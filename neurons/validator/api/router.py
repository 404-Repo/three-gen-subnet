import time
import typing

import bittensor as bt
from fastapi import APIRouter, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response
from starlette.status import HTTP_403_FORBIDDEN, HTTP_404_NOT_FOUND, HTTP_429_TOO_MANY_REQUESTS

from validator.api import ApiKeyManager
from validator.api.task_registry import TaskRegistry


router = APIRouter()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_task_registry(request: Request) -> TaskRegistry:
    return typing.cast(TaskRegistry, request.app.state.task_registry)


def get_api_key_manager(request: Request) -> ApiKeyManager:
    return typing.cast(ApiKeyManager, request.app.state.api_key_manager)


class PromptData(BaseModel):
    prompt: str


@router.get(
    "/generate/",
    response_description="Binary data",
    responses={
        200: {"content": {"application/octet-stream": {}}, "description": "base64 encoded hdf5"},
        403: {"description": "Could not validate credentials"},
        404: {"description": "Failed to generate"},
        429: {"description": "Rate limit"},
        500: {"description": "Internal server error"},
    },
)
async def generate(
    request: Request,
    data: PromptData,
    api_key: str = Security(api_key_header),
) -> Response:
    api_key_manager = get_api_key_manager(request)
    if not api_key_manager.is_registered(api_key):
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials")

    if not api_key_manager.is_allowed(api_key):
        raise HTTPException(status_code=HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests, please try again later.")

    bt.logging.info(f"New organic prompt received: {data.prompt}")

    start_time = time.time()
    task_registry = get_task_registry(request)
    task_id = task_registry.add_task(data.prompt)

    first_results = await task_registry.get_first_results(task_id)
    if first_results is None:
        bt.logging.error(f"Failed to generate results for organic prompt: {data.prompt}")
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Failed to generate")

    bt.logging.debug(
        f"First results received in {time.time() - start_time:.2f} seconds. "
        f"Prompt: {data.prompt}. Miner: {first_results.hotkey}. Size: {len(first_results.results or '')}"
    )

    best_results = await task_registry.get_best_results(task_id)
    if best_results is None:
        bt.logging.error(f"Failed to generate results for organic prompt: {data.prompt}")
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Failed to generate")

    bt.logging.debug(
        f"Best results received in {time.time() - start_time:.2f} seconds. "
        f"Prompt: {data.prompt}. Miner: {best_results.hotkey}. Size: {len(best_results.results or '')}"
    )

    task_registry.clean_task(task_id)

    return Response(content=best_results.results, media_type="application/octet-stream")
