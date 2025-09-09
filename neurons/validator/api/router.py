"""
This module is a legacy way of interacting with the validator.
Gateway API is a new way of interacting with the validator.
TODO: remove this module after the transition to the new API is complete.
"""

import time
import typing
from uuid import uuid4

import bittensor as bt
from fastapi import APIRouter
from fastapi.security import APIKeyHeader
from starlette.websockets import WebSocket, WebSocketDisconnect

from validator.api.api_key_manager import ApiKeyManager
from validator.api.protocol import Auth, PromptData, TaskResults, TaskStatus, TaskUpdate
from validator.task_manager.task_manager import TaskManager
from validator.task_manager.task_storage.organic_task import LegacyOrganicTask


# This router is used by clients that want to generate 3D assets from prompts.
router = APIRouter()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

task_manager: TaskManager  # Set in serve_validator.


def get_api_key_manager(request: WebSocket) -> ApiKeyManager:
    return typing.cast(ApiKeyManager, request.app.state.api_key_manager)


@router.websocket("/ws/generate/")
async def websocket_generate(websocket: WebSocket) -> None:
    """
    WebSocket endpoint to manage 3D generation tasks.

    This endpoint requires an initial authentication message with a valid API key.
    Once authenticated, the client can send a prompt and receive task statuses.

    WebSocket pipeline is following:
    - Initial message with API key.
    - Message with prompt to generate 3D asset.
    - Wait for task to be assigned to some miner.
    - Send Started status to client.
    - Wait for first results from miner.
    - Wait for best results from miner.
    - Send best results to client.
    - Close connection.

    ```
    Initial message format:
    {
        "api_key": "your_api_key"
    }

    Subsequent message format for prompt:
    {
        "prompt": "3D model prompt",
    }
    ```

    Server responses for task updates:
    ```
    {
        "status": "started/best_results",
        "results": {
            "score": float,
            "assets": "base64_encoded_string"
        }
    }
    ```
    """

    await websocket.accept()

    message = await websocket.receive_json()
    auth = Auth.model_validate(message)

    api_key_manager = get_api_key_manager(websocket)
    if not api_key_manager.is_registered(auth.api_key):
        await websocket.close(code=4003, reason="Invalid API Key")
        return

    if not api_key_manager.is_allowed(auth.api_key):
        await websocket.close(code=4429, reason="Too Many Requests")
        return

    key_name = api_key_manager.get_name(auth.api_key)
    client_name = f"{key_name} ({websocket.client})"

    try:
        await _websocket_generate(websocket, client_name)
    except WebSocketDisconnect:
        bt.logging.debug(f"websocket client disconnected: {client_name}")
    except Exception as e:
        bt.logging.error(f"Error while processing generation request from {client_name}: {e}")
        await websocket.close(code=4500, reason="Internal Server Error")


async def _websocket_generate(websocket: WebSocket, client_name: str) -> None:
    message = await websocket.receive_json()
    prompt_data = PromptData.model_validate(message)

    bt.logging.info(f"New organic prompt received from [{client_name}]: {prompt_data.prompt}")

    start_time = time.time()
    legacy_task = task_manager._organic_task_storage.add_legacy_task(  # noqa: F821
        task=LegacyOrganicTask.create_task(id=str(uuid4()), prompt=prompt_data.prompt)
    )

    await legacy_task.start_future

    update = TaskUpdate(status=TaskStatus.STARTED)
    await websocket.send_json(update.model_dump())

    first_result = await legacy_task.first_result_future
    if first_result is None:
        bt.logging.warning(f"Failed to generate results for organic prompt: {prompt_data.prompt}")
        await websocket.close(code=4404, reason="Generation failed")
        return

    bt.logging.debug(
        f"First result received in {time.time() - start_time:.2f} seconds. "
        f"Prompt: {prompt_data.prompt}. Miner: {first_result.hotkey}. "
        f"Size: {len(first_result.compressed_result or '')}"
    )

    best_result = await task_manager._organic_task_storage.get_best_results(task_id=legacy_task.id)  # noqa: F821
    if best_result is None:
        bt.logging.warning(f"Failed to generate results for organic prompt: {prompt_data.prompt}")
        await websocket.close(code=4404, reason="Generation failed")
        return

    bt.logging.debug(
        f"Best result received in {time.time() - start_time:.2f} seconds. "
        f"Prompt: {prompt_data.prompt}. Miner: {best_result.hotkey}. "
        f"Size: {len(best_result.compressed_result or '')}"
    )

    stats = legacy_task.get_stats()

    update = TaskUpdate(
        status=TaskStatus.BEST_RESULTS,
        results=TaskResults(
            hotkey=best_result.hotkey, assets=best_result.decompress_results(), score=best_result.rating
        ),
        statistics=stats,
    )
    await websocket.send_json(update.model_dump_json())
    await websocket.close()
