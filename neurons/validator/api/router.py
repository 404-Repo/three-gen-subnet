import time
import typing

import bittensor as bt
from fastapi import APIRouter
from fastapi.security import APIKeyHeader
from starlette.websockets import WebSocket, WebSocketDisconnect

from validator.api import ApiKeyManager
from validator.api.protocol import Auth, PromptData, TaskResults, TaskStatus, TaskUpdate
from validator.api.task_registry import TaskRegistry


router = APIRouter()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_task_registry(request: WebSocket) -> TaskRegistry:
    return typing.cast(TaskRegistry, request.app.state.task_registry)


def get_api_key_manager(request: WebSocket) -> ApiKeyManager:
    return typing.cast(ApiKeyManager, request.app.state.api_key_manager)


@router.websocket("/ws/generate/")
async def websocket_generate(websocket: WebSocket) -> None:
    """
    WebSocket endpoint to manage 3D generation tasks.

    This endpoint requires an initial authentication message with a valid API key.
    Once authenticated, the client can send a prompt and receive task statuses.

    ```
    Initial message format:
    {
        "api_key": "your_api_key"
    }

    Subsequent message format for prompt:
    {
        "prompt": "3D model prompt",
        "send_first_results": true/false
    }
    ```

    Server responses for task updates:
    ```
    {
        "status": "started/first_results/best_results",
        "results": {
            "score": float,
            "assets": "base64_encoded_string"
        }
    }
    ```
    """

    await websocket.accept()

    message = await websocket.receive_text()
    auth = Auth.parse_raw(message)

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
    task_registry = get_task_registry(websocket)

    message = await websocket.receive_text()
    prompt_data = PromptData.parse_raw(message)

    bt.logging.info(f"New organic prompt received from [{client_name}]: {prompt_data.prompt}")

    start_time = time.time()
    task_id = task_registry.add_task(prompt_data.prompt)

    await task_registry.get_started(task_id)

    update = TaskUpdate(status=TaskStatus.STARTED)
    await websocket.send_text(update.json())

    first_results = await task_registry.get_first_results(task_id)
    if first_results is None:
        bt.logging.error(f"Failed to generate results for organic prompt: {prompt_data.prompt}")
        await websocket.close(code=4404, reason="Generation failed")
        return

    bt.logging.debug(
        f"First results received in {time.time() - start_time:.2f} seconds. "
        f"Prompt: {prompt_data.prompt}. Miner: {first_results.hotkey}. Size: {len(first_results.results or '')}"
    )

    if prompt_data.send_first_results:
        update = TaskUpdate(
            status=TaskStatus.FIRST_RESULTS,
            results=TaskResults(hotkey=first_results.hotkey, assets=first_results.results, score=first_results.score),
        )
        await websocket.send_text(update.json())

    best_results = await task_registry.get_best_results(task_id)
    if best_results is None:
        bt.logging.error(f"Failed to generate results for organic prompt: {prompt_data.prompt}")
        await websocket.close(code=4404, reason="Generation failed")
        return

    bt.logging.debug(
        f"Best results received in {time.time() - start_time:.2f} seconds. "
        f"Prompt: {prompt_data.prompt}. Miner: {best_results.hotkey}. Size: {len(best_results.results or '')}"
    )

    stats = task_registry.get_stats(task_id)
    task_registry.clean_task(task_id)

    update = TaskUpdate(
        status=TaskStatus.BEST_RESULTS,
        results=TaskResults(hotkey=best_results.hotkey, assets=best_results.results, score=best_results.score),
        statistics=stats,
    )
    await websocket.send_text(update.json())
    await websocket.close()
