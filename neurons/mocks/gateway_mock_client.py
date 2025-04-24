import argparse
import asyncio
import random as rd
import traceback
from datetime import datetime
from pathlib import Path

import aiohttp
import anyio
from pydantic import BaseModel


class GatewayTestClientTask(BaseModel):
    id: str
    prompt: str
    created_at: datetime | None = None
    result_at: datetime | None = None
    gateway: str | None = None


class GatewayTestClient:
    """For test purposes only. Sends requests to the gateway on behalf of the client."""

    _INTERVAL_SEND_TASK_SEC: int = 1
    _INTERVAL_GET_RESULT_SEC: int = 10

    def __init__(self, *, x_api_key: str) -> None:
        self._x_api_key = x_api_key
        self._tasks: dict[str, GatewayTestClientTask] = {}

    async def send_task_cron(self) -> None:
        prompts_file = Path(__file__).resolve().parent.parent.parent / "resources/prompts.txt"
        with prompts_file.open("r") as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts]

        gateways = [
            "https://gateway-us-west.404.xyz:4443",
            "https://gateway-us-east.404.xyz:4443",
            "https://gateway-eu.404.xyz:4443",
        ]

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    prompt = rd.choice(prompts)  # noqa: S311 # nosec: B311
                    gateway = rd.choice(gateways)  # noqa: S311 # nosec: B311
                    url = f"{gateway}/add_task"

                    headers = {"x-api-key": self._x_api_key}
                    payload = {"prompt": prompt}

                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status == 200:
                            result_data = await response.json()
                            task = GatewayTestClientTask.model_validate(result_data)
                            task.created_at = datetime.now()
                            task.gateway = gateway
                            self._tasks[task.id] = task
                            print(f"({task.id}) Sent prompt: {prompt} to gateway: {gateway}")
                        else:
                            print(f"Failed to send task. Status: {response.status}")

                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"Error sending prompt: {e}")
                await asyncio.sleep(self._INTERVAL_SEND_TASK_SEC)

    async def get_result_cron(self) -> None:
        async with aiohttp.ClientSession() as session:
            while True:
                for _, task in self._tasks.items():
                    if task.result_at is None:
                        try:
                            await asyncio.sleep(0.1)
                            url = f"{task.gateway}/get_result?id={task.id}"
                            headers = {"x-api-key": self._x_api_key}

                            async with session.get(url, headers=headers) as response:
                                if response.status == 200:
                                    task.result_at = datetime.now()
                                    if task.created_at is not None:
                                        print(f"({task.id}) Received result after {task.result_at - task.created_at}")
                                else:
                                    # Task might still be processing
                                    continue

                        except Exception:
                            print(f"Error getting result for task {task.id}: {traceback.format_exc()}")
                            continue

                await asyncio.sleep(self._INTERVAL_GET_RESULT_SEC)


async def main(x_api_key: str) -> None:
    client = GatewayTestClient(x_api_key=x_api_key)
    asyncio.create_task(client.send_task_cron())
    asyncio.create_task(client.get_result_cron())
    event = anyio.Event()
    await event.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-api-key", type=str)
    args = parser.parse_args()
    asyncio.run(main(x_api_key=args.x_api_key))
