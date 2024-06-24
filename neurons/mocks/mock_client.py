import asyncio
import base64
import json
import re
import unicodedata
from pathlib import Path

from aiohttp import ClientSession, WSMsgType
from validator.api.protocol import Auth, PromptData, TaskStatus, TaskUpdate


def normalize_filename(filename: str) -> str:
    filename = filename.lower()
    filename = filename.replace(" ", "_")
    filename = unicodedata.normalize("NFKD", filename).encode("ASCII", "ignore").decode("ASCII")
    filename = re.sub(r"[^\w\-.]", "", filename)
    filename = filename.strip("._")
    return filename


async def main() -> None:
    await asyncio.gather(*[work() for _ in range(1)])


async def work() -> None:
    prompt = "Donald Duck"

    async with ClientSession() as session:
        async with session.ws_connect("wss://1qshev6dbe7gdz-8888.proxy.runpod.net/ws/generate/") as ws:
            await ws.send_json(Auth(api_key="API KEY GOES HERE").dict())
            await ws.send_json(PromptData(prompt=prompt, send_first_results=True).dict())

            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    update = TaskUpdate(**json.loads(msg.data))
                    if update.status == TaskStatus.STARTED:
                        print("Task started")
                    elif update.status == TaskStatus.FIRST_RESULTS:
                        score = update.results.score if update.results else None
                        assets = update.results.assets or "" if update.results else ""
                        print(f"First results. Score: {score}. Size: {len(assets)}")
                    elif update.status == TaskStatus.BEST_RESULTS:
                        score = update.results.score if update.results else None
                        assets = update.results.assets or "" if update.results else ""
                        print(f"Best results. Score: {score}. Size: {len(assets)}")
                        print(f"Stats: {update.statistics}")

                        if assets:
                            with Path(normalize_filename(prompt) + ".ply").open("wb") as f:  # noqa
                                f.write(base64.b64decode(assets.encode("utf-8")))
                elif msg.type == WSMsgType.ERROR:
                    print(f"WebSocket connection closed with exception: {ws.exception()}")


if __name__ == "__main__":
    asyncio.run(main())
