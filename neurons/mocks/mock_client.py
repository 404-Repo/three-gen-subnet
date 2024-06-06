import asyncio
import json

from aiohttp import ClientSession, WSMsgType
from validator.api.protocol import Auth, PromptData, TaskStatus, TaskUpdate


async def main() -> None:
    await asyncio.gather(*[work() for _ in range(20)])


async def work() -> None:
    async with ClientSession() as session:
        async with session.ws_connect("wss://1qshev6dbe7gdz-8888.proxy.runpod.net/ws/generate/") as ws:
            await ws.send_json(Auth(api_key="").dict())
            await ws.send_json(PromptData(prompt="Donald Duck", send_first_results=True).dict())

            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    update = TaskUpdate(**json.loads(msg.data))
                    if update.status == TaskStatus.STARTED:
                        print("Task started")
                    elif update.status == TaskStatus.FIRST_RESULTS:
                        print(f"First results. Score: {update.results.score}. Size: {len(update.results.assets)}")
                    elif update.status == TaskStatus.BEST_RESULTS:
                        print(f"Best results. Score: {update.results.score}. Size: {len(update.results.assets)}")
                    else:
                        print("Wrong status")
                elif msg.type == WSMsgType.ERROR:
                    print(f"WebSocket connection closed with exception: {ws.exception()}")


if __name__ == "__main__":
    asyncio.run(main())
