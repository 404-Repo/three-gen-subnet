import asyncio

import aiohttp
import bittensor as bt


async def main() -> None:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "http://127.0.0.1:8092/generate",
            json={
                "prompt": "Bill Gates",
            },
            headers={"X-API-Key": "THE-API-KEY"},
        ) as response:
            data = await response.read()
            bt.logging.info(f"{len(data)} bytes received. Http reason: {response.reason}")


if __name__ == "__main__":
    asyncio.run(main())
