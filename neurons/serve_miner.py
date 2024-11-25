import asyncio

from miner import Miner
from miner.config import read_config


async def main() -> None:
    config = read_config()
    miner = Miner(config)
    await miner.run()


if __name__ == "__main__":
    asyncio.run(main())
