import asyncio
import bittensor as bt

from miner import read_config, Miner


async def main():
    config = read_config()
    bt.logging.info(f"Starting with config: {config}")

    miner = Miner(config)
    await miner.run()


if __name__ == "__main__":
    asyncio.run(main())
