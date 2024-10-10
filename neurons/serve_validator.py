import asyncio

import bittensor as bt
from validator import Validator
from validator.config import read_config


async def main() -> None:
    config = read_config()
    bt.logging.info(f"Starting with config: {config}")

    neuron = Validator(config)
    await neuron.run()


if __name__ == "__main__":
    asyncio.run(main())
