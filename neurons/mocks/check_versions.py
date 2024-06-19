import argparse
import asyncio
import typing

import bittensor as bt
from common.protocol import GetVersion


async def main() -> None:
    config = await get_config()
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = bt.metagraph(netuid=config.netuid, network=subtensor.network, sync=False)
    metagraph.sync(subtensor=subtensor)
    dendrite = bt.dendrite(wallet)

    validators = [neuron.uid for neuron in metagraph.neurons if neuron.axon_info.is_serving and neuron.stake.tao > 1000]
    validators.append(47)
    bt.logging.info(f"Active validators: {validators}")

    versions = typing.cast(
        list[GetVersion],
        await dendrite.forward(
            axons=[metagraph.axons[uid] for uid in validators], synapse=GetVersion(), deserialize=False, timeout=30
        ),
    )

    for version in versions:
        bt.logging.info(
            f"{version.axon.hotkey} | "
            f"{metagraph.hotkeys.index(version.axon.hotkey)} | "
            f"{version.version} | "
            f"{version.validation_version} | "
            f"{version.dendrite.status_message} "
        )


async def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=89)
    return bt.config(parser)


if __name__ == "__main__":
    asyncio.run(main())
