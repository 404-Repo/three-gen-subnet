import argparse
import asyncio
import typing
import uuid

import bittensor as bt

from common import synapses


async def main():
    config = await get_config()
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = bt.metagraph(netuid=config.netuid, network=subtensor.network, sync=False)
    metagraph.sync(subtensor=subtensor)
    dendrite = bt.dendrite(wallet)
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

    bt.logging.info(f"Stake: {metagraph.S[uid]}")

    axons = list(metagraph.axons)
    del axons[uid]

    bt.logging.info(f"Other axons: {axons}")

    handshakes = await dendrite.forward(
        axons=axons,
        synapse=synapses.TGHandshakeV1(),
        deserialize=False,
        timeout=5,
    )

    bt.logging.info(f"Handshakes: {handshakes}")

    coros = [send_task(prompt, dendrite, axons) for prompt in ("House", "Dog")]

    await asyncio.gather(*coros)


async def send_task(prompt: str, dendrite: bt.dendrite, axons: list[bt.axon]) -> None:
    rs = typing.cast(
        list[synapses.TGTaskV1],
        await dendrite.forward(
            axons=axons,
            synapse=synapses.TGTaskV1(task_id=str(uuid.uuid4()), prompt=prompt),
            deserialize=False,
            timeout=60,
        ),
    )

    tasks = {r.task_id: axon for axon, r in zip(axons, rs)}
    bt.logging.info(f"IDs received: {tasks}")

    await asyncio.sleep(5.0)

    while tasks:
        for task_id, axon in list(tasks.items()):
            poll = typing.cast(
                synapses.TGPollV1,
                await dendrite.call(
                    target_axon=axon,
                    synapse=synapses.TGPollV1(task_id=task_id),
                    deserialize=False,
                    timeout=60,
                ),
            )
            bt.logging.info(poll)
            if poll.status not in {None, "IN QUEUE", "IN PROGRESS"}:
                tasks.pop(task_id)
        await asyncio.sleep(10.0)


async def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=29)
    return bt.config(parser)


if __name__ == "__main__":
    asyncio.run(main())
