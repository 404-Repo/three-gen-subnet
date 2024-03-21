import argparse
import asyncio
import base64
import time
import typing
import uuid

import bittensor as bt

from common import protocol


async def main():
    config = await get_config()
    bt.logging(config=config)

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

    task = protocol.TGTask(prompt="Dog", task_id=str(uuid.uuid4()))

    await dendrite.forward(
        axons=axons,
        synapse=task,
        deserialize=False,
        timeout=60,
    )

    bt.logging.info("Task sent")

    poll = protocol.TGPoll(task_id=task.task_id)
    while True:
        rs = typing.cast(
            list[protocol.TGPoll],
            await dendrite.forward(
                axons=axons,
                synapse=poll,
                deserialize=False,
                timeout=60,
            ),
        )
        bt.logging.info({a.hotkey: r.status for r, a in zip(rs, axons)})
        for r in rs:
            if r.status == "DONE":
                with open("content_pcl.h5", "w") as f:
                    f.write(r.results)
                bt.logging.info("Result save to `content_pcl.h5`")
                return
        await asyncio.sleep(10)


async def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=29)
    return bt.config(parser)


if __name__ == "__main__":
    asyncio.run(main())
