import argparse
import asyncio
import typing
import uuid
from pathlib import Path

import aiohttp
import bittensor as bt
from common import protocol


async def main() -> None:
    config = await get_config()
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = bt.metagraph(netuid=config.netuid, network=subtensor.network, sync=False)
    metagraph.sync(subtensor=subtensor)
    dendrite = bt.dendrite(wallet)
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

    bt.logging.info(f"Stake: {metagraph.S[uid]}")

    alive_axon_uid = 1

    prompt = "tourmaline tassel earring"
    task = protocol.TGTask(prompt=prompt, task_id=str(uuid.uuid4()))

    await dendrite.call(
        target_axon=metagraph.axons[alive_axon_uid],
        synapse=task,
        deserialize=False,
    )

    bt.logging.info("Task sent")

    poll = protocol.TGPoll(task_id=task.task_id)
    results = None
    while True:
        await asyncio.sleep(10)

        r = typing.cast(
            protocol.TGPoll,
            await dendrite.call(
                target_axon=metagraph.axons[alive_axon_uid],
                synapse=poll,
                deserialize=False,
            ),
        )

        bt.logging.info(f"Response to the poll. Status: {r.status}. Bytes: {len(r.results or '')}")

        if r.status in {None, "IN QUEUE", "IN PROGRESS"}:
            continue

        if r.status == "DONE" and r.results is not None:
            results = r.results
            with Path("content_pcl.h5").open("w") as f:  # noqa
                f.write(results)
            bt.logging.info("Result save to `content_pcl.h5`")
            break

        bt.logging.info("Generation failed")
        break

    if results is None:
        return

    async with aiohttp.ClientSession() as session:
        async with session.post("http://127.0.0.1:8094/validate/", json={"prompt": prompt, "data": results}) as req:
            validation = await req.json()
            bt.logging.info(f"Validation: {validation.json()}")


async def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=89)
    return bt.config(parser)


if __name__ == "__main__":
    asyncio.run(main())
