import argparse
import asyncio
import typing

import bittensor as bt
from common.protocol import Generate, PollResults


async def main() -> None:
    config = await get_config()
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = bt.metagraph(netuid=config.netuid, network=subtensor.network, sync=False)
    metagraph.sync(subtensor=subtensor)
    dendrite = bt.dendrite(wallet)

    bt.logging.info("Requesting a job")

    validator_uid = 0
    task = await generate(dendrite, metagraph, validator_uid)
    if task.task_id is None:
        bt.logging.info("Generation rejected")
        return

    bt.logging.info(f"Generation started: {task}")

    while True:
        await asyncio.sleep(10.0)
        poll = await poll_results(dendrite, metagraph, validator_uid, task.task_id)
        bt.logging.info(f"Status: {poll.status}. Results: {len(poll.results or '')}")
        if poll.status == "DONE":
            break


async def generate(dendrite: bt.dendrite, metagraph: bt.metagraph, validator_uid: int) -> Generate:
    synapse = Generate(prompt="tourmaline tassel earring")
    response = typing.cast(
        Generate,
        await dendrite.call(
            target_axon=metagraph.axons[validator_uid], synapse=synapse, deserialize=False, timeout=12.0
        ),
    )
    return response


async def poll_results(dendrite: bt.dendrite, metagraph: bt.metagraph, validator_uid: int, task_id: str) -> PollResults:
    synapse = PollResults(task_id=task_id)
    response = typing.cast(
        PollResults,
        await dendrite.call(
            target_axon=metagraph.axons[validator_uid],
            synapse=synapse,
            deserialize=False,
            timeout=300.0,
        ),
    )
    return response


async def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=89)
    return bt.config(parser)


if __name__ == "__main__":
    asyncio.run(main())
