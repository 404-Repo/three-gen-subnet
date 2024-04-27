import argparse
import asyncio
import typing

import bittensor as bt
from api import Generate, StatusCheck, TaskStatus


async def main() -> None:
    config = await get_config()
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = bt.metagraph(netuid=config.netuid, network=subtensor.network, sync=False)
    metagraph.sync(subtensor=subtensor)
    dendrite = bt.dendrite(wallet)

    bt.logging.info("Requesting the job")

    validator_uid = 0
    task = await generate(dendrite, metagraph, validator_uid)
    if task.task_id is None:
        bt.logging.info("Generation rejected")
        return

    bt.logging.info(f"Generation started: {task}")

    while True:
        await asyncio.sleep(10.0)
        check = await check_task(dendrite, metagraph, validator_uid, task.task_id)
        bt.logging.info(f"Status: {check.status}. Results: {len(check.results or '')}")
        if check.status == TaskStatus.DONE:
            break


async def generate(dendrite: bt.dendrite, metagraph: bt.metagraph, validator_uid: int) -> Generate:
    synapse = Generate(prompt="Bill Gates")
    response = typing.cast(
        Generate,
        await dendrite.call(
            target_axon=metagraph.axons[validator_uid], synapse=synapse, deserialize=False, timeout=12.0
        ),
    )
    return response


async def check_task(dendrite: bt.dendrite, metagraph: bt.metagraph, validator_uid: int, task_id: str) -> StatusCheck:
    synapse = StatusCheck(task_id=task_id)
    response = typing.cast(
        StatusCheck,
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
