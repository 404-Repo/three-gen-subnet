import argparse
import asyncio
import time
import typing
from base64 import b64encode
from pathlib import Path

import bittensor as bt
from common.protocol import Feedback, PullTask, SubmitResults, Task, Version


async def main() -> None:
    config = await get_config()
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = bt.metagraph(netuid=config.netuid, network=subtensor.network, sync=False)
    metagraph.sync(subtensor=subtensor)
    dendrite = bt.dendrite(wallet)

    bt.logging.info("Pulling a task")

    validator_uid = 0
    task = await pull_task(dendrite, metagraph, validator_uid)
    if task is None:
        bt.logging.info("No task received")
        return

    bt.logging.info(f"Task received: {task}")

    bt.logging.info("Faking the generation time for 10 seconds")

    await asyncio.sleep(20.0)

    bt.logging.info("Submitting results")

    feedback, cooldown = await submit_results(dendrite, metagraph, validator_uid, task)

    bt.logging.info(f"Received feedback {feedback} and cooldown for {int(cooldown - time.time())} seconds")


async def pull_task(dendrite: bt.dendrite, metagraph: bt.metagraph, validator_uid: int) -> Task | None:
    synapse = PullTask(version=Version(major=0, minor=0, patch=1))
    axon = metagraph.axons[validator_uid]
    response = typing.cast(
        PullTask,
        await dendrite.call(target_axon=axon, synapse=synapse, deserialize=False, timeout=12.0),
    )

    bt.logging.info(f"Response: {response}")

    return response.task


async def submit_results(
    dendrite: bt.dendrite, metagraph: bt.metagraph, validator_uid: int, task: Task
) -> tuple[Feedback | None, int]:
    with Path("monkey.h5").open("r") as f:  # noqa
        results = f.read()

    axon = metagraph.axons[validator_uid]

    message = f"{0}{task.prompt}{metagraph.hotkeys[validator_uid]}{dendrite.keypair.ss58_address}"
    signature = dendrite.keypair.sign(message)
    synapse = SubmitResults(task=task, results=results, submit_time=0, signature=b64encode(signature))
    response = typing.cast(
        SubmitResults,
        await dendrite.call(
            target_axon=axon,
            synapse=synapse,
            deserialize=False,
            timeout=300.0,
        ),
    )
    return response.feedback, response.cooldown_until


async def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=89)
    return bt.config(parser)


if __name__ == "__main__":
    asyncio.run(main())
