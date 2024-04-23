import argparse
import asyncio
import time

import bittensor as bt
from common.protocol import Feedback, PullTask, SubmitResults, Task, Version


async def main() -> None:
    config = await get_config()
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = bt.metagraph(netuid=config.netuid, network=subtensor.network, sync=False)
    metagraph.sync(subtensor=subtensor)
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

    bt.logging.info(f"Stake: {metagraph.S[uid]}")

    axon = bt.axon(wallet=wallet, config=config)

    axon.attach(
        forward_fn=pull_task,
    ).attach(
        forward_fn=submit_results,
    )

    axon.serve(netuid=config.netuid, subtensor=subtensor)
    axon.start()

    while True:
        await asyncio.sleep(30.0)


def pull_task(synapse: PullTask) -> PullTask:
    synapse.version = Version(major=0, minor=0, patch=17)
    synapse.task = Task(prompt="Yeti")
    synapse.submit_before = int(time.time() + 600)
    return synapse


def submit_results(synapse: SubmitResults) -> SubmitResults:
    synapse.feedback = Feedback(
        task_fidelity_score=0.75,
        average_fidelity_score=0.8,
        generations_within_8_hours=42,
        current_miner_reward=0.8 * 42,
    )
    synapse.cooldown_until = int(time.time() + 60.0)
    return synapse


async def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.axon.add_args(parser)
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=89)
    return bt.config(parser)


if __name__ == "__main__":
    asyncio.run(main())
