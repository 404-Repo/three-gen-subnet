import argparse
import asyncio
import copy
import os
import random
import time
import uuid
from typing import cast

import bittensor as bt
import torch
from bittensor.utils import weight_utils

from common import create_neuron_dir
from common.protocol import TGTask, TGPoll
from validator.dataset import Dataset
from validator.fidelity_check import validate

# TODO: support dynamic neurons number increase
NEURONS_LIMIT = 256


class Validator:
    uid: int
    """Each validator gets a unique identity (UID) in the network for differentiation."""
    config: bt.config
    """Copy of the original config."""
    wallet: bt.wallet
    """The wallet holds the cryptographic key pairs for the miner."""
    subtensor: bt.subtensor
    """The subtensor is the connection to the Bittensor blockchain."""
    metagraph: bt.metagraph
    """The metagraph holds the state of the network, letting us know about other validators and miners."""
    dendrite: bt.dendrite
    """Dendrite to access other neurons."""
    scores: torch.Tensor
    """Miners' current scores. On a scale from 0 to 1"""
    active_miners: set[int]
    """Miners that were active during the last loads."""
    last_sync_time = 0.0
    """Last metagraph sync time."""
    last_info_time = 0.0
    """Last time neuron info was logged."""
    last_load_time = 0.0
    """Last time miners were loaded."""
    dataset: Dataset

    def __init__(self, config: bt.config):
        self.config: bt.config = copy.deepcopy(config)
        create_neuron_dir(self.config)

        bt.logging(config=config, logging_dir=config.full_path)

        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        self._check_for_registration()

        self.metagraph = bt.metagraph(
            netuid=self.config.netuid, network=self.subtensor.network, sync=False
        )  # Make sure not to sync without passing subtensor
        self.metagraph.sync(subtensor=self.subtensor)  # Sync metagraph with subtensor.
        bt.logging.info(f"Metagraph: {self.metagraph}")

        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        self.last_sync_time = time.time()

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet {self.config.netuid} with uid {self.uid} "
            f"using network: {self.subtensor.chain_endpoint}"
        )

        self.scores = torch.zeros(NEURONS_LIMIT, dtype=torch.float32)
        self.active_miners = set()

        self.dataset = Dataset(self.config.dataset.path)

        self._load_state()

    def _check_for_registration(self) -> None:
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            raise RuntimeError(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again."
            )

    async def run(self) -> None:
        bt.logging.debug("Starting the validator.")

        while True:
            await asyncio.sleep(5)
            self._log_info()
            self._sync_and_set_weights()
            await self._load_miners()

    def _log_info(self) -> None:
        if self.last_info_time + self.config.neuron.log_info_interval > time.time():
            return

        self.last_info_time = time.time()

        metagraph = self.metagraph

        log = (
            "Validator | "
            f"UID:{self.uid} | "
            f"Block:{metagraph.block.item()} | "
            f"Stake:{metagraph.S[self.uid]:.4f} | "
            f"VTrust:{metagraph.Tv[self.uid]:.4f} | "
            f"Dividends:{metagraph.D[self.uid]:.6f} | "
            f"Emission:{metagraph.E[self.uid]:.6f}"
        )
        bt.logging.info(log)

    def _sync_and_set_weights(self) -> None:
        if self.last_sync_time + self.config.neuron.sync_interval > time.time():
            return

        self.last_sync_time = time.time()

        bt.logging.info("Synchronizing metagraph")
        self.metagraph.sync(subtensor=self.subtensor)
        bt.logging.info("Metagraph synchronized")

        # Scores are not reset deliberately, so that new miners can start from the lowest emission but not from zero.

        if self.metagraph.last_update[self.uid] + self.config.neuron.epoch_length > self.metagraph.block:
            return

        raw_weights = torch.nn.functional.normalize(self.scores, p=1, dim=0)

        bt.logging.debug(f"Raw weights: {raw_weights}")

        (
            processed_weight_uids,
            processed_weights,
        ) = weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        (
            converted_uids,
            converted_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )

        self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=converted_uids,
            weights=converted_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=1,
        )

        self._save_state()

    async def _load_miners(self) -> None:
        if self.last_load_time + self.config.neuron.sample_interval > time.time():
            return

        self.last_load_time = time.time()

        cavies = self._get_random_miners(self.config.neuron.sample_size)
        prompt = self.dataset.get_random_prompt()
        task = TGTask(prompt=prompt, task_id=str(uuid.uuid4()))

        bt.logging.debug(f"Loading miners: {cavies}. Task_id: {task.task_id}. Prompt: {task.prompt}")

        synapses = cast(
            list[TGTask],
            await self.dendrite.forward(
                axons=[self.metagraph.axons[uid] for uid in cavies],
                synapse=task,
                deserialize=False,
            ),
        )

        strong = {uid for uid, s in zip(cavies, synapses) if s.status == "IN QUEUE"}
        weak = [uid for uid, s in zip(cavies, synapses) if s.status != "IN QUEUE"]

        bt.logging.debug(f"Registered task with {strong}. Failed with {weak}")

        if strong:
            self.active_miners.update(strong)
            asyncio.create_task(self._concurrent_poll(prompt, task.task_id, strong))

        for uid in weak:
            self._update_miner_score(uid, 0.0)

    def _get_random_miners(self, sample_size: int) -> list[int]:
        all_uids = list(range(self.metagraph.n))
        del all_uids[self.uid]
        random.shuffle(all_uids)
        return all_uids[:sample_size]

    def _update_miner_score(self, uid: int, observation: float) -> None:
        alpha = self.config.neuron.moving_average_alpha
        prev = self.scores[uid].item()
        new = prev * (1 - alpha) + alpha * observation
        self.scores[uid] = new

        bt.logging.debug(f"Miner {uid:3d} score. Prev: {prev:.2f} | Obs: {observation:.2f} | New: {new:.2f}")

    async def _concurrent_poll(self, prompt: str, task_id: str, miners: set[int]) -> None:
        # TODO: implement registry and callback
        current_time = time.time()
        start_time = current_time
        poll_time = current_time

        while bool(miners) and start_time + self.config.neuron.task_timeout > current_time:
            await asyncio.sleep(max(5, poll_time + self.config.neuron.task_poll_interval - current_time))

            poll_time = time.time()
            poll = TGPoll(task_id=task_id)
            synapses = cast(
                list[TGPoll],
                await self.dendrite.forward(
                    axons=[self.metagraph.axons[uid] for uid in miners],
                    synapse=poll,
                    deserialize=False,
                    timeout=30,
                ),
            )
            current_time = time.time()

            bt.logging.trace(f"Task {task_id} statuses: {', '.join(s.status or 'None' for s in synapses)}")

            for uid, s in zip(set(miners), synapses):
                if s.status in {None, "IN QUEUE", "IN PROGRESS"}:
                    continue

                miners.remove(uid)

                if s.status != "DONE" or s.results is None:
                    bt.logging.debug(f"Miner {uid} failed.")
                    self._update_miner_score(uid, 0)
                    continue

                time_factor = max(0.0, 1 - (poll_time - start_time) / self.config.neuron.task_timeout)

                bt.logging.debug(
                    f"Miner {uid} completed task in {poll_time - start_time:.2f} seconds. "
                    f"Task size: {len(s.results or b'')}. Reward time factor: {time_factor:.2f}"
                )

                asyncio.create_task(self._concurrent_reward(uid, prompt, s.results, time_factor))

        if bool(miners):
            bt.logging.info(f"Task {task_id} timed out for {miners}")
            for uid in miners:
                self._update_miner_score(uid, 0)

    async def _concurrent_reward(self, uid: int, prompt: str, result: str, time_factor: float) -> None:
        fidelity_factor = await validate(self.config.validation.endpoint, prompt, result)
        self._update_miner_score(uid, fidelity_factor * time_factor)

    def _save_state(self):
        bt.logging.info("Saving validator state.")

        torch.save(
            {
                "scores": self.scores,
            },
            self.config.neuron.full_path + "/state.pt",
        )

    def _load_state(self):
        bt.logging.info("Loading validator state.")

        if not os.path.exists(self.config.neuron.full_path + "/state.pt"):
            bt.logging.warning("No saved state found")
            return

        try:
            state = torch.load(self.config.neuron.full_path + "/state.pt")
            self.scores = state["scores"]
        except Exception as e:
            bt.logging.exception(f"Failed to load the state: {e}")


def read_config() -> bt.config:
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.axon.add_args(parser)

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=29)

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Name of the neuron, used to determine the neuron directory",
        default="validator",
    )

    parser.add_argument("--neuron.sync_interval", type=int, help="Metagraph sync interval, seconds", default=10 * 60)

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="Weight set interval, measured in blocks.",
        default=100,
    )

    # parser.add_argument("--neuron.sync_interval", type=int, help="Metagraph sync interval, seconds", default=20)
    #
    # parser.add_argument(
    #     "--neuron.epoch_length",
    #     type=int,
    #     help="Weight set interval, measured in blocks.",
    #     default=5,
    # )

    parser.add_argument(
        "--neuron.sample_interval",
        type=int,
        help="Synthetic requests interval, seconds.",
        default=300,
    )

    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="Number of neurons to query for synthetic requests.",
        default=10,
    )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha parameter for the score updating.",
        default=0.05,
    )

    parser.add_argument(
        "--neuron.log_info_interval",
        type=int,
        help="Interval for logging the validator state, seconds.",
        default=30,
    )

    parser.add_argument(
        "--neuron.task_poll_interval",
        type=int,
        help="Task polling interval, seconds.",
        default=30,
    )

    parser.add_argument(
        "--neuron.task_timeout",
        type=int,
        help="Task execution timeout, seconds.",
        default=600,
    )

    parser.add_argument(
        "--dataset.path",
        type=str,
        help="Path to the file with the prompts (relative or absolute)",
        default="resources/prompts.txt",
    )

    parser.add_argument(
        "--validation.endpoint",
        type=str,
        help="Specifies the URL of the endpoint responsible for scoring 3D models. "
        "This endpoint should handle the /validate/ POST route.",
        default="http://127.0.0.1:8094",
    )

    return bt.config(parser)
