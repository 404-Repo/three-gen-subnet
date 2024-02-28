import argparse
import asyncio
import copy
import threading
import time
from functools import partial
from typing import Tuple

import bittensor as bt

from common import create_neuron_dir, synapses
from miner.task_registry import TaskRegistry
from miner.workers import worker_task


class Miner:
    uid: int
    """Each miner gets a unique identity (UID) in the network for differentiation."""
    config: bt.config
    """Copy of the original config."""
    wallet: bt.wallet
    """The wallet holds the cryptographic key pairs for the miner."""
    subtensor: bt.subtensor
    """The subtensor is our connection to the Bittensor blockchain."""
    metagraph: bt.metagraph
    """The metagraph holds the state of the network, letting us know about other validators and miners."""
    axon: bt.axon | None = None
    """Axon for external connections."""
    last_sync_time = 0.0
    """Last time the metagraph was synchronized."""
    task_registry: TaskRegistry

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

        self.last_sync_time = time.time()

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet {self.config.netuid} with uid {self.uid} "
            f"using network: {self.subtensor.chain_endpoint}"
        )

        if not self.config.blacklist.vpermit_only:
            bt.logging.warning("Security risk: requests from non-validators are allowed.")

        self.axon = bt.axon(wallet=self.wallet, config=self.config)

        bt.logging.info("Attaching forward function to the miner axon.")

        self.axon.attach(
            forward_fn=self._handshake,
            blacklist_fn=self._blacklist_handshake,
        ).attach(
            forward_fn=self._task,
            blacklist_fn=self._blacklist_task,
        ).attach(
            forward_fn=self._poll,
            blacklist_fn=self._blacklist_poll,
        )

        bt.logging.info(f"Axon created: {self.axon}")

        self.task_registry = TaskRegistry()

        # TODO: add queue limit

    def _check_for_registration(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            raise RuntimeError(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again."
            )

    async def run(self):
        bt.logging.debug("Starting the workers.")

        for endpoint in [self.config.generation.endpoint]:
            thread = threading.Thread(target=partial(worker_task, endpoint, self.task_registry))
            thread.start()

        bt.logging.debug("Starting the miner.")

        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} "
            f"with netuid: {self.config.netuid}"
        )

        while True:
            bt.logging.info("Miner running...", time.time())
            await asyncio.sleep(30)

            if self.last_sync_time + self.config.neuron.sync_interval < time.time():
                bt.logging.info("Re-syncing the metagraph")
                self.last_sync_time = time.time()
                self.metagraph.sync(subtensor=self.subtensor)

    async def _handshake(self, synapse: synapses.TGHandshakeV1) -> synapses.TGHandshakeV1:
        bt.logging.debug(f"Handshake received from: {synapse.dendrite.hotkey}")
        synapse.active_generation_endpoints = 1
        return synapse

    async def _blacklist_handshake(self, synapse: synapses.TGHandshakeV1) -> Tuple[bool, str]:
        return self._blacklist(synapse)

    async def _task(self, synapse: synapses.TGTaskV1) -> synapses.TGTaskV1:
        bt.logging.debug(f"Task received from: {synapse.dendrite.hotkey}. Prompt: {synapse.prompt}")

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.error("Unexpected happened. Hotkey was removed between blacklisting and processing")
            return synapse

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        stake = float(self.metagraph.S[uid])

        self.task_registry.add_task(synapse, stake)
        return synapse

    async def _blacklist_task(self, synapse: synapses.TGTaskV1) -> Tuple[bool, str]:
        return self._blacklist(synapse)

    async def _poll(self, synapse: synapses.TGPollV1) -> synapses.TGPollV1:
        task = self.task_registry.get_task(synapse.task_id)
        if task is None:
            synapse.status = "NOT FOUND"
        elif task.validator_hotkey != synapse.dendrite.hotkey:
            synapse.status = "FORBIDDEN"
        elif task.results is not None:
            synapse.status = "DONE"
            synapse.results = task.results
            self.task_registry.remove_task(synapse.task_id)
        elif task.failed:
            synapse.status = "FAILED"
        elif task.in_progress:
            synapse.status = "IN PROGRESS"
        else:
            synapse.status = "IN QUEUE"

        bt.logging.debug(
            f"Poll received from: {synapse.dendrite.hotkey}. Status: {synapse.status}. Task: {synapse.task_id}"
        )

        return synapse

    async def _blacklist_poll(self, synapse: synapses.TGPollV1) -> Tuple[bool, str]:
        return self._blacklist(synapse)

    def _blacklist(self, synapse: bt.Synapse) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.trace(f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if self.config.blacklist.vpermit_only and not self.metagraph.validator_permit[uid]:
            bt.logging.debug(f"Blacklisting validator without the permit {synapse.dendrite.hotkey}")
            return True, "No validator permit"

        if self.metagraph.S[uid] < self.config.blacklist.min_stake:
            bt.logging.debug(
                f"Blacklisting - not enough stake {synapse.dendrite.hotkey} " f"with {self.metagraph.S[uid]} TAO "
            )
            return True, "No validator permit"

        return False, "OK"


def read_config() -> bt.config:
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.axon.add_args(parser)

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=29)

    parser.add_argument("--neuron.sync_interval", type=int, help="Metagraph sync interval, seconds", default=30*60)

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Name of the neuron, used to determine the neuron directory",
        default="miner",
    )

    parser.add_argument(
        "--blacklist.vpermit_only",
        action="store_true",
        help="If set (default behaviour), only validators with permit are allowed to send requests.",
        default=True,
    )

    parser.add_argument(
        "--blacklist.min_stake",
        type=int,
        help="Defines the minimum stake validator should have to send requests",
        default=1,
    )

    parser.add_argument(
        "--generation.endpoint",
        type=str,
        help="Specifies the URL of the endpoint responsible for generating 3D models. "
        "This endpoint should handle the /generation/ POST route.",
        default="http://127.0.0.1:10006",
    )

    return bt.config(parser)
