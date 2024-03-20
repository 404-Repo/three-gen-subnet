import argparse
import asyncio
import copy
import time

import bittensor as bt
import torch
from bittensor.utils import weight_utils

from common import create_neuron_dir

# TODO: support dynaic neurons number increase
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
    last_sync_time = 0.0
    """Last metagraph sync time."""
    last_info_time = 0.0
    """Last time neuron info was logged."""

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

        # TODO: load/save state
        # TODO: weights update

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
        # bt.logging.debug("Starting the workers.")

        # for endpoint in [self.config.generation.endpoint]:
        #     thread = threading.Thread(target=partial(worker_task, endpoint, self.task_registry))
        #     thread.start()

        bt.logging.debug("Starting the validator.")

        while True:
            await asyncio.sleep(5)
            self._log_info()
            self._set_weights()

    def _log_info(self):
        if self.last_info_time + self.config.neuron.log_info_interval > time.time():
            return

        self.last_info_time = time.time()

        metagraph = self.metagraph

        log = (
            "Validator | "
            f"UID:{self.uid} | "
            f"Block:{metagraph.block.int()} | "
            f"Stake:{metagraph.S[self.uid]:.4f} | "
            f"VTrust:{metagraph.Tv[self.uid]:.4f} | "
            f"Dividends:{metagraph.D[self.uid]:.6f} | "
            f"Emission:{metagraph.E[self.uid]:.6f}"
        )
        bt.logging.info(log)

    def _set_weights(self):
        if self.last_sync_time + self.config.neuron.sync_interval > time.time():
            return

        self.last_sync_time = time.time()

        bt.logging.info("Synchronizing metagraph")
        self.metagraph.sync(subtensor=self.subtensor)
        bt.logging.info("Metagraph synchronized")

        # Scores are not reset deliberately, so that new miners can start from the lowest emission but not from zero.

        bt.logging.info(f"!!!!!!!!!!!!!!!! {len(self.metagraph.axons)}")
        bt.logging.info(f"!!!!!!!!!!!!!!!! {self.metagraph.W}")

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

        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=converted_uids,
            weights=converted_weights,
            wait_for_finalization=True,
            wait_for_inclusion=True,
            version_key=1,
        )
        if not result:
            bt.logging.error("WAAAGH!")

    def probe_or_better_name(self):
        # get random neurons with endpoints (passed handshake)
        # create task
        # send task to random
        # for now, task timeout should be equal to delay
        # ? Synthetic vs organic ?
        # Delay next synthetic if organic (enough organic) received?
        # repeat polling, removing completed/failed
        pass


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
        "--neuron.log_info_interval",
        type=int,
        help="Interval for logging the validator state, seconds.",
        default=30,
    )

    parser.add_argument(
        "--validation.endpoint",
        type=str,
        help="Specifies the URL of the endpoint responsible for scoring 3D models. "
             "This endpoint should handle the /validate/ POST route.",
        default="http://127.0.0.1:10006",
    )

    return bt.config(parser)
