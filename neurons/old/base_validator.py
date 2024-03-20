import argparse
import asyncio
import os
import random
import threading
import time
from abc import ABC
from typing import List
from traceback import print_exception

import bittensor as bt
import bittensor.utils.weight_utils
import torch
from bittensor.utils import weight_utils


class BaseValidatorNeuron(ABC):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    axon: bt.axon | None = None
    """Axon for external connections."""

    scores: torch.Tensor
    """1D (vector) with scores for each neuron on the subnet."""
    hotkeys: List[str]
    """A copy of metagraph.hotkeys."""

    is_background_thread_running: bool = False
    """Indicates whether background thread is running."""
    should_stop_background_thread: bool = False
    """When set background thread is supposed to stop."""
    should_exit: threading.Event
    """Set in the background thread on errors. Should lead to the application stop."""
    thread: threading.Thread | None = None
    """Background thread."""

    step: int = 0
    """Increased each time the validator sends probes to the miners."""

    loop: asyncio.AbstractEventLoop
    """Async loop."""

    def __init__(self, config: bt.config):
        # # Set up initial scores.
        self.scores = torch.zeros(self.metagraph.n).to(self.device)
        self.hotkeys = list(self.metagraph.hotkeys)

        self._load_state()

        # It's important to resync metagraph, there could have been changes since the last save.
        self._resync_metagraph()

    def get_random_miners_uids(self, sample_size=None) -> List[int]:
        miners = [uid for uid, stake in enumerate(self.metagraph.S) if stake < 1.0]
        random.shuffle(miners)
        return miners if sample_size is None else miners[:sample_size]

    def update_scores(self, scores: torch.tensor, miner_uids: List[int]):
        """Updates moving average scores based on the recent received scores"""

        # Check if rewards contains NaN values.
        if torch.isnan(scores).any():
            bt.logging.warning(f"NaN values detected in rewards: {scores}")
            # Replace any NaN values in rewards with 0.
            scores = torch.nan_to_num(scores, 0)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_scores: torch.FloatTensor = self.scores.scatter(
            0, torch.tensor(miner_uids).to(self.device), scores.to(self.device)
        )
        bt.logging.debug(f"Scattered rewards: {scattered_scores}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores = alpha * scattered_scores + (1 - alpha) * self.scores.to(self.device)
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def _save_state(self):
        bt.logging.info("Saving validator state.")

        torch.save(
            {
                "step": self.step,
                "scores": self.scores,
                "hotkeys": self.hotkeys,
            },
            self.config.neuron.full_path + "/state.pt",
        )

    def _load_state(self):
        bt.logging.info("Loading validator state.")

        if not os.path.exists(self.config.neuron.full_path + "/state.pt"):
            bt.logging.warning("No saved state found")
            return

        state = torch.load(self.config.neuron.full_path + "/state.pt")
        self.step = state["step"]
        self.scores = state["scores"].to(self.device)
        self.hotkeys = state["hotkeys"]

    def _run(self):
        try:
            while True:
                self._sync()  # Check that the validator is registered on the network.

                self.loop.run_until_complete(self._concurrent_forward())
        except KeyboardInterrupt:
            self.should_exit.set()
            bt.logging.success("Validator killed by keyboard interrupt.")
        except Exception as e:
            self.should_exit.set()
            bt.logging.exception(f"Error during validation: {e}")
            bt.logging.debug(print_exception(type(e), e, e.__traceback__))

    async def _concurrent_forward(self):
        coroutines = [self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)]
        await asyncio.gather(*coroutines)

    def _sync(self):
        if self._should_sync_metagraph():
            self._resync_metagraph()

        if self._should_set_weights():
            self._set_weights()

        # Always save state.
        self._save_state()

    def _should_set_weights(self) -> bool:
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        # Define appropriate logic for when set weights.
        return (self.block() - self.metagraph.last_update[self.uid]) > self.config.neuron.epoch_length

    def _set_weights(self):
        """
        Sets the validator weights to the metagraph public keys based on scores it has received from the miners.
        The weights determine the trust level and block propagation priority the validator assigns to each miner node.
        Higher weighted miners have higher priority for block propagation checks and incentive rewards from the
        validator's pool.
        """

        if torch.isnan(self.scores).any():
            bt.logging.error(
                "Scores contain NaN values. This may be due to a lack of responses from miners, "
                "or a bug in your reward functions."
            )
            return

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = torch.nn.functional.normalize(self.scores, p=1, dim=0)

        bt.logging.debug(f"Raw weights: {raw_weights}")

        (
            processed_weight_uids,
            processed_weights,
        ) = weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids.to("cpu"),
            weights=raw_weights.to("cpu"),
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        bt.logging.debug(f"Processed weights: {processed_weights}")

        (
            converted_uids,
            converted_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )

        bt.logging.debug(f"Converted weights: {converted_weights}")
        bt.logging.debug(f"Converted uids: {converted_uids}")

        # Set the weights on chain via our subtensor connection.
        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=converted_uids,
            weights=converted_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=__spec_version__,
        )
        if result is True:
            bt.logging.info("Weights set on chain successfully")
        else:
            bt.logging.error("Weights set failed")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """
        Add Validator specific arguments.
        """
        # Netuid Arg: The netuid of the subnet to connect to.
        parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

        parser.add_argument(
            "--neuron.name",
            type=str,
            help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
            default="validator",
        )

        parser.add_argument(
            "--neuron.device",
            type=str,
            help="Device to run on (cpu/cuda:%d).",
            default="cpu",
        )

        parser.add_argument(
            "--neuron.epoch_length",
            type=int,
            help="The default epoch length (how often we set weights, measured in 12 second blocks).",
            default=100,
        )

        parser.add_argument(
            "--neuron.events_retention_size",
            type=str,
            help="Events retention size.",
            default="2 GB",
        )

        parser.add_argument(
            "--neuron.dont_save_events",
            action="store_true",
            help="If set, we dont save events to a log file.",
            default=False,
        )

        parser.add_argument(
            "--neuron.num_concurrent_forwards",
            type=int,
            help="The number of concurrent forwards running at any time.",
            default=1,
        )

        parser.add_argument(
            "--neuron.sample_size",
            type=int,
            help="The number of miners to query in a single step.",
            default=10,
        )

        parser.add_argument(
            "--neuron.disable_set_weights",
            action="store_true",
            help="Disables setting weights.",
            default=False,
        )

        parser.add_argument(
            "--neuron.moving_average_alpha",
            type=float,
            help="Moving average alpha parameter, how much to add of the new observation.",
            default=0.05,
        )

        # Note: the validator needs to serve an Axon with their IP or they may
        #   be blacklisted by the firewall of serving peers on the network.
        parser.add_argument(
            "--neuron.axon_off",
            "--axon_off",
            action="store_true",
            help="Set this flag to not attempt to serve an Axon.",
            default=False,
        )
