import abc
import argparse
import copy
import threading
import time
import typing
from abc import ABC

import bittensor as bt
import bittensor.utils.weight_utils
import torch
from config import check_config
from version import __spec_version__

from neurons import protocol


class BaseMinerNeuron(ABC):
    config: bt.config
    """Copy of the original config."""
    uid: int
    """Each miner gets a unique identity (UID) in the network for differentiation."""
    device: torch.device
    """Device to run computations on."""
    wallet: bt.wallet
    """The wallet holds the cryptographic key pairs for the miner."""
    subtensor: bt.subtensor
    """The subtensor is our connection to the Bittensor blockchain."""
    metagraph: bt.metagraph
    """The metagraph holds the state of the network, letting us know about other validators and miners."""
    axon: bt.axon | None = None
    """Axon for external connections."""

    is_background_thread_running: bool = False
    """Indicates whether background thread is running."""
    should_stop_background_thread: bool = False
    """When set background thread is supposed to stop."""
    should_exit: threading.Event
    """Set in the background thread on errors. Should lead to the application stop."""
    thread: threading.Thread | None = None
    """Background thread."""

    _cached_block: int = 0
    _cached_block_time: float = 0.0
    _cached_block_ttl: float = 12.0
    """Not to request the current block on each validator tick, it's value is cached for the next `ttl` seconds."""

    def __init__(self, config: bt.config):
        self.config: bt.config = copy.deepcopy(config)
        check_config(config)
        bt.logging(config=config, logging_dir=config.full_path)

        self.device = torch.device(self.config.neuron.device)
        if (
            self.device.type.lower().startswith("cuda")
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                f"{self.device.type} device is selected while CUDA is not available"
            )

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

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet {self.config.netuid} with uid {self.uid} "
            f"using network: {self.subtensor.chain_endpoint}"
        )

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )

        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        self.axon = bt.axon(wallet=self.wallet, port=self.config.axon.port)

        bt.logging.info(f"Attaching forward function to the miner axon.")

        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

        bt.logging.info(f"Axon created: {self.axon}")

        self.should_exit = threading.Event()

    def __enter__(self):
        self._run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_background_thread()

    @abc.abstractmethod
    def forward(self, synapse: protocol.Task404) -> protocol.Task404:
        ...

    @abc.abstractmethod
    def blacklist(self, synapse: protocol.Task404) -> typing.Tuple[bool, str]:
        ...

    @abc.abstractmethod
    def priority(self, synapse: protocol.Task404) -> float:
        ...

    def block(self):
        cur_time = time.time()
        if cur_time > self._cached_block_time + self._cached_block_ttl:
            self._cached_block = self.subtensor.get_current_block()
            self._cached_block_time = cur_time

        return self._cached_block

    def _run_in_background_thread(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if self.is_background_thread_running:
            return
        bt.logging.debug("Starting miner in a background thread.")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.should_stop_background_thread = False
        self.is_background_thread_running = True
        bt.logging.debug("Background thread started.")

    def _stop_background_thread(self):
        """
        Stops the miner's background operations.
        """
        if not self.is_background_thread_running:
            return
        bt.logging.debug("Stopping the miner background thread.")
        self.should_stop_background_thread = True
        self.thread.join(30)
        self.is_background_thread_running = False
        if self.axon is not None:
            self.axon.stop()
        bt.logging.debug("Background thread stopped.")

    def _run(self):
        """
        Manages the Bittensor miner's main loop, which ensures an orderly shutdown during keyboard interrupts
        and logs unexpected errors.

        Primary responsibilities include:
        1. Verifying Bittensor network registration.
        2. Initiating the miner's axon to participate in the network.
        3. Updating the metagraph periodically to sync with the latest chain state and adjust weights.

        Operation continues until 'should_stop_background_thread' is set to True or an external
        interruption is received. In each epoch, the miner:
        - Monitors for new network blocks,
        - Refreshes its view of the network via the metagraph,
        - Updates its weights to reflect the current network dynamics,
        ensuring the miner's active and current standing within the network.

        Notes:
        - It relies on global configuration from miner initialization.
        - The miner's axon serves as the point of contact for network communication.

        Raises:
        - KeyboardInterrupt: On manual intervention.
        - Exception: For other unexpected issues during operation, with logs captured for debugging.
        """

        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} "
            f"with netuid: {self.config.netuid}"
        )

        bt.logging.info(f"Miner starting at block: {self.block()}")

        try:
            while True:
                self._sync()

                if self.should_stop_background_thread:
                    break

                time.sleep(10)
        except KeyboardInterrupt:
            self.should_exit.set()
            bt.logging.success("Miner killed by keyboard interrupt.")
        except Exception as e:
            self.should_exit.set()
            bt.logging.exception(f"Error during validation: {e}")

    def _check_for_registration(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            raise RuntimeError(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again."
            )

    def _sync(self):
        # Ensure miner or validator hotkey is still registered on the network.
        self._check_for_registration()

        if self._should_sync_metagraph():
            self._resync_metagraph()

        if self._should_set_weights():
            self._set_weights()

    def _should_sync_metagraph(self) -> bool:
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (
            self.block() - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def _resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""

        bt.logging.info("Resyncing metagraph")

        self.metagraph.sync(subtensor=self.subtensor)

    def _should_set_weights(self) -> bool:
        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        # Define appropriate logic for when set weights.
        return (
            self.block() - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length

    def _set_weights(self):
        """
        Self-assigns a weight of 1 to the current miner (identified by its UID) and
        a weight of 0 to all other peers in the network.
        The weights determine the trust level the miner assigns to other nodes on the network.
        """
        weights = torch.zeros(self.metagraph.n, dtype=torch.float)
        weights[self.uid] = 1.0

        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids.to("cpu"),
            weights=weights.to("cpu"),
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

        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=converted_uids,
            weights=converted_weights,
            wait_for_finalization=False,
            wait_for_inclusion=True,
            version_key=__spec_version__,
        )

        if result is True:
            bt.logging.info("Weights set on chain successfully")
        else:
            bt.logging.error("Weights set failed")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """
        Add Miner specific arguments.
        """
        # Netuid Arg: The netuid of the subnet to connect to.
        parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

        parser.add_argument(
            "--neuron.name",
            type=str,
            help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
            default="miner",
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
            "--blacklist.force_validator_permit",
            action="store_true",
            help="If set, we will force incoming requests to have a permit.",
            default=False,
        )

        parser.add_argument(
            "--blacklist.min_stake",
            type=int,
            help="Minimal validator stake to send requests",
            default=0,
        )
