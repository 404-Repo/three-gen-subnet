import asyncio
import copy

import bittensor as bt
from common import create_neuron_dir

from miner.metagraph_sync import MetagraphSynchronizer
from miner.validator_selector import ValidatorSelector
from miner.workers import worker_routine


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
    dendrite: bt.dendrite
    """Client module to fetch tasks."""
    metagraph_sync: MetagraphSynchronizer
    """Encapsulates metagraph syncs."""
    validator_selector: ValidatorSelector
    """Encapsulates validator selection."""

    def __init__(self, config: bt.config) -> None:
        self.config: bt.config = copy.deepcopy(config)
        create_neuron_dir(self.config)

        bt.logging(config=config, logging_dir=config.full_path)

        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        self._self_check_for_registration()

        self.metagraph = bt.metagraph(
            netuid=self.config.netuid, network=self.subtensor.network, sync=False, lite=True
        )  # Make sure not to sync without passing the subtensor

        self.metagraph_sync = MetagraphSynchronizer(
            self.metagraph, self.subtensor, self.config.neuron.sync_interval, self.config.neuron.log_info_interval
        )
        self.metagraph_sync.sync()

        bt.logging.info(f"Metagraph: {self.metagraph}")

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet {self.config.netuid} with uid {self.uid} "
            f"using network: {self.subtensor.chain_endpoint}"
        )

        self.validator_selector = ValidatorSelector(self.metagraph, self.config.neuron.min_stake_to_set_weights)

    def _self_check_for_registration(self) -> None:
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            raise RuntimeError(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again."
            )

    async def run(self) -> None:
        bt.logging.debug("Starting the workers.")

        for endpoint in self.config.generation.endpoints:
            asyncio.create_task(worker_routine(endpoint, self.wallet, self.metagraph, self.validator_selector))

        bt.logging.debug("Starting the miner.")

        while True:
            await asyncio.sleep(5)
            self.metagraph_sync.log_info(self.uid)
            self.metagraph_sync.sync()
