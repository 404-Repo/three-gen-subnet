import asyncio
import copy
import random as rd
from uuid import uuid4

import anyio
import bittensor as bt
from common.protocol import SubmitResults
from validator.config import _build_parser
from validator.miner_data import MinerData
from validator.task_manager.task_manager import TaskManager, task_manager
from validator.validation_service import ValidationResponse, ValidationService
from validator.validator import Validator


class MockMetagraph:
    """Mock Metagraph wrapper around bt.metagraph with additional properties for testing."""

    def __init__(self, metagraph: bt.metagraph) -> None:
        """
        Initialize MockMetagraph with an existing bt.metagraph instance.
        """
        self._metagraph = metagraph
        self.hotkeys: list[str] = []
        self.coldkeys: list[str] = []
        self.S: list[int] = []


class MockValidation(ValidationService):
    """Mock validation service that returns uniform scores between 0.5 and 0.99."""

    def __init__(self, min_score: float = 0.5, max_score: float = 0.99) -> None:
        self.min_score = min_score
        self.max_score = max_score
        bt.logging.info(f"MockValidation initialized with score range [{min_score}, {max_score}]")

    async def validate(self, *, synapse: SubmitResults, neuron_uid: int) -> ValidationResponse:
        score = rd.uniform(self.min_score, self.max_score)  # noqa: S311 # nosec: B311
        return ValidationResponse(score=score)

    async def version(self) -> str:
        return "mock-validation-1.0.0"

    def is_available(self) -> bool:
        return True


class MockValidator(Validator):
    """Mock Validator that accepts arbitrary hotkeys not registered on the network and validates
    miner results randomly."""

    def __init__(
        self,
        *,
        config: bt.config,
        task_manager: TaskManager,
        validation_service: ValidationService,
        gateway_wallet: bt.wallet,
        wallet: bt.wallet | None = None,
        subtensor: bt.subtensor | None = None,
    ) -> None:
        # Store the allowed hotkeys for mock miners
        self.hotkeys: list[str] = []
        self.wallet_for_gateway: bt.wallet = gateway_wallet

        # Initialize the parent Validator
        super().__init__(
            config=config,
            task_manager=task_manager,
            validation_service=validation_service,
            wallet=wallet,
            subtensor=subtensor,
        )

        # Override the metagraph with a mock metagraph
        self.metagraph: MockMetagraph = MockMetagraph(self.metagraph)

        # Override the metagraph with a mock metagraph
        self.metagraph = MockMetagraph(self.metagraph)

        # Clear miners
        self.miners.clear()

        # Change wallet to the main net wallet in order to pass verification in gateway
        task_manager._organic_task_storage._wallet = gateway_wallet
        task_manager._synthetic_task_storage._wallet = gateway_wallet

    def _get_neuron_uid(self, hotkey: str) -> int | None:
        if hotkey in self.metagraph.hotkeys:
            return self.metagraph.hotkeys.index(hotkey)

        # Create hotkey if it doesn't exist
        self.metagraph.hotkeys.append(hotkey)
        self.metagraph.S.append(20000)
        self.metagraph.coldkeys.append(str(uuid4()))
        self.miners.append(MinerData(uid=len(self.metagraph.hotkeys) - 1))
        return len(self.metagraph.hotkeys) - 1

    def _set_weights(self) -> None:
        return

    def _check_miner_signature(self, *, synapse: SubmitResults, miner: MinerData) -> bool:
        return True

    async def run(self) -> None:
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        bt.logging.info(
            f"Serving validator axon {self.axon} on network: {self.config.subtensor.chain_endpoint} "
            f"with netuid: {self.config.netuid}"
        )

        if self.public_server is not None:
            self.public_server.start()

        bt.logging.debug("Starting the validator.")

        event = anyio.Event()
        await event.wait()


async def main() -> None:
    # Parse command line arguments
    parser = _build_parser()
    parser.add_argument(
        "--gateway_wallet_name", type=str, required=True, help="Name of the wallet to use for gateway authentication"
    )
    parser.add_argument("--gateway_wallet_hotkey", type=str, required=True, help="Hotkey name for the gateway wallet")

    args = parser.parse_args()
    config = bt.config(parser)

    # Create gateway wallet using the parsed arguments
    config_copy = copy.deepcopy(config)
    config_copy.wallet.name = args.gateway_wallet_name
    config_copy.wallet.hotkey = args.gateway_wallet_hotkey
    gateway_wallet = bt.wallet(config=config_copy)

    neuron = MockValidator(
        config=config,
        task_manager=task_manager,
        validation_service=MockValidation(),
        gateway_wallet=gateway_wallet,
    )
    task_manager._organic_task_storage._wallet = gateway_wallet
    task_manager._synthetic_task_storage._wallet = gateway_wallet
    asyncio.create_task(neuron.task_manager._synthetic_task_storage.fetch_synthetic_tasks_cron())
    asyncio.create_task(neuron.task_manager._organic_task_storage.fetch_gateway_tasks_cron())
    await neuron.run()


if __name__ == "__main__":
    asyncio.run(main())
