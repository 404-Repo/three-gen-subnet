import argparse
import asyncio
import copy
import random as rd
from uuid import uuid4

import bittensor as bt
from common.protocol import SubmitResults
from mocks.mock_metagraph import MockMetagraph
from validator.config import config
from validator.miner_data import MinerData
from validator.task_manager.task_manager import task_manager
from validator.validation_service import ValidationResponse, ValidationService
from validator.validator import Validator


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
        task_manager,
        validation_service,
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
        self.metagraph = MockMetagraph(self.metagraph)

        # Clear miners
        self.miners.clear()

        # Change wallet to the main net wallet in order to pass verification in gateway
        task_manager.wallet = gateway_wallet

    def _get_neuron_uid(self, hotkey: str) -> int | None:
        if hotkey in self.metagraph.hotkeys:
            return self.metagraph.hotkeys.index(hotkey)

        # Create hotkey if it doesn't exist
        self.metagraph.hotkeys.append(hotkey)
        self.metagraph.S.append(20000)
        self.metagraph.coldkeys.append(str(uuid4()))
        self.miners.append(MinerData(uid=len(self.metagraph.hotkeys)))
        return len(self.metagraph.hotkeys)

    def _set_weights(self) -> None:
        return


async def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MockValidator with gateway wallet configuration")
    parser.add_argument(
        "--gateway-wallet-name", type=str, required=True, help="Name of the wallet to use for gateway authentication"
    )
    parser.add_argument("--gateway-wallet-hotkey", type=str, required=True, help="Hotkey name for the gateway wallet")

    args = parser.parse_args()

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
    await neuron.run()


if __name__ == "__main__":
    asyncio.run(main())
