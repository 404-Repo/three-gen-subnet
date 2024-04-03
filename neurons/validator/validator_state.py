from pathlib import Path

import bittensor as bt
from pydantic import BaseModel

from validator.miner_data import MinerData


class ValidatorState(BaseModel):
    miners: list[MinerData]

    def save(self, path: Path) -> None:
        try:
            with path.open("w") as f:
                f.write(self.json())
        except Exception as e:
            bt.logging.exception(f"Validator state saving failed with {e}")

    def load(self, path: Path) -> None:
        if not path.exists():
            bt.logging.warning("No saved state found")
            return

        try:
            with path.open("r") as f:
                content = f.read()
            self.parse_raw(content)
        except Exception as e:
            bt.logging.exception(f"Failed to load the state: {e}")

        bt.logging.info("Validator state loaded.")
