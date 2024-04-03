import random
from pathlib import Path

import bittensor as bt


class Dataset:
    def __init__(self, path: str):
        dataset_path = Path(path)
        if not dataset_path.is_absolute():
            dataset_path = Path(__file__).parent / ".." / ".." / dataset_path
        else:
            dataset_path = Path(path)

        if not dataset_path.exists():
            raise RuntimeError(f"Dataset file {dataset_path} not found")

        with dataset_path.open() as f:
            self.prompts = f.read().strip().split("\n")

        bt.logging.info(f"{len(self.prompts)} prompts loaded")

    def get_random_prompt(self) -> str:
        prompt = random.choice(self.prompts)  # noqa # nosec
        return prompt
