import os
import random
import bittensor as bt
from pathlib import Path


class Dataset:
    def __init__(self, path: str):
        if not os.path.isabs(path):
            path = Path(__file__).parent / ".." / ".." / path

        if not os.path.exists(path):
            raise RuntimeError(f"Dataset file {path} not found")

        with open(path) as f:
            self.prompts = f.read().strip().split("\n")

        bt.logging.info(f"{len(self.prompts)} prompts loaded")

    def get_random_prompt(self) -> str:
        return random.choice(self.prompts)
