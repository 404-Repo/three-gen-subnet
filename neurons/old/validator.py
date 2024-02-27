import argparse
import os
import os.path
import pandas
import requests
import time
import random

import bittensor as bt
from old.config import read_config
from validating import Validate3DModels, load_models, score_responses
from protocol import TextTo3D

from old.base_validator import BaseValidatorNeuron


class Validator(BaseValidatorNeuron):
    models: Validate3DModels
    dataset: list[str]

    def __init__(self, config: bt.config):
        super(Validator, self).__init__(config)

        if self.config.neuron.opengl_platform in ("egl", "osmesa"):
            os.environ["PYOPENGL_PLATFORM"] = self.config.neuron.opengl_platform

        self.models = load_models(self.device, config.neuron.full_path)
        self.dataset = self.load_dataset()

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        miner_uids = self.get_random_miners_uids(self.config.neuron.sample_size)

        prompt = random.choice(self.dataset)

        bt.logging.debug(f"Sending prompt: {prompt}")

        responses = await self.dendrite.forward(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=TextTo3D(prompt_in=prompt),
            deserialize=False,
            timeout=30,
        )

        n = len([r for r in responses if r.mesh_out is not None])

        bt.logging.info(f"Received {n} responses")

        if n == 0:
            return

        scores = score_responses(prompt, responses, self.device, self.models)

        bt.logging.info(f"Scored responses: {scores}")

        self.update_scores(scores, miner_uids)

    def load_dataset(self) -> list[str]:
        dataset_path = f"{self.config.neuron.full_path}/dataset.csv"
        if not os.path.exists(dataset_path):
            bt.logging.info(f"Downloading dataset from {self.config.neuron.dataset_url}")

            with requests.get(self.config.neuron.dataset_url, stream=True) as r:
                r.raise_for_status()
                with open(dataset_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            bt.logging.info("Dataset downloaded successfully")

        bt.logging.info("Loading the dataset")
        dt = pandas.read_csv(dataset_path, header=None, usecols=[1])
        return dt[1].to_list()

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        BaseValidatorNeuron.add_args(parser)

        parser.add_argument(
            "--neuron.opengl_platform",
            type=str,
            help="Pyrender backend (pyglet, egl, osmesa).",
            default="egl",
        )

        parser.add_argument(
            "--neuron.dataset_url",
            type=str,
            help="URL to the dataset with prompts",
            default="https://huggingface.co/datasets/tiange/Cap3D/resolve/main/Cap3D_automated_Objaverse_no3Dword.csv",
        )


def main():
    config = read_config(Validator)
    bt.logging.info(f"Starting with config: {config}")

    with Validator(config) as validator:
        while True:
            bt.logging.debug("Validator running...", time.time())
            time.sleep(60)

            if validator.should_exit.is_set():
                bt.logging.debug("Stopping the validator")
                break


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    main()
