import os.path
import pandas
import requests
import time
import random

import bittensor as bt
from config import read_config
from validating import Validate3DModels, load_models, score_responses
from protocol import TextTo3D

from neurons.base_validator import BaseValidatorNeuron


class Validator(BaseValidatorNeuron):
    models: Validate3DModels
    dataset: list[str]

    def __init__(self, config: bt.config):
        super(Validator, self).__init__(config)

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

        bt.logging.info(
            f"Received {len([r for r in responses if r.mesh_out is not None])} responses"
        )

        scores = score_responses(prompt, responses, self.device, self.models)

        bt.logging.info(f"Scored responses: {scores}")

        self.update_scores(scores, miner_uids)

    def load_dataset(self) -> list[str]:
        dataset_path = f"{self.config.neuron.full_path}/dataset.csv"
        if not os.path.exists(dataset_path):
            bt.logging.info(
                f"Downloading dataset from {self.config.neuron.dataset_url}"
            )

            with requests.get(self.config.neuron.dataset_url, stream=True) as r:
                r.raise_for_status()
                with open(dataset_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            bt.logging.info(f"Dataset downloaded successfully")

        bt.logging.info(f"Loading the dataset")
        dt = pandas.read_csv(dataset_path, header=None, usecols=[1])
        return dt[1].to_list()


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
