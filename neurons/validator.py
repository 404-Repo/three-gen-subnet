import time
from typing import List

import bittensor as bt
import torch
from config import read_config
from protocol import Task404

from neurons.base_validator import BaseValidatorNeuron


class Validator(BaseValidatorNeuron):
    def __init__(self, config: bt.config):
        super(Validator, self).__init__(config)

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

        # TODO: move probe creation to a separate module
        responses = await self.dendrite.forward(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=Task404(dummy_input=self.step),
            deserialize=False,
        )

        bt.logging.info(f"Received responses: {responses}")

        # TODO: move scoring to a separate module
        scores = self.score_responses(responses=responses)

        bt.logging.info(f"Scored responses: {scores}")

        self.update_scores(scores, miner_uids)

    def score_response(self, synapse: Task404) -> float:
        if synapse.dummy_output is None:
            return 0.0
        return 1.0 if synapse.dummy_output == synapse.dummy_input * 2 else 0.0

    def score_responses(self, responses: List[Task404]) -> torch.FloatTensor:
        return torch.FloatTensor(
            [self.score_response(synapse) for synapse in responses]
        ).to(self.device)


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
