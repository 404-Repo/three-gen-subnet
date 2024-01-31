import time
import typing

import bittensor as bt

from mining import load_models, TextTo3DModels, forward
from neurons import protocol
from neurons.base_miner import BaseMinerNeuron
from neurons.config import read_config


class Miner(BaseMinerNeuron):
    models: TextTo3DModels

    def __init__(self, config: bt.config):
        super(Miner, self).__init__(config=config)

        self.models = load_models(config.neuron.device, config.neuron.full_path)

    async def forward(self, synapse: protocol.TextTo3D) -> protocol.TextTo3D:
        bt.logging.debug(f"Text-to-3D task received: {synapse.prompt_in}")
        forward(synapse, self.models)
        return synapse

    async def blacklist(self, synapse: protocol.TextTo3D) -> typing.Tuple[bool, str]:
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            self.config.blacklist.force_validator_permit
            and not self.metagraph.validator_permit[uid]
        ):
            bt.logging.debug(
                f"Blacklisting validator without the permit {synapse.dendrite.hotkey}"
            )
            return True, "No validator permit"

        if self.metagraph.S[uid] < self.config.blacklist.min_stake:
            bt.logging.debug(
                f"Blacklisting - not enough stake {synapse.dendrite.hotkey} "
                f"with {self.metagraph.S[uid]} TAO "
            )
            return True, "No validator permit"

        return False, "OK"

    async def priority(self, synapse: protocol.TextTo3D) -> float:
        try:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        except ValueError:
            bt.logging.error(
                f"Unregistered caller for priority calculation: {synapse.dendrite.hotkey}"
            )
            return 0.0

        return float(self.metagraph.S[uid])


def main():
    config = read_config(Miner)
    bt.logging.info(f"Starting with config: {config}")

    with Miner(config) as miner:
        pass
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(60)

            if miner.should_exit.is_set():
                bt.logging.debug("Stopping the validator")
                break


if __name__ == "__main__":
    main()
