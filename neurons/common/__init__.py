import os

import bittensor as bt


def create_neuron_dir(config: bt.config) -> None:
    full_path = os.path.expanduser(
        f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/"
        f"netuid{config.netuid}/{config.neuron.name}"
    )
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)
