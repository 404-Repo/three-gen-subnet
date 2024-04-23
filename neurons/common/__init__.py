from pathlib import Path

import bittensor as bt


def create_neuron_dir(config: bt.config) -> None:
    full_path = (
        Path(config.logging.logging_dir)
        / config.wallet.name
        / config.wallet.hotkey
        / f"netuid{config.netuid}"
        / config.neuron.name
    ).expanduser()

    config.neuron.full_path = full_path
    if not full_path.exists():
        full_path.mkdir(exist_ok=True, parents=True)
