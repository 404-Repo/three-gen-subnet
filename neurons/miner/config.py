import argparse

import bittensor as bt


def read_config() -> bt.config:
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.axon.add_args(parser)

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=29)

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Name of the neuron, used to determine the neuron directory",
        default="miner",
    )

    parser.add_argument("--neuron.sync_interval", type=int, help="Metagraph sync interval, seconds", default=30 * 60)

    parser.add_argument(
        "--neuron.log_info_interval",
        type=int,
        help="Logging interval for the node state (in seconds).",
        default=30,
    )

    parser.add_argument(
        "--neuron.min_stake_to_set_weights",
        type=int,
        help="Minimal required stake to set weights.",
        default=1000,
    )

    parser.add_argument(
        "--generation.endpoints",
        "--generation.endpoint",
        type=str,
        nargs="*",
        help="Specifies the URL of the endpoint responsible for generating 3D assets. "
        "This endpoint should handle the /generation/ POST route.",
        default=["http://127.0.0.1:8093"],
    )

    return bt.config(parser)
