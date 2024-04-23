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
        help="Identifies the neuron for directory and logging purposes.",
        default="validator",
    )

    parser.add_argument(
        "--neuron.sync_interval",
        type=int,
        help="Interval for synchronizing with the Metagraph (in seconds).",
        default=10 * 60,  # 10 minutes
    )

    parser.add_argument(
        "--neuron.weight_set_interval",
        type=int,
        help="Weight set interval, measured in blocks.",
        default=100,
    )

    parser.add_argument(
        "--neuron.min_stake_to_set_weights",
        type=int,
        help="Minimal required stake to set weights.",
        default=1000,
    )

    # parser.add_argument("--neuron.sync_interval", type=int, help="Metagraph sync interval, seconds", default=20)
    #
    # parser.add_argument(
    #     "--neuron.weight_set_interval",
    #     type=int,
    #     help="Weight set interval, measured in blocks.",
    #     default=5,
    # )
    #
    # parser.add_argument(
    #     "--neuron.min_stake_to_set_weights",
    #     type=int,
    #     help="Minimal required stake to set weights.",
    #     default=10,
    # )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Alpha parameter for updating score using moving average.",
        default=0.05,
    )

    parser.add_argument(
        "--neuron.log_info_interval",
        type=int,
        help="Logging interval for the node state (in seconds).",
        default=30,
    )

    parser.add_argument(
        "--neuron.strong_miners_count",
        type=int,
        help="Number of top miners that are considered strong.",
        default=100,
    )

    parser.add_argument(
        "--public_api.enabled",
        action="store_true",
        help="Enables public API access for whitelisted wallets.",
        default=False,
    )

    parser.add_argument(
        "--public_api.whitelist_disabled",
        action="store_true",
        help="Disables wallet whitelisting, allowing any wallet to access.",
        default=False,
    )

    parser.add_argument(
        "--public_api.polling_interval",
        type=int,
        help="Minimum interval between task polling (in seconds).",
        default=30,
    )

    parser.add_argument(
        "--public_api.rate_limit.requests",
        type=int,
        help="The maximum number of requests that are allowed for a single key within the specified time period.",
        default=10,
    )

    parser.add_argument(
        "--public_api.rate_limit.period",
        type=int,
        help="The time period, in seconds, for which the request limit (specified by max_requests) applies",
        default=120,
    )

    parser.add_argument(
        "--public_api.copies",
        type=int,
        help="Number of copies to generate to chose the best from.",
        default=4,
    )

    parser.add_argument(
        "--public_api.wait_after_first_copy",
        type=int,
        help="Maximum wait time for the second copy after the first acceptable copy was generated.",
        default=60,
    )

    parser.add_argument(
        "--generation.task_timeout",
        type=int,
        help="Time limit for submitting tasks (in seconds).",
        default=10 * 60,
    )

    parser.add_argument(
        "--generation.task_cooldown",
        type=int,
        help="Cooldown period between tasks from the same miner (in seconds).",
        default=60,
    )

    parser.add_argument(
        "--validation.endpoint",
        type=str,
        help="Specifies the URL of the endpoint responsible for scoring 3D assets. "
        "This endpoint should handle the /validate/ POST route.",
        default="http://127.0.0.1:8094",
    )

    parser.add_argument(
        "--dataset.path",
        type=str,
        help="Path to the file with the prompts (relative or absolute)",
        default="resources/prompts.txt",
    )

    return bt.config(parser)
