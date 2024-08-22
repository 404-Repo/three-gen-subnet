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
        default=100,  # 100 * 12 seconds = 20 minutes
    )

    parser.add_argument(
        "--neuron.min_stake_to_set_weights",
        type=int,
        help="Minimal required stake to set weights.",
        default=1000,
    )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Alpha parameter for fidelity factor update.",
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
        help="Number of top miners that are considered strong. Used for Public API",
        default=100,
    )

    parser.add_argument(
        "--neuron.auto_update_disabled",
        action="store_true",
        help="Disables neuron auto-update.",
        default=False,
    )

    parser.add_argument(
        "--neuron.auto_update_interval",
        type=int,
        help="Version check interval",
        default=30 * 60,  # 30 minutes
    )

    parser.add_argument(
        "--neuron.cooldown_violation_penalty",
        type=int,
        help="If miner asks for a new task while on a cooldown, additional cooldown is added",
        default=10,
    )

    parser.add_argument(
        "--neuron.cooldown_violations_threshold",
        type=int,
        help="Cooldown violation threshold to consider the miner behaviour malicious",
        default=100,
    )

    parser.add_argument(
        "--generation.task_timeout",
        type=int,
        help="Time limit for submitting tasks (in seconds).",
        default=10 * 60,  # 10 minutes
    )

    parser.add_argument(
        "--generation.task_cooldown",
        type=int,
        help="Cooldown period between tasks from the same miner (in seconds).",
        default=60,
    )

    parser.add_argument(
        "--generation.cooldown_penalty",
        type=int,
        help="Penalty cooldown if miner submits unacceptable results.",
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
        "--dataset.default_prompts_path",
        type=str,
        help="Path to the file with the default prompts (relative or absolute)",
        default="resources/prompts.txt",
    )

    parser.add_argument(
        "--dataset.prompter.endpoint",
        type=str,
        help="Specifies the URL of the endpoint responsible for providing fresh batches of prompts. "
        "This endpoint should handle the /get/ GET route.",
        default="http://35.222.204.141:9100",
    )

    parser.add_argument(
        "--dataset.prompter.fetch_interval",
        type=int,
        help="Defines the fetch interval. The prompt batch is quite big (100k+) prompts. No need to fetch frequently",
        default=60 * 60,  # one hour
    )

    parser.add_argument(
        "--public_api.enabled",
        action="store_true",
        help="Enables public API access for whitelisted wallets.",
        default=False,
    )

    parser.add_argument(
        "--public_api.server_port",
        type=int,
        help="The local port public api endpoint is bound to. i.e. 8888",
        default=8888,
    )

    parser.add_argument(
        "--public_api.copies",
        type=int,
        help="Number of copies to generate to chose the best from. Only one copy is sent back to the client.",
        default=4,
    )

    parser.add_argument(
        "--public_api.wait_after_first_copy",
        type=int,
        help="Maximum wait time for the second copy after the first acceptable copy was generated.",
        default=30,
    )

    parser.add_argument(
        "--public_api.sync_api_keys_interval",
        type=int,
        help="Defines the interval at which the api keys are re-fetched from the database.",
        default=30 * 60,  # one hour
    )

    parser.add_argument(
        "--storage.enabled",
        action="store_true",
        help="If enabled, generated assets are stored on the Bittensor storage subnet (SN21).",
        default=False,
    )

    parser.add_argument("--storage.netuid", type=int, help="Storage subnet netuid.", default=21)

    parser.add_argument(
        "--storage.testnet", action="store_true", help="Set if testnet is used for storage", default=False
    )

    parser.add_argument(
        "--storage.validator.hotkey",
        type=str,
        help="Storage subnet validator to use for storing",
        default="",
    )

    parser.add_argument(
        "--storage.wallet.name",
        type=str,
        help="Wallet to use for assets storing on storage subnet",
        default=bt.defaults.wallet.name,
    )

    parser.add_argument(
        "--storage.wallet.hotkey",
        type=str,
        help="Wallet to use for assets storing on storage subnet",
        default=bt.defaults.wallet.hotkey,
    )

    parser.add_argument(
        "--storage.queue_size",
        type=int,
        help="Maximum, concurrent number of generated assets waiting in queue to be saved.",
        default=256,
    )

    parser.add_argument(
        "--storage.ttl",
        type=int,
        help="Assets storing time.",
        default=60 * 60 * 24 * 30 * 12,  # almost 1 year
    )
    # We expect to significantly increase the assets quality. Storing for one year only.

    return bt.config(parser)
