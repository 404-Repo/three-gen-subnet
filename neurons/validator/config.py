import argparse

import bittensor as bt


def read_config() -> bt.config:
    parser = _build_parser()
    return bt.config(parser)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.axon.add_args(parser)

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=29)

    add_neuron_args(parser)
    add_generation_args(parser)
    add_validation_args(parser)
    add_dataset_args(parser)
    add_public_api_args(parser)
    add_storage_args(parser)

    return parser


def add_neuron_args(parser: argparse.ArgumentParser) -> None:
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


def add_generation_args(parser: argparse.ArgumentParser) -> None:
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
        default=600,
    )


def add_validation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--validation.endpoints",
        "--validation.endpoint",
        type=str,
        nargs="*",
        help="Specifies the URL of the endpoint responsible for scoring 3D assets. "
        "This endpoint should handle the /validate/ POST route.",
        default=["http://127.0.0.1:8094"],
    )


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
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


def add_public_api_args(parser: argparse.ArgumentParser) -> None:
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


def add_storage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--storage.enabled",
        action="store_true",
        help="Enables storing 3D assets that exceed the validation score threshold",
        default=False,
    )
    parser.add_argument(
        "--storage.endpoint_url",
        type=str,
        help="Base URL for the storage service (must handle /store/ POST route)",
        default="http://127.0.0.1:8100",
    )
    parser.add_argument(
        "--storage.service_api_key",
        type=str,
        help="API key for storage service authentication (min 8 alphanumeric chars)",
        default="",
    )
    parser.add_argument(
        "--storage.validation_score_threshold",
        type=float,
        help="Minimum validation score required for storing 3D assets (0.0 to 1.0)",
        default=0.6,
    )
