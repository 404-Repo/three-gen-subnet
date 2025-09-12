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
    add_public_api_args(parser)
    add_storage_args(parser)
    add_telemetry_args(parser)
    add_task_args(parser)
    add_duels_args(parser)

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
        default=100,  # 100 * 12 seconds ~ 20 minutes
    )
    parser.add_argument(
        "--neuron.min_stake_to_set_weights",
        type=int,
        help="Minimal required stake to set weights.",
        default=1000,
    )
    parser.add_argument(
        "--neuron.log_info_interval",
        type=int,
        help="Logging interval for the node state (in seconds).",
        default=30,
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


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--generation.throttle_period",
        type=int,
        help="Minimum throttle period (in seconds) for task completion.",
        default=30,
    )
    parser.add_argument(
        "--generation.task_cooldown",
        type=int,
        help="Cooldown period between tasks from the same miner (in seconds).",
        default=300,
    )
    parser.add_argument(
        "--generation.cooldown_violation_penalty",
        type=int,
        help="If miner asks for a new task while on a cooldown, additional cooldown is added",
        default=10,
    )
    parser.add_argument(
        "--generation.cooldown_violations_threshold",
        type=int,
        help="Cooldown violation threshold to consider the miner behaviour malicious",
        default=100,
    )

    parser.add_argument(
        "--generation.cooldown_penalty",
        type=int,
        help="Penalty cooldown if miner submits unacceptable results.",
        default=600,
    )
    parser.add_argument(
        "--generation.quality_threshold",
        type=int,
        help="Minimum score required for task results to be accepted by validator.",
        default=0.6,
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
    parser.add_argument(
        "--validation.moving_average_alpha",
        type=float,
        help="Alpha parameter for fidelity factor update.",
        default=0.05,
    )
    parser.add_argument(
        "--validation.validation_lock_duration",
        type=int,
        help="Duration of the validation lock (in seconds).",
        default=30,
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
        "--public_api.strong_miners_count",
        type=int,
        help="Number of top miners that are considered strong",
        default=100,
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


def add_telemetry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--telemetry.disabled",
        action="store_true",
        help="Used to disable validator metrics being sent to the metrics gateway",
        default=False,
    )
    parser.add_argument(
        "--telemetry.push_gateway",
        type=str,
        help="Push gateway for validator metrics",
        default="https://dashboard.404.xyz/metrics",
    )


def add_task_args(parser: argparse.ArgumentParser) -> None:
    # Organic task parameters
    parser.add_argument(
        "--task.organic.assigned_miners_count",
        type=int,
        help="Defines the number of miners that can be assigned to a task",
        default=4,
    )
    parser.add_argument(
        "--task.organic.send_result_timeout",
        type=int,
        help="Maximum wait time in seconds for the next results after the first submission occurred.",
        default=30,
    )
    parser.add_argument(
        "--task.organic.task_timeout",
        type=int,
        help="Time limit for submitting task solutions (in seconds).",
        default=5 * 60,  # 5 minutes
    )

    # Gateway task parameters
    parser.add_argument(
        "--task.gateway.enabled",
        action="store_true",
        help="Enables fetching tasks from the gateway",
        default=False,
    )
    parser.add_argument(
        "--task.gateway.bootstrap_gateway",
        type=str,
        help="Host that is used to collect information about available gateways",
        default="https://gateway-eu.404.xyz:4443",
    )
    parser.add_argument(
        "--task.gateway.task_queue_size",
        type=int,
        help="Maximum number of organic tasks that can be stored in the queue",
        default=50,
    )
    parser.add_argument(
        "--task.gateway.task_fetch_interval",
        type=int,
        help="Interval for periodically fetching tasks from the gateway",
        default=1,
    )

    # Synthetic task parameters
    parser.add_argument(
        "--task.synthetic.default_text_prompts_path",
        type=str,
        help="Path to the file with the default synthetic text prompts (relative or absolute)",
        default="resources/text_prompts.txt",
    )
    parser.add_argument(
        "--task.synthetic.default_image_prompts_path",
        type=str,
        help="Path to the file with the URLS for default synthetic image prompts (relative or absolute)",
        default="resources/image_prompts.txt",
    )
    parser.add_argument(
        "--task.synthetic.text_tasks_ratio",
        type=float,
        help="Ratio of text tasks to total synthetic tasks (0.0 = all image, 1.0 = all text)",
        default=0.8,
    )
    parser.add_argument(
        "--task.synthetic.get_prompts.endpoint",
        type=str,
        help="Base URL for the prompt generation service. Must support POST requests to "
        "/text-prompts/get/ and /image-prompts/get/ routes for fetching synthetic prompt batches.",
        default="http://44.219.222.104:9100",
    )
    parser.add_argument(
        "--task.synthetic.get_prompts.text_fetch_interval",
        type=int,
        help="Interval in seconds between fetching text prompt batches (100k prompts per batch). "
        "Default: 30 minutes.",
        default=30 * 60,
    )
    parser.add_argument(
        "--task.synthetic.get_prompts.image_fetch_interval",
        type=int,
        help="Interval in seconds between fetching image prompt batches (10k prompts per batch). "
        "Default: 10 minutes.",
        default=10 * 60,
    )
    parser.add_argument(
        "--task.synthetic.get_prompts.initial_delay",
        type=int,
        help="Delay in seconds before the first fetch of synthetic prompts.",
        default=30,
    )


def add_duels_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--duels.disabled",
        action="store_true",
        help="Disable the duels rating system",
        default=False,
    )
    parser.add_argument(
        "--duels.start_delay",
        type=int,
        help="Delay in seconds before starting duels to collect active miner information",
        default=1800,  # 30 minutes
    )
    parser.add_argument(
        "--duels.inactivity_time",
        type=int,
        help="Time in seconds after which inactive miners are excluded from duels.",
        default=1200,  # 20 minutes
    )
    parser.add_argument(
        "--duels.judge_workers",
        type=int,
        help="Number of concurrent workers for judging duels",
        default=1,
    )
    parser.add_argument(
        "--duels.duels_per_minute",
        type=float,
        help="Maximum duel requests per minute (rate limiting for pay-per-request billing)",
        default=6,
    )
    parser.add_argument(
        "--duels.no_pause_on_slow_validation",
        action="store_false",
        dest="duels.pause_duels_on_slow_validation",
        help="Disable automatic stopping of duels when validation becomes slow.",
        default=True,
    )
    parser.add_argument(
        "--duels.slow_validation_threshold",
        type=float,
        help="Validation time threshold in seconds above which duels are stopped.",
        default=10.0,
    )
    parser.add_argument(
        "--duels.judge_endpoint",
        type=str,
        help="URL of the judge service endpoint (should accept POST requests to /)",
        default="http://127.0.0.1:8095/v1/",
    )
    parser.add_argument(
        "--duels.judge_api_key",
        type=str,
        help="API key for authentication with the judge service",
        default="local",
    )
    parser.add_argument(
        "--duels.duel_saver_endpoint",
        type=str,
        help="URL of the duel results storage endpoint (should accept POST requests to /)",
        default="http://35.246.174.235:4001/api/save_duel/",
    )
