import asyncio
import copy
import hashlib
import json
import secrets
import shutil
import time
import typing
from base64 import b64encode
from pathlib import Path
from typing import Any
from uuid import uuid4

import base58
import bittensor as bt
from common.owner import HOTKEY
from common.protocol import Feedback, PullTask, SubmitResults, Task
from mnemonic import Mnemonic
from validator.config import _build_parser


MINER_NAME = "miner.mock"


def generate_fake_hotkey_full() -> dict[str, Any]:
    """Generate a fake hotkey and save it to a json file."""
    # Generate mnemonic and seed
    mnemo = Mnemonic("english")
    secret_phrase = mnemo.generate(strength=256)
    seed_bytes = mnemo.to_seed(secret_phrase)
    secret_seed = "0x" + seed_bytes.hex()

    # Generate fake private/public key pair
    private_key = secrets.token_bytes(64)
    public_key = hashlib.sha256(private_key).digest()

    private_key_hex = "0x" + private_key.hex()
    public_key_hex = "0x" + public_key.hex()

    # SS58 address using 42 prefix (0x2a)
    ss58_address = base58.b58encode_check(b"\x2a" + public_key[:32]).decode("utf-8")
    # Return full hotkey structure
    return {
        "accountId": public_key_hex,
        "publicKey": public_key_hex,
        "privateKey": private_key_hex,
        "secretPhrase": secret_phrase,
        "secretSeed": secret_seed,
        "ss58Address": ss58_address,
    }


def save_to_json_file(data: dict[str, Any], hotkey: str) -> Path:
    """Save the hotkey to a json file in the mock wallet directory."""
    dir_path = Path(f"~/.bittensor/wallets/{MINER_NAME}/hotkeys").expanduser()
    dir_path.mkdir(parents=True, exist_ok=True)
    filepath = dir_path / hotkey
    with filepath.open("w") as f:
        json.dump(data, f, indent=4)
    bt.logging.info(f"Created hotkey at {filepath}")
    return filepath


def cleanup_mock_wallets() -> None:
    """Clean up all files in the mock wallet directory and remove the directory."""
    wallet_dir = Path(f"~/.bittensor/wallets/{MINER_NAME}").expanduser()

    if wallet_dir.exists():
        bt.logging.info(f"Cleaning up mock wallet directory: {wallet_dir}")
        try:
            # Remove all files and subdirectories
            shutil.rmtree(wallet_dir)
            bt.logging.info(f"Successfully removed {wallet_dir}")
        except Exception as e:
            bt.logging.error(f"Error removing {wallet_dir}: {e}")
    else:
        bt.logging.info("Mock wallet directory does not exist, nothing to clean up.")


async def run_miner(
    hotkey: str, metagraph: bt.metagraph, validator_uid: int, initial_timeout: float, config: bt.config
) -> None:
    """Run a single miner indefinitely."""

    # Generate fake miner hotkey
    config = copy.deepcopy(config)
    hotkey_data = generate_fake_hotkey_full()
    hotkey_name = str(uuid4())
    save_to_json_file(hotkey_data, hotkey_name)
    config.wallet.hotkey = str(hotkey_name)
    bt.logging(config=config)
    wallet = bt.wallet(config=config)

    # Create a new subtensor instance with the updated config
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor.network}")
    dendrite = bt.dendrite(wallet)

    hotkey = hotkey_data["ss58Address"]
    bt.logging.info(
        f"Miner {hotkey} started, targeting validator UID: "
        f"{validator_uid} with hotkey: {metagraph.hotkeys[validator_uid]}"
    )

    # Run the miner indefinitely
    while True:
        bt.logging.info(f"Miner {hotkey}: Pulling a task")
        await asyncio.sleep(initial_timeout)  # Cooldown to wait for other miners to start

        task, cooldown_until = await pull_task(dendrite, metagraph, validator_uid)
        if task is None:
            cooldown_duration = max(0, cooldown_until - time.time())
            bt.logging.info(f"Miner {hotkey}: No task received, waiting for cooldown: {int(cooldown_duration)} seconds")
            await asyncio.sleep(cooldown_duration)
            continue

        bt.logging.info(f"Miner {hotkey}: Task received: {task}")

        # Wait 1 second before submitting results
        await asyncio.sleep(1.0)

        bt.logging.info(f"Miner {hotkey}: Submitting results")

        feedback, cooldown_until = await submit_results(dendrite, metagraph, validator_uid, task)

        cooldown_duration = cooldown_until - time.time()
        bt.logging.info(
            f"Miner {hotkey}: Received feedback {feedback} and cooldown for {int(cooldown_duration)} seconds"
        )

        # Respect the cooldown period before requesting next task
        if cooldown_duration > 0:
            bt.logging.info(f"Miner {hotkey}: Waiting for cooldown to expire: {int(cooldown_duration)} seconds")
            await asyncio.sleep(cooldown_duration)


async def pull_task(dendrite: bt.dendrite, metagraph: bt.metagraph, validator_uid: int) -> tuple[Task | None, int]:
    synapse = PullTask()
    axon = metagraph.axons[validator_uid]
    response = typing.cast(
        PullTask,
        await dendrite.call(target_axon=axon, synapse=synapse, deserialize=False, timeout=12.0),
    )

    bt.logging.info(f"Response: {response}")

    return response.task, response.cooldown_until


async def submit_results(
    dendrite: bt.dendrite, metagraph: bt.metagraph, validator_uid: int, task: Task
) -> tuple[Feedback | None, int]:
    # Get the absolute path of the current file (mock_miners.py)
    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent
    with Path(current_dir / "monkey.encoded.ply").open("r") as f:  # noqa
        results = f.read()

    axon = metagraph.axons[validator_uid]

    message = f"{0}{task.prompt}{metagraph.hotkeys[validator_uid]}{dendrite.keypair.ss58_address}"
    signature = dendrite.keypair.sign(message)
    synapse = SubmitResults(task=task, results=results, submit_time=0, signature=b64encode(signature))
    response = typing.cast(
        SubmitResults,
        await dendrite.call(
            target_axon=axon,
            synapse=synapse,
            deserialize=False,
            timeout=300.0,
        ),
    )
    return response.feedback, response.cooldown_until


async def main() -> None:
    # Parse command line arguments
    parser = _build_parser()
    parser.add_argument("--miners", type=int, default=1, help="Number of miners to run")
    parser.add_argument("--initial_timeout", type=float, default=60.0, help="Initial timeout for cooldown")
    args = parser.parse_args()

    config = bt.config(parser)
    config.wallet.name = MINER_NAME

    # Create and sync the metagraph once
    bt.logging.info("Creating and syncing metagraph...")
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor network: {subtensor.network}")
    metagraph = bt.metagraph(netuid=config.netuid, network=subtensor.network, sync=False)
    metagraph.sync(subtensor=subtensor)
    bt.logging.info(f"Metagraph synced with {len(metagraph.axons)} axons")

    # Find the validator UID with the specified HOTKEY
    validator_uid = 0  # Default to 0 if not found
    for uid, hotkey in enumerate(metagraph.hotkeys):
        if hotkey == HOTKEY:
            validator_uid = uid
            bt.logging.info(f"Found validator with HOTKEY {HOTKEY} at UID {validator_uid}")
            break

    if validator_uid == 0 and metagraph.hotkeys[0] != HOTKEY:
        bt.logging.warning(
            f"Validator with HOTKEY {HOTKEY} not found in metagraph. "
            f"Using default UID 0 with hotkey {metagraph.hotkeys[0]}"
        )

    # Create tasks for all miners to run concurrently
    tasks = []
    try:
        for miner_id in range(args.miners):
            # Generate a unique identifier for each miner task
            miner_identifier = f"miner_{miner_id}"
            tasks.append(
                asyncio.create_task(run_miner(miner_identifier, metagraph, validator_uid, args.initial_timeout, config))
            )

        # Wait for all tasks (they won't complete as they're infinite loops)
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        bt.logging.info("Received keyboard interrupt, shutting down miners...")
    except Exception as e:
        bt.logging.error(f"Error running miners: {e}")
    finally:
        # Cleanup mock wallet directory at the end
        bt.logging.info("Cleaning up mock wallet directories...")
        cleanup_mock_wallets()
        bt.logging.info("Cleanup completed. Exiting.")


if __name__ == "__main__":
    asyncio.run(main())
