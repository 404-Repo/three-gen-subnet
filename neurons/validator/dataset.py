import base64
import random
import time
from pathlib import Path

import aiohttp
import bittensor as bt


FIRST_FETCH_DELAY = 5 * 60  # 5 minutes


class Dataset:
    def __init__(
        self, default_prompts_path: str, prompter_url: str, fetch_prompt_interval: int, wallet: bt.wallet
    ) -> None:
        self._default_prompts: list[str] = [
            "Monkey",
        ]
        self._load_default_prompts(default_prompts_path)

        self._prompter_url = prompter_url
        self._last_fetch_time = time.time() - fetch_prompt_interval + FIRST_FETCH_DELAY
        self._fetch_prompt_interval = fetch_prompt_interval
        self._wallet = wallet

        self._fresh_prompts: list[str] = []

    def _load_default_prompts(self, path: str) -> None:
        dataset_path = Path(path)
        if not dataset_path.is_absolute():
            dataset_path = Path(__file__).parent / ".." / ".." / dataset_path
        else:
            dataset_path = Path(path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file {dataset_path} not found")

        with dataset_path.open() as f:
            self._default_prompts = f.read().strip().split("\n")

        bt.logging.info(f"{len(self._default_prompts)} default prompts loaded")

    def get_random_prompt(self) -> str:
        available_prompts = self._fresh_prompts or self._default_prompts
        return random.choice(available_prompts)  # noqa # nosec

    def should_fetch_fresh_prompts(self) -> bool:
        return time.time() > self._last_fetch_time + self._fetch_prompt_interval

    async def fetch_fresh_prompts(self) -> None:
        self._last_fetch_time = time.time()

        url = self._prompter_url
        bt.logging.info(f"Fetching new prompts from {url}")

        hotkey = self._wallet.hotkey
        nonce = time.time_ns()
        message = f"{nonce}{hotkey.ss58_address}"
        signature = base64.b64encode(self._wallet.hotkey.sign(message)).decode(encoding="utf-8")
        payload = {"hotkey": hotkey.ss58_address, "nonce": nonce, "signature": signature}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/get", json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    prompts = result["prompts"]

            # Basic correctness check
            if len(prompts) < 1000:
                bt.logging.error(f"Insufficient amount ({len(prompts)}) of prompts fetched")
                return

            self._fresh_prompts = prompts
            bt.logging.info(f"{len(prompts)} fresh prompts fetched")
        except aiohttp.ClientConnectorError:
            bt.logging.error(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {url}.")
        except TimeoutError:
            bt.logging.error(f"The request to the endpoint timed out: {url}")
        except aiohttp.ClientError as e:
            bt.logging.error(f"An unexpected client error occurred: {e} ({url})")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e} ({url})")
