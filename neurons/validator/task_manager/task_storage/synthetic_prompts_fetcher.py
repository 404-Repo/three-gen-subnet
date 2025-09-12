import asyncio
import random
import time

import aiohttp
import bittensor as bt
import pybase64
from bittensor_wallet import Wallet


class SyntheticPromptsFetcher:
    """
    Responsible for fetching and managing synthetic prompts from the prompt service.
    Handles both the HTTP communication and the periodic fetching logic.
    """

    def __init__(
        self,
        *,
        config: bt.config,
        wallet: Wallet | None = None,
        min_prompts_threshold: int = 1000,
    ) -> None:
        self._wallet = bt.wallet(config=config) if wallet is None else wallet
        self._prompt_service_url = config.task.synthetic.get_prompts.endpoint
        self._initial_delay = config.task.synthetic.get_prompts.initial_delay
        self._text_fetch_interval = config.task.synthetic.get_prompts.text_fetch_interval
        self._image_fetch_interval = config.task.synthetic.get_prompts.image_fetch_interval
        self._min_prompts_threshold = min_prompts_threshold

        self._latest_text_prompts: list[str] = []
        self._latest_image_prompts: list[str] = []

        self._fetch_text_prompts_job: asyncio.Task | None = None
        self._fetch_image_prompts_job: asyncio.Task | None = None

    async def start_fetching_prompts(self) -> None:
        if not self._prompt_service_url or not self._wallet:
            bt.logging.warning("Cannot fetch tasks - missing prompt service or wallet")
            return

        self._fetch_text_prompts_job = asyncio.create_task(
            self._fetch_prompts_in_a_loop(
                route="text-prompts/get/", prompt_type="text", interval=self._text_fetch_interval
            )
        )
        self._fetch_image_prompts_job = asyncio.create_task(
            self._fetch_prompts_in_a_loop(
                route="image-prompts/get/", prompt_type="image", interval=self._image_fetch_interval
            )
        )

    async def stop_fetching_prompts(self) -> None:
        if self._fetch_text_prompts_job is not None:
            self._fetch_text_prompts_job.cancel()
        if self._fetch_image_prompts_job is not None:
            self._fetch_image_prompts_job.cancel()

    def get_random_text_prompt(self, default_prompts: list[str]) -> str:
        if self._latest_text_prompts:
            return random.choice(self._latest_text_prompts)  # noqa: S311 # nosec: B311
        else:
            return random.choice(default_prompts)  # noqa: S311 # nosec: B311

    def get_random_image_prompt(self, default_prompts: list[str]) -> str:
        if self._latest_image_prompts:
            return random.choice(self._latest_image_prompts)  # noqa: S311 # nosec: B311
        else:
            return random.choice(default_prompts)  # noqa: S311 # nosec: B311

    async def _fetch_prompts_in_a_loop(self, *, route: str, prompt_type: str, interval: int) -> None:
        await asyncio.sleep(self._initial_delay)
        while True:
            try:
                prompts = await self._fetch_prompts(route=route, prompt_type=prompt_type)

                if len(prompts) < self._min_prompts_threshold:
                    bt.logging.error(f"Insufficient amount ({len(prompts)}) of {prompt_type} prompts fetched")
                    continue

                if prompt_type == "text":
                    self._latest_text_prompts = prompts
                elif prompt_type == "image":
                    self._latest_image_prompts = prompts
            except Exception as e:
                bt.logging.error(f"Failed fetching {prompt_type} prompts: {e}")
            finally:
                await asyncio.sleep(interval)

    async def _fetch_prompts(self, *, route: str, prompt_type: str) -> list[str]:
        """
        Fetch synthetic prompts from the prompt service.
        """
        prompts: list[str] = []
        nonce = time.time_ns()
        hotkey = self._wallet.hotkey
        message = f"{nonce}{hotkey.ss58_address}"
        signature = pybase64.b64encode(hotkey.sign(message)).decode(encoding="utf-8")
        payload = {"hotkey": hotkey.ss58_address, "nonce": nonce, "signature": signature}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self._prompt_service_url}/{route}", json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                prompts = result["prompts"]

        bt.logging.info(f"{len(prompts)} synthetic {prompt_type} prompts fetched.")
        return prompts
