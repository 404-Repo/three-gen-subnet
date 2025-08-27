import time

import aiohttp
import bittensor as bt
import pybase64
from bittensor_wallet import Keypair


class SyntheticPromptService:
    """
    Service for fetching synthetic prompts from the prompt service.
    """

    def __init__(self, *, prompt_service_url: str, batch_size: int) -> None:
        self._prompt_service_url = prompt_service_url
        self._batch_size = batch_size

    async def get_prompts(self, *, hotkey: Keypair) -> list[str]:
        """
        Fetches synthetic prompts from the prompt service.
        """
        prompts: list[str] = []
        nonce = time.time_ns()
        message = f"{nonce}{hotkey.ss58_address}"
        signature = pybase64.b64encode(hotkey.sign(message)).decode(encoding="utf-8")
        payload = {"hotkey": hotkey.ss58_address, "nonce": nonce, "signature": signature}
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self._prompt_service_url}/get", json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                prompts = result["prompts"]

        if len(prompts) < 1000:
            bt.logging.error(f"Insufficient amount ({len(prompts)}) of prompts fetched")
            return []

        bt.logging.info(f"{len(prompts)}/{self._batch_size} synthetic prompts fetched.")
        return prompts
