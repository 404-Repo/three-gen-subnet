import time
import urllib.parse

import aiohttp
import bittensor as bt
from common.protocol import SubmitResults

from validator.fidelity_check import ValidationResponse


class StorageWrapper:
    def __init__(
        self, enabled: bool, service_api_key: str, endpoint_url: str, validation_score_threshold: float
    ) -> None:
        self.enabled = enabled
        self.service_api_key = service_api_key
        self.url = urllib.parse.urljoin(endpoint_url, "/store")
        self.validation_score_threshold = validation_score_threshold

        self._headers = {
            "X-Api-Key": self.service_api_key,
        }

    async def save_assets(
        self, synapse: SubmitResults, results: str, signature: str, validation: ValidationResponse
    ) -> None:
        if validation.score < self.validation_score_threshold:
            return None

        try:
            data = {
                "prompt": synapse.task.prompt,
                "assets": results,
                "compression": synapse.compression,
                "preview": validation.preview,
                "meta": {
                    "validator": synapse.axon.hotkey,
                    "miner": synapse.dendrite.hotkey,
                    "submit_time": synapse.submit_time,
                    "signature": signature,
                    "score": {
                        "1.14.0": validation.score,
                        "iqa": validation.iqa,
                        "clip": validation.clip,
                        "ssim": validation.ssim,
                        "lpips": validation.lpips,
                    },
                },
            }
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, json=data, headers=self._headers) as response:
                    response.raise_for_status()
            bt.logging.debug(f"Saving took {time.time() - start_time:.2f} seconds")
        except aiohttp.ClientConnectorError:
            bt.logging.error(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {self.url}.")
        except TimeoutError:
            bt.logging.error(f"The request to the endpoint timed out: {self.url}")
        except aiohttp.ClientError as e:
            bt.logging.error(f"An unexpected client error occurred: {e} ({self.url})")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e} ({self.url})")
