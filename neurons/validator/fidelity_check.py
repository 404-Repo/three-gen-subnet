import asyncio
import urllib.parse

import aiohttp
import requests
import bittensor as bt
from pydantic import BaseModel

_VALUE_IF_FAILED = 0.9


class ResponseData(BaseModel):
    score: float


def _fidelity_factor(prompt: str, validation_score: float) -> float:
    bt.logging.debug(f"Fidelity score: {validation_score:.2f} | Prompt: {prompt}")

    if validation_score > 0.8:
        return 1.0
    if validation_score > 0.6:
        return 0.75
    return 0.0


async def validate(endpoint: str, prompt: str, data: str) -> float:
    generate_url = urllib.parse.urljoin(endpoint, "/validate/")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(generate_url, json={"prompt": prompt, "data": data}) as response:
                if response.status == 200:
                    results = await response.json()
                    return _fidelity_factor(prompt, results["score"])
                else:
                    bt.logging.error(f"Validation failed with code: {response.status}")
        except aiohttp.ClientConnectorError:
            bt.logging.error(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {endpoint}.")
        except asyncio.TimeoutError:
            bt.logging.error(f"The request to the endpoint timed out: {endpoint}")
        except aiohttp.ClientError as e:
            bt.logging.error(f"An unexpected client error occurred: {e} ({endpoint})")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e} ({endpoint})")

    return _VALUE_IF_FAILED
