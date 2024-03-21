import urllib.parse
import requests
import bittensor as bt
from pydantic import BaseModel

_VALUE_IF_FAILED = 0.9


class ResponseData(BaseModel):
    score: float


def _quality_factor(prompt: str, validation_score: float) -> float:
    bt.logging.debug(f"Validation score: {validation_score:.2f} | Prompt: {prompt}")

    if validation_score > 0.8:
        return 1.0
    if validation_score > 0.6:
        return 0.75
    return 0.0


async def prove_generation(endpoint: str, prompt: str, data: str) -> float:
    generate_url = urllib.parse.urljoin(endpoint, "/validate/")

    try:
        response = requests.post(generate_url, json={"prompt": prompt, "data": data})
        if response.status_code == 200:
            results = response.json()
            return _quality_factor(prompt, results["score"])
        else:
            bt.logging.error(f"Validation failed with code: {response.status_code}")
    except requests.ConnectionError:
        bt.logging.error(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {endpoint}.")
    except requests.Timeout:
        bt.logging.error(f"The request to the endpoint timed out: {endpoint}")
    except requests.RequestException as e:
        bt.logging.error(f"An unexpected client error occurred: {e} ({endpoint})")
    except Exception as e:
        bt.logging.error(f"An unexpected error occurred: {e} ({endpoint})")

    return _VALUE_IF_FAILED
