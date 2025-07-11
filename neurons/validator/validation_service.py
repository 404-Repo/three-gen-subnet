import urllib.parse

import aiohttp
import bittensor as bt
import pydantic
from common.protocol import SubmitResults
from pydantic import BaseModel, Field

from validator.config import config


class ValidationResponse(BaseModel):
    score: float = Field(default=0.0, description="Validation score, from 0.0 to 1.0")
    iqa: float = Field(default=0.0, description="Aesthetic Predictor (quality) score")
    clip: float = Field(default=0.0, description="Clip similarity score")
    ssim: float = Field(default=0.0, description="Structure similarity score")
    lpips: float = Field(default=0.0, description="Perceptive similarity score")
    preview: str | None = Field(default=None, description="Optional. Preview image, base64 encoded PNG")


class ValidationService:
    def __init__(self, *, endpoints: list[str], storage_enabled: bool, validation_score_threshold: float) -> None:
        self._endpoints = endpoints
        self._storage_enabled = storage_enabled
        self._validation_score_threshold = validation_score_threshold

    async def validate(self, *, synapse: SubmitResults, neuron_uid: int) -> ValidationResponse | None:
        """Validates miner's result using validation service."""
        prompt = synapse.task.prompt
        data = synapse.results
        endpoint = self._endpoints[neuron_uid % len(self._endpoints)]
        validate_url = urllib.parse.urljoin(endpoint, "/validate_txt_to_3d_ply/")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    validate_url,
                    json={
                        "prompt": prompt,
                        "data": data,
                        "compression": 2,
                        "generate_preview": self._storage_enabled,
                        "preview_score_threshold": self._validation_score_threshold - 0.1,
                    },
                ) as response:
                    if response.status == 200:
                        data_dict = await response.json()
                        results = ValidationResponse(**data_dict)
                        bt.logging.debug(f"Validation score: {results.score:.2f}. Prompt: {prompt[:100]}")
                        return results
                    else:
                        bt.logging.error(
                            f"Validation failed: [{response.status}] {response.reason}. " f"Prompt: {prompt[:100]}"
                        )
            except aiohttp.ClientConnectorError:
                bt.logging.error(
                    f"Failed to connect to the endpoint. The endpoint might be inaccessible: {validate_url}."
                )
            except TimeoutError:
                bt.logging.error(f"The request to the endpoint timed out: {validate_url}. Prompt: {prompt[:100]}")
            except aiohttp.ClientError as e:
                bt.logging.error(f"An unexpected client error occurred: {e} ({validate_url}). Prompt: {prompt[:100]}")
            except pydantic.ValidationError as e:
                bt.logging.error(
                    f"Incompatible validation response format: {e} ({validate_url}). Prompt: {prompt[:100]}"
                )
            except Exception as e:
                bt.logging.error(f"An unexpected error occurred: {e} ({validate_url}). Prompt: {prompt[:100]}")

        return None

    async def version(self) -> str:
        async with aiohttp.ClientSession() as session:
            try:
                if not self._endpoints:
                    raise ValueError("ValidationService: No endpoints provided")

                version_url = urllib.parse.urljoin(self._endpoints[0], "/version/")
                async with session.get(version_url) as response:
                    if response.status == 200:
                        endpoint_version = await response.text()
                        bt.logging.debug(f"Validation endpoint version: {endpoint_version}")
                        return str(endpoint_version)
                    else:
                        bt.logging.error(f"Validation failed with code: {response.status}")
                        return f"{response.status}:{response.reason}"
            except aiohttp.ClientConnectorError:
                bt.logging.error(
                    f"Failed to connect to the endpoint. The endpoint might be inaccessible: {version_url}."
                )
                return "can't reach endpoint"
            except TimeoutError:
                bt.logging.error(f"The request to the endpoint timed out: {version_url}")
                return "request timeout"
            except aiohttp.ClientError as e:
                bt.logging.error(f"An unexpected client error occurred: {e} ({version_url})")
                return "unexpected error"
            except Exception as e:
                bt.logging.error(f"An unexpected error occurred: {e} ({version_url})")

        return "unexpected error"


endpoints = config.validation.endpoints
validation_service = ValidationService(
    endpoints=config.validation.endpoints,
    storage_enabled=config.storage.enabled,
    validation_score_threshold=config.storage.validation_score_threshold,
)
