import time
import urllib.parse

import aiohttp
import bittensor as bt
import pydantic
from common.protocol import SubmitResults
from pydantic import BaseModel, Field


VALIDATION_TIME_DECAY_FACTOR = 0.95


class ValidationResponse(BaseModel):
    score: float = Field(default=0.0, description="Validation score, from 0.0 to 1.0")
    iqa: float = Field(default=0.0, description="Aesthetic Predictor (quality) score")
    alignment_score: float = Field(
        default=0.0, description="prompt vs rendered images or prompt-image vs rendered images score."
    )
    ssim: float = Field(default=0.0, description="Structure similarity score")
    lpips: float = Field(default=0.0, description="Perceptive similarity score")
    preview: str | None = Field(default=None, description="Optional. Preview image, base64 encoded PNG")
    grid_preview: str | None = Field(
        default=None, description="Base64-encoded PNG of 2x2 grid showing multiple angles/views"
    )


class ValidationService:
    def __init__(self, *, endpoints: list[str], validation_score_threshold: float) -> None:
        self._endpoints = endpoints
        self._validation_score_threshold = validation_score_threshold
        self._peak_validation_time = 1.0

    async def validate(
        self, *, synapse: SubmitResults, neuron_uid: int, generate_single_preview: bool, generate_grid_preview: bool
    ) -> ValidationResponse | None:
        """Validates miner's result using validation service."""
        prompt = synapse.task.prompt
        data = synapse.results
        endpoint = self._endpoints[neuron_uid % len(self._endpoints)]
        validate_url = urllib.parse.urljoin(endpoint, "/validate_txt_to_3d_ply/")
        results = None

        async with aiohttp.ClientSession() as session:
            try:
                start_time = time.time()
                async with session.post(
                    validate_url,
                    json={
                        "prompt": prompt,
                        "data": data,
                        "compression": 2,
                        "generate_single_preview": generate_single_preview,
                        "generate_grid_preview": generate_grid_preview,
                        "preview_score_threshold": self._validation_score_threshold - 0.1,
                    },
                ) as response:
                    if response.status == 200:
                        data_dict = await response.json()
                        results = ValidationResponse(**data_dict)
                        validation_time = time.time() - start_time
                        self._peak_validation_time = max(
                            validation_time, self._peak_validation_time * VALIDATION_TIME_DECAY_FACTOR
                        )
                        bt.logging.debug(
                            f"Validation score: {results.score:.2f}, "
                            f"time: {validation_time:.2f} sec ({self._peak_validation_time:.2f} sec). "
                            f"Prompt: {prompt[:100]}."
                        )
                    else:
                        bt.logging.error(f"Validation failed: [{response.status}] {response.reason}. ({prompt[:100]}).")
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

        return results

    def peak_validation_time(self) -> float:
        """Returns "the high watermark" of the validation time with decay."""
        return self._peak_validation_time

    async def render_duel_view(self, *, synapse: SubmitResults, neuron_uid: int) -> bytes | None:
        """Renders 2x2 render for the duel using validation service."""
        prompt = synapse.task.prompt
        data = synapse.results
        endpoint = self._endpoints[neuron_uid % len(self._endpoints)]
        render_url = urllib.parse.urljoin(endpoint, "/render_duel_view/")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    render_url,
                    json={
                        "prompt": prompt,
                        "data": data,
                        "compression": 2,
                    },
                ) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        bt.logging.error(
                            f"Rendering failed: [{response.status}] {response.reason}. Prompt: {prompt[:100]}"
                        )
            except aiohttp.ClientConnectorError:
                bt.logging.error(
                    f"Failed to connect to the endpoint. The endpoint might be inaccessible: {render_url}."
                )
            except TimeoutError:
                bt.logging.error(f"The request to the endpoint timed out: {render_url}. Prompt: {prompt[:100]}")
            except aiohttp.ClientError as e:
                bt.logging.error(f"An unexpected client error occurred: {e} ({render_url}). Prompt: {prompt[:100]}")
            except Exception as e:
                bt.logging.error(f"An unexpected error occurred: {e} ({render_url}). Prompt: {prompt[:100]}")

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
