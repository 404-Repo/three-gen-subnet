import urllib.parse

import aiohttp
import bittensor as bt


async def validate(endpoint: str, prompt: str, data: str, data_format: str, data_ver: int) -> float | None:
    if data_format == "ply":
        validate_url = urllib.parse.urljoin(endpoint, "/validate_ply/")
    else:
        validate_url = urllib.parse.urljoin(endpoint, "/validate/")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                validate_url, json={"prompt": prompt, "data": data, "data_ver": data_ver}
            ) as response:
                if response.status == 200:
                    results = await response.json()

                    validation_score = float(results["score"])
                    bt.logging.debug(f"Validation score: {validation_score:.2f} | Prompt: {prompt}")

                    return validation_score
                else:
                    bt.logging.error(f"Validation failed: [{response.status}] {response.reason}")
        except aiohttp.ClientConnectorError:
            bt.logging.error(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {endpoint}.")
        except TimeoutError:
            bt.logging.error(f"The request to the endpoint timed out: {endpoint}")
        except aiohttp.ClientError as e:
            bt.logging.error(f"An unexpected client error occurred: {e} ({endpoint})")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e} ({endpoint})")

    return None


async def version(endpoint: str) -> str:
    version_url = urllib.parse.urljoin(endpoint, "/version/")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(version_url) as response:
                if response.status == 200:
                    endpoint_version = await response.text()
                    bt.logging.debug(f"Validation endpoint version: {endpoint_version}")
                    return endpoint_version
                else:
                    bt.logging.error(f"Validation failed with code: {response.status}")
                    return f"{response.status}:{response.reason}"
        except aiohttp.ClientConnectorError:
            bt.logging.error(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {endpoint}.")
            return "can't reach endpoint"
        except TimeoutError:
            bt.logging.error(f"The request to the endpoint timed out: {endpoint}")
            return "request timeout"
        except aiohttp.ClientError as e:
            bt.logging.error(f"An unexpected client error occurred: {e} ({endpoint})")
            return "unexpected error"
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e} ({endpoint})")

    return "unexpected error"
