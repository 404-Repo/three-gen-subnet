import urllib.parse

import aiohttp
import bittensor as bt


async def validate(endpoint: str, prompt: str, data: str) -> float | None:
    generate_url = urllib.parse.urljoin(endpoint, "/validate/")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(generate_url, json={"prompt": prompt, "data": data}) as response:
                if response.status == 200:
                    results = await response.json()

                    validation_score = float(results["score"])
                    bt.logging.debug(f"Validation score: {validation_score:.2f} | Prompt: {prompt}")

                    return validation_score
                else:
                    bt.logging.error(f"Validation failed with code: {response.status}")
        except aiohttp.ClientConnectorError:
            bt.logging.error(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {endpoint}.")
        except TimeoutError:
            bt.logging.error(f"The request to the endpoint timed out: {endpoint}")
        except aiohttp.ClientError as e:
            bt.logging.error(f"An unexpected client error occurred: {e} ({endpoint})")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e} ({endpoint})")

    return None
