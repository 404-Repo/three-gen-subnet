import argparse
import asyncio
import time
import urllib.parse
from pathlib import Path

import aiohttp
import bittensor as bt


async def main() -> None:
    config = await get_config()
    bt.logging(config=config)

    with Path(config.data).open() as f:  # noqa
        data = f.read()

    # with Path(config.data).open("rb") as f:  # noqa
    #     data = base64.b64encode(f.read()).decode("utf-8")

    start_time = time.time()
    jobs = [validate(config.validation.endpoint, config.prompt, data) for _ in range(1)]
    scores = await asyncio.gather(*jobs)
    bt.logging.info(f"Validation scores: {scores}")
    bt.logging.info(f"Validation time: {time.time() - start_time}")


async def validate(endpoint: str, prompt: str, data: str) -> float | None:
    validate_url = urllib.parse.urljoin(endpoint, "/validate_ply/")

    bt.logging.info(f"Url: {validate_url}, prompt: {prompt}, results size: {len(data)}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(validate_url, json={"prompt": prompt, "data": data}) as response:
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


async def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    parser.add_argument(
        "--validation.endpoint",
        type=str,
        help="Specifies the URL of the endpoint responsible for scoring 3D assets. "
        "This endpoint should handle the /validate/ POST route.",
        default="http://127.0.0.1:8094",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the file with generated results.",
        default="monkey.encoded.ply",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt used for generation.",
        default="baby monkey",
    )
    return bt.config(parser)


if __name__ == "__main__":
    asyncio.run(main())
