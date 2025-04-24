import argparse
import asyncio
import random as rd
from pathlib import Path

from validator.gateway.http3_client.http3_client import Http3Client


class GatewayTestClient:
    """For test purposes only. Sends requests to the gateway on behalf of the client."""

    _INTERVAL_SEC: int = 1

    async def run(self, *, x_api_key: str) -> None:
        prompts_file = Path(__file__).resolve().parent.parent.parent.parent.parent / "resources/prompts.txt"
        with prompts_file.open("r") as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts]
        http3_client = Http3Client()
        gateways = [
            "https://gateway-us-west.404.xyz:4443",
            "https://gateway-us-east.404.xyz:4443",
            "https://gateway-eu.404.xyz:4443",
        ]

        while True:
            try:
                prompt = rd.choice(prompts)  # noqa: S311 # nosec: B311
                gateway = rd.choice(gateways)  # noqa: S311 # nosec: B311
                print(f"Sending prompt: {prompt} to gateway: {gateway}")
                url = f"{gateway}/add_task"
                # TODO: generate error in case of incorrect response
                await http3_client.post(url=url, payload={"prompt": prompt}, headers={"x-api-key": x_api_key})
            except Exception as e:
                print(f"Error sending prompt: {e}")
            await asyncio.sleep(self._INTERVAL_SEC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-api-key", type=str)
    args = parser.parse_args()
    asyncio.run(GatewayTestClient().run(x_api_key=args.x_api_key))  # type: ignore
