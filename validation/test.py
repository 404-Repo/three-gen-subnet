import asyncio
import base64
import json

import aiohttp


def h5_to_base64():
    file_path = "/Users/maximbladyko/atlas/three-gen-subnet/resources/tiger_pcl.h5"
    with open(file_path, "rb") as file:
        content = file.read()
        encoded_content = base64.b64encode(content)
        encoded_content_str = encoded_content.decode("utf-8")
        return encoded_content_str


async def send_json_request(url, json_body):
    async with aiohttp.ClientSession() as session:
        headers = {"Content-Type": "application/json"}
        async with session.post(url, data=json.dumps(json_body), headers=headers) as response:
            response_json = await response.json()
            return response_json


async def main():
    url = "http://127.0.0.1:8094/validate/"

    json_body = {"prompt": "A tiger", "data": h5_to_base64()}

    # Send the JSON request
    r = await asyncio.gather(send_json_request(url, json_body), send_json_request(url, json_body))
    print(r)


# Run the main function
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
