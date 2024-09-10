import argparse
import base64
import inspect
import json
import os

import requests


CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
DEFAULT_PROMPT = "A hamburger"
DEFAULT_FILEPATH = os.path.join(CURRENT_DIR, "resources/hamburger.ply")
DEFAULT_URL = "http://localhost:10006/validate_ply/"
DEFAULT_VERSION = 0


def send_post_request(prompt, file_path, url, version):
    try:
        # Read the binary file content
        with open(file_path, "rb") as file:
            file_data = file.read()

        # Encode the binary data to base64
        encoded_data = base64.b64encode(file_data).decode("utf-8")

        # Create the payload
        payload = {"prompt": prompt, "data": encoded_data, "data_ver": version}

        # Send the POST request
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=json.dumps(payload), headers=headers)

        # Print the response from the server
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a POST request with binary file content.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="The prompt to send the POST request to.")
    parser.add_argument("--file_path", type=str, default=DEFAULT_FILEPATH, help="The path to the file to be read.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="The URL to send the POST request to.")
    parser.add_argument(
        "--version", type=int, default=DEFAULT_VERSION, help="The data version to send the POST request to."
    )

    args = parser.parse_args()

    send_post_request(args.prompt, args.file_path, args.url, args.version)
