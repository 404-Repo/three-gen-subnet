import argparse
import pybase64
import inspect
import sys
import os
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, PARENT_DIR + "/validation")

import requests
from loguru import logger
from engine.data_structures import ValidationRequest


current_file_path = Path(__file__).resolve()
test_data_folder = current_file_path.parent / "resources"
DEFAULT_FILEPATH = test_data_folder / "hamburger.ply"
DEFAULT_PROMPT = "A hamburger"
DEFAULT_PROMPT_IMAGE = test_data_folder / "test_render_ply.png"
DEFAULT_URL_TXT_TO_3D = "http://localhost:10006/validate_txt_to_3d_ply/"
DEFAULT_URL_IMG_TO_3D = "http://localhost:10006/validate_img_to_3d_ply/"
DEFAULT_VERSION = 0
GENERATE_PREVIEW = False


def send_post_request(
    prompt: str | None, prompt_image: str | None, ply_file_path: str, url: str, generate_preview: bool
) -> None:
    try:
        # Read the binary file content
        with open(ply_file_path, "rb") as file:
            file_data = file.read()

        # Encode the binary data to base64
        encoded_data = pybase64.b64encode(file_data).decode("utf-8")

        # Create the payload
        if prompt is not None:
            request_data = ValidationRequest(
                prompt=prompt,
                data=encoded_data,
                generate_preview=generate_preview,
            )
        elif prompt_image is not None:
            with open(prompt_image, "rb") as img_file:
                encoded_prompt_image = pybase64.b64encode(img_file.read()).decode("utf-8")

            request_data = ValidationRequest(
                prompt_image=encoded_prompt_image,
                data=encoded_data,
                generate_preview=generate_preview,
            )
        else:
            raise ValueError("prompt or prompt image variables were not specified.")

        # Send the POST request
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=request_data.model_dump(), headers=headers)

        # Print the response from the server
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response content: {response.text}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a POST request with binary file content.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="The prompt to send via the POST request.")
    parser.add_argument(
        "--prompt_image", type=str, default=DEFAULT_PROMPT_IMAGE, help="The prompt image to send via POST request."
    )
    parser.add_argument("--file_path", type=str, default=DEFAULT_FILEPATH, help="The path to the file to be read.")
    parser.add_argument("--preview", type=bool, default=GENERATE_PREVIEW, help="enable/disable preview generation.")
    args = parser.parse_args()

    send_post_request(args.prompt, None, args.file_path, DEFAULT_URL_TXT_TO_3D, args.preview)
    send_post_request(None, args.prompt_image, args.file_path, DEFAULT_URL_IMG_TO_3D, args.preview)
