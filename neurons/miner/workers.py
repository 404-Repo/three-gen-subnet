import urllib.parse
import bittensor as bt

import requests
from miner.task_registry import TaskRegistry


def worker_task(endpoint: str, task_registry: TaskRegistry):
    bt.logging.info(f"Worker ({endpoint}) started")

    generate_url = urllib.parse.urljoin(endpoint, "/generate/")
    while True:
        task = task_registry.start_next_task()
        results = None
        try:
            response = requests.post(generate_url, data={"prompt": task.prompt})
            if response.status_code == 200:
                results = response.content
            else:
                bt.logging.error(f"Generation failed with code: {response.status_code}")
        except requests.ConnectionError:
            bt.logging.error(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {endpoint}.")
        except requests.Timeout:
            bt.logging.error(f"The request to the endpoint timed out: {endpoint}")
        except requests.RequestException as e:
            bt.logging.error(f"An unexpected client error occurred: {e} ({endpoint})")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e} ({endpoint})")

        if results is not None:
            task_registry.complete_task(task.id, results)
        else:
            task_registry.fail_task(task.id)
