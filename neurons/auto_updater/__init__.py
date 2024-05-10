import asyncio
import re
import time
from pathlib import Path

import aiohttp
import bittensor as bt


class AutoUpdater:
    def __init__(self, disabled: bool, interval: int, local_version: int) -> None:
        self._disabled = disabled
        self._interval = interval
        self._local_version = local_version
        self._last_check_time = 0.0

    async def should_update(self) -> bool:
        if self._disabled:
            return False

        if time.time() < self._last_check_time + self._interval:
            return False

        self._last_check_time = time.time()

        remote_content = await fetch_remote_version()
        if not remote_content:
            bt.logging.error("Failed to fetch latest validator version")
            return False

        remote_version = extract_version(remote_content)
        if remote_version is None:
            bt.logging.error(f"Unexpected remote version format: {remote_content}")
            return False

        if remote_version <= self._local_version:
            bt.logging.debug("Validator version is up-to-date")
            return False

        return True

    @staticmethod
    async def update() -> None:
        bt.logging.info("New version detected. Running update scripts...")
        pull_script = Path(__file__).parent.parent.parent / "git_pull.sh"
        return_code = await run_script(pull_script)
        if return_code != 0:
            return

        update_script = Path(__file__).parent.parent.parent / "update_validator.sh"
        await run_script(update_script)
        if return_code != 0:
            return

        bt.logging.info("Update scripts executed successfully.")


async def fetch_remote_version() -> str | None:
    url = "https://raw.githubusercontent.com/404-Repo/three-gen-subnet/main/neurons/validator/version.py"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()
    except aiohttp.ClientError as e:
        bt.logging.exception(f"Could not access Git to check the latest validator version: {e}")
        return None


def extract_version(file_content: str) -> int | None:
    version_match = re.search(r"VALIDATOR_VERSION\s*=\s*(\d+)", file_content)
    if version_match:
        return int(version_match.group(1))
    return None


async def run_script(script_path: Path) -> int | None:
    process = await asyncio.create_subprocess_exec(
        "/bin/bash", str(script_path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        bt.logging.error(f"Auto-update failed: {stderr.decode().strip()}")
    else:
        bt.logging.info(f"{script_path} finished successfully")
    return process.returncode
