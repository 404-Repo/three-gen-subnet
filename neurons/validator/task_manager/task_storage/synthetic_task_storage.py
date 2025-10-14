import asyncio
import random as rd
from pathlib import Path

import aiohttp
import bittensor as bt
import pybase64
from common.protocol import ImageTask, ProtocolTask, SubmitResults, TextTask

from validator.task_manager.task_storage.base_task_storage import BaseTaskStorage
from validator.task_manager.task_storage.synthetic_asset_storage import SyntheticAssetStorage
from validator.task_manager.task_storage.synthetic_prompts_fetcher import (
    SyntheticPromptsFetcher,
)
from validator.task_manager.validator_task import ValidatorTask
from validator.validation_service import ValidationResponse


class SyntheticTask(ValidatorTask):
    pass


class SyntheticTaskStorage(BaseTaskStorage):
    """Storage that is used to store synthetic tasks."""

    def __init__(
        self,
        *,
        default_text_prompts_path: str,
        default_image_prompts_path: str,
        config: bt.config,
        wallet: bt.wallet | None = None,
        synthetic_prompts_fetcher: SyntheticPromptsFetcher,
        synthetic_asset_storage: SyntheticAssetStorage,
    ) -> None:
        super().__init__(config=config, wallet=wallet)

        self._default_text_prompts: list[str] = self._load_default_prompts(default_text_prompts_path)
        self._default_image_prompts: list[str] = self._load_default_prompts(default_image_prompts_path)

        self._text_tasks_ratio = config.task.synthetic.text_tasks_ratio

        self._tasks: set[str] = set()

        self._synthetic_prompts_fetcher = synthetic_prompts_fetcher
        self._synthetic_asset_storage = synthetic_asset_storage

    def _get_random_text_task(self) -> TextTask:
        prompt = self._synthetic_prompts_fetcher.get_random_text_prompt(self._default_text_prompts)
        return TextTask(prompt=prompt)

    async def _try_get_random_image_task(self) -> ImageTask | None:
        """Downloads a random image from a URL and returns it as base64-encoded data."""
        if not self._default_image_prompts:
            return None
        url = self._synthetic_prompts_fetcher.get_random_image_prompt(self._default_image_prompts)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    content = await resp.read()
                    prompt = pybase64.b64encode(content).decode("utf-8")
                    task = ImageTask(prompt=prompt)
                    bt.logging.debug(f"Image prompt downloaded from: {url} ({task.log_id})")
                    return task
        except Exception as e:
            bt.logging.error(f"Failed to download image prompt from {url}: {e}")
            return None

    async def get_random_task(self) -> ProtocolTask:
        """Returns a random synthetic task.
        The task type (text/image) is chosen randomly based on text_tasks_ratio config."""
        if self._text_tasks_ratio >= 1.0:
            desired_type = "text"
        elif self._text_tasks_ratio <= 0.0:
            desired_type = "image"
        else:
            desired_type = "text" if rd.random() < self._text_tasks_ratio else "image"  # noqa: S311  # nosec: B311

        task: ProtocolTask | None = None
        if desired_type == "image":
            task = await self._try_get_random_image_task()

        if task is None:
            task = self._get_random_text_task()

        self._tasks.add(task.id)
        return task

    async def get_next_task(self, *, miner_uid: int) -> SyntheticTask:
        """Returns a synthetic task (random task)."""
        task = await self.get_random_task()
        bt.logging.info(f"[{miner_uid}] received synthetic {task.type} task ({task.log_id})")
        return SyntheticTask(protocol=task)

    def has_task(self, *, task_id: str) -> bool:
        return task_id in self._tasks

    async def submit_result(
        self, *, protocol_task: ProtocolTask, synapse: SubmitResults, validation_res: ValidationResponse, miner_uid: int
    ) -> None:
        bt.logging.info(
            f"[{miner_uid}] submit synthetic task results with score {validation_res.score} ({protocol_task.log_id})"
        )
        self._tasks.discard(synapse.task_id)

        if self._synthetic_asset_storage.enabled and protocol_task.type == "text":
            asyncio.create_task(
                self._synthetic_asset_storage.save_assets(
                    protocol_task, synapse, synapse.results, synapse.signature, validation_res
                )
            )

    def fail_task(self, *, protocol_task: ProtocolTask, miner_uid: int) -> None:
        bt.logging.info(f"[{miner_uid}] failed synthetic task ({protocol_task.log_id}).")
        self._tasks.discard(protocol_task.id)

    def _load_default_prompts(self, path: str) -> list[str]:
        """Loads default synthetic prompts from the file if it exists."""
        dataset_path = Path(path)
        if not dataset_path.is_absolute():
            dataset_path = Path(__file__).parent / ".." / ".." / ".." / ".." / dataset_path
        else:
            dataset_path = Path(path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file {dataset_path} not found")

        with dataset_path.open() as f:
            default_prompts = f.read().strip().split("\n")

        bt.logging.info(f"{len(default_prompts)} default prompts loaded from {dataset_path}")
        return default_prompts
