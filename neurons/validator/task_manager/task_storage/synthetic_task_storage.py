import asyncio
import random as rd
import uuid
from pathlib import Path

import bittensor as bt
from common.protocol import SubmitResults

from validator.config import config
from validator.task_manager.task import SyntheticTask
from validator.task_manager.task_storage.base_task_storage import BaseTaskStorage
from validator.task_manager.task_storage.synthetic_asset_storage import SyntheticAssetStorage, synthetic_asset_storage
from validator.task_manager.task_storage.synthetic_prompt_service import (
    SyntheticPromptService,
    synthetic_prompt_service,
)
from validator.validation_service import ValidationResponse


class SyntheticTaskStorage(BaseTaskStorage):
    """Storage that is used to store synthetic tasks."""

    def __init__(
        self,
        *,
        default_prompts_path: str,
        config: bt.config,
        wallet: bt.wallet | None = None,
        synthetic_prompt_service: SyntheticPromptService,
        synthetic_asset_storage: SyntheticAssetStorage,
    ) -> None:
        super().__init__(config=config, wallet=wallet)

        self._default_prompts: list[str] = []
        self._load_default_prompts(default_prompts_path)
        self._prompts: list[str] = []
        self._tasks: dict[str, str] = {}
        self._synthetic_prompt_service = synthetic_prompt_service
        self._synthetic_asset_storage = synthetic_asset_storage

    def get_next_task(self, *, miner_uid: int) -> SyntheticTask:  # type: ignore
        """Returns random prompt. First returns from fresh_prompts and then from default_prompts."""
        if self._prompts:
            prompt = rd.choice(self._prompts)  # noqa: S311 # nosec: B311
        else:
            prompt = rd.choice(self._default_prompts)  # noqa: S311 # nosec: B311
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = prompt
        bt.logging.info(f"[{miner_uid}] received synthetic task ({prompt[:100]})")
        return SyntheticTask(id=task_id, prompt=prompt)

    async def fetch_synthetic_tasks_cron(self) -> None:
        """Fetches new prompts from the prompter service."""
        if not self._synthetic_prompt_service or not self._wallet:
            bt.logging.warning("Cannot fetch tasks - missing prompt service or wallet")
            return

        await asyncio.sleep(self._config.task.synthetic.prompter.delay)
        while True:
            try:
                prompts = await self._synthetic_prompt_service.get_prompts(hotkey=self._wallet.hotkey)
                if prompts:
                    self._prompts = prompts
            except Exception as e:
                bt.logging.error(f"Failed fetching synthetic tasks cron: {e}")
            finally:
                await asyncio.sleep(self._config.task.synthetic.prompter.fetch_interval)

    def has_task(self, *, task_id: str) -> bool:
        return task_id in self._tasks

    def has_tasks(self) -> bool:
        return True

    def submit_result(self, *, synapse: SubmitResults, validation_res: ValidationResponse, miner_uid: int) -> None:
        bt.logging.info(
            f"[{miner_uid}] submit synthetic task results with score {validation_res.score} "
            f"({synapse.task.prompt[:100]})"
        )
        self._tasks.pop(synapse.task.id)
        if self._synthetic_asset_storage.enabled:
            asyncio.create_task(
                self._synthetic_asset_storage.save_assets(synapse, synapse.results, synapse.signature, validation_res)
            )

    def fail_task(self, *, task_id: str, task_prompt: str, hotkey: str, miner_uid: int) -> None:  # noqa: B027
        bt.logging.info(f"[{miner_uid}] failed synthetic task ({task_prompt[:50]}).")
        self._tasks.pop(task_id)

    def _load_default_prompts(self, path: str) -> None:
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
            self._default_prompts = default_prompts

        bt.logging.info(f"{len(self._default_prompts)} default prompts loaded")


synthetic_task_storage = SyntheticTaskStorage(
    default_prompts_path=config.task.synthetic.default_prompts_path,
    synthetic_prompt_service=synthetic_prompt_service,
    synthetic_asset_storage=synthetic_asset_storage,
    config=config,
)
