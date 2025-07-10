import bittensor as bt
from common.protocol import SubmitResults

from validator.config import config
from validator.duels.duels_task_storage import DuelsTaskStorage, duel_task_storage
from validator.task_manager.task import (
    ValidatorTask,
)
from validator.task_manager.task_storage.organic_task_storage import OrganicTaskStorage, organic_task_storage
from validator.task_manager.task_storage.synthetic_task_storage import SyntheticTaskStorage, synthetic_task_storage
from validator.validation_service import ValidationResponse


class TaskManager:
    def __init__(
        self,
        *,
        organic_task_storage: OrganicTaskStorage,
        synthetic_task_storage: SyntheticTaskStorage,
        duel_task_storage: DuelsTaskStorage,
        config: bt.config,
    ) -> None:
        self._config = config
        """Main application config."""
        self._synthetic_task_storage = synthetic_task_storage
        """Registry responsible for storing and managing synthetic tasks."""
        self._organic_task_storage = organic_task_storage
        """Registry responsible for storing and managing organic tasks."""
        self._duel_task_storage = duel_task_storage
        """Registry responsible for storing and managing duel tasks."""

    async def get_next_task(self, *, miner_uid: int, is_strong_miner: bool, metagraph: bt.metagraph) -> ValidatorTask:
        """Return the next task by miner's request."""
        organic_task = self._organic_task_storage.get_next_task(
            miner_uid=miner_uid,
            hotkey=metagraph.axons[miner_uid].hotkey,
            is_strong_miner=is_strong_miner,
        )
        if organic_task is not None:
            return organic_task

        duel_task = self._duel_task_storage.get_next_task(miner_uid=miner_uid)
        if duel_task is not None:
            return duel_task

        return self._synthetic_task_storage.get_next_task(miner_uid=miner_uid)

    async def submit_result(
        self, *, synapse: SubmitResults, validation_res: ValidationResponse, miner_uid: int, metagraph: bt.metagraph
    ) -> None:
        """Method is called when miner completes the task."""
        task_id = synapse.task.id
        if self._organic_task_storage.has_task(task_id=task_id):
            self._organic_task_storage.submit_result(
                synapse=synapse, validation_res=validation_res, miner_uid=miner_uid
            )
            return

        if self._duel_task_storage.has_task(task_id=task_id):
            await self._duel_task_storage.submit_result(
                synapse=synapse, validation_res=validation_res, miner_uid=miner_uid, metagraph=metagraph
            )
            return

        if self._synthetic_task_storage.has_task(task_id=task_id):
            await self._synthetic_task_storage.submit_result(
                synapse=synapse, validation_res=validation_res, miner_uid=miner_uid
            )
            return

        bt.logging.warning(f"[{miner_uid}]: Unexpected behavior. Undefined task {synapse.task.prompt[:50]}.")

    async def fail_task(
        self, *, task_id: str, task_prompt: str, hotkey: str, miner_uid: int, metagraph: bt.metagraph
    ) -> None:
        if self._organic_task_storage.has_task(task_id=task_id):
            self._organic_task_storage.fail_task(
                task_id=task_id, task_prompt=task_prompt, hotkey=hotkey, miner_uid=miner_uid
            )
            return

        if self._duel_task_storage.has_task(task_id=task_id):
            await self._duel_task_storage.fail_task(
                task_id=task_id, task_prompt=task_prompt, miner_uid=miner_uid, metagraph=metagraph
            )

        self._synthetic_task_storage.fail_task(task_id=task_id, task_prompt=task_prompt, miner_uid=miner_uid)


task_manager = TaskManager(
    organic_task_storage=organic_task_storage,
    synthetic_task_storage=synthetic_task_storage,
    duel_task_storage=duel_task_storage,
    config=config,
)
