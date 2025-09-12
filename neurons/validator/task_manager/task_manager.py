import bittensor as bt
from common.protocol import ProtocolTask, SubmitResults

from validator.duels.duels_task_storage import DuelsTaskStorage
from validator.task_manager.task_storage.organic_task_storage import OrganicTaskStorage
from validator.task_manager.task_storage.synthetic_task_storage import SyntheticTaskStorage
from validator.task_manager.validator_task import (
    ValidatorTask,
)
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

        duel_task = await self._duel_task_storage.get_next_task(miner_uid=miner_uid)
        if duel_task is not None:
            return duel_task

        return await self._synthetic_task_storage.get_next_task(miner_uid=miner_uid)

    def grid_preview_needed(self, *, synapse: SubmitResults) -> bool:
        """Some types of tasks require a grid preview (e.g. for duels)."""
        task_id = synapse.task_id
        if self._organic_task_storage.has_task(task_id=task_id):
            return True
        if self._duel_task_storage.has_task(task_id=task_id):
            return True
        return False

    async def submit_result(
        self,
        *,
        task: ProtocolTask,
        synapse: SubmitResults,
        validation_res: ValidationResponse,
        miner_uid: int,
        metagraph: bt.metagraph,
    ) -> None:
        """Method is called when miner completes the task."""
        task_id = synapse.task_id
        if self._organic_task_storage.has_task(task_id=task_id):
            await self._organic_task_storage.submit_result(
                protocol_task=task, synapse=synapse, validation_res=validation_res, miner_uid=miner_uid
            )
            return

        if self._duel_task_storage.has_task(task_id=task_id):
            await self._duel_task_storage.submit_result(
                protocol_task=task,
                synapse=synapse,
                validation_res=validation_res,
                miner_uid=miner_uid,
                metagraph=metagraph,
            )
            return

        if self._synthetic_task_storage.has_task(task_id=task_id):
            await self._synthetic_task_storage.submit_result(
                protocol_task=task, synapse=synapse, validation_res=validation_res, miner_uid=miner_uid
            )
            return

    async def fail_task(self, *, task: ProtocolTask, hotkey: str, miner_uid: int, metagraph: bt.metagraph) -> None:
        if self._organic_task_storage.has_task(task_id=task.id):
            self._organic_task_storage.fail_task(protocol_task=task, hotkey=hotkey, miner_uid=miner_uid)
            return

        if self._duel_task_storage.has_task(task_id=task.id):
            await self._duel_task_storage.fail_task(protocol_task=task, miner_uid=miner_uid, metagraph=metagraph)
            return

        self._synthetic_task_storage.fail_task(protocol_task=task, miner_uid=miner_uid)
