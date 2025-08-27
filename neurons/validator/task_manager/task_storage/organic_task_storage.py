import asyncio
import time
import traceback
from collections import deque

import bittensor as bt
import pybase64
from common.protocol import SubmitResults

from validator.duels.organic_judge_service import OrganicJudgeService
from validator.duels.ratings import DuelRatings
from validator.gateway.gateway_api import GetGatewayTasksResult
from validator.gateway.gateway_manager import GatewayManager
from validator.task_manager.task_storage.base_task_storage import BaseTaskStorage
from validator.task_manager.task_storage.organic_task import (
    AssignedMiner,
    GatewayOrganicTask,
    LegacyOrganicTask,
    OrganicTask,
    OrganicTaskJudgeQueuePriority,
)
from validator.validation_service import ValidationResponse


class OrganicTaskStorage(BaseTaskStorage):
    """Class responsible for storing organic tasks."""

    def __init__(
        self,
        *,
        config: bt.config,
        gateway_manager: GatewayManager,
        ratings: DuelRatings,
        wallet: bt.wallet | None = None,
    ) -> None:
        super().__init__(config=config, wallet=wallet)

        self._gateway_task_queue: deque[GatewayOrganicTask] = deque(maxlen=self._config.task.gateway.task_queue_size)
        """Queue of not-assigned organic tasks."""
        self._running_tasks: dict[str, OrganicTask] = {}
        """Organic tasks that are in progress (were assigned to any miner)."""
        self._running_task_queue: deque[OrganicTask] = deque()
        """Organic tasks that are waiting to be assigned to a miner."""
        self._gateway_manager = gateway_manager
        """Manager responsible for communication with gateways."""
        self._organic_task_expire_timeout: float = 2 * self._config.task.organic.task_timeout
        """Timeout for the task to be removed from the task manager."""
        self._ratings = ratings
        """Reference to the ratings structure that manages all miner ratings."""
        self._pending_judgment_queue: asyncio.PriorityQueue[tuple[OrganicTaskJudgeQueuePriority, OrganicTask]] = (
            asyncio.PriorityQueue()
        )
        """Tasks scheduled for judgment."""
        self._finished_judgment_queue: asyncio.Queue[OrganicTask] = asyncio.Queue()
        """Tasks with some judged results."""
        self._judge_service = OrganicJudgeService(
            config=self._config,
            pending_judgment_queue=self._pending_judgment_queue,
            finished_judgment_queue=self._finished_judgment_queue,
        )
        """Wrapper that communicates with the vllm server."""
        self._process_next_finished_job: asyncio.Task | None = None
        """Async task that processes results from the `_finished_judgment_queue`"""

        # TODO: deprecated
        self._legacy_task_queue: deque[LegacyOrganicTask] = deque()
        """Queue responsible for storing and managing legacy tasks."""

    def get_next_task(self, *, miner_uid: int, hotkey: str, is_strong_miner: bool) -> OrganicTask | None:
        """Returns the next task from the queue.
        Priority: active tasks, gateway tasks, legacy tasks."""

        current_time = time.time()
        self._remove_expired_tasks()

        task: OrganicTask | None = None
        for running_task in self._running_tasks.values():
            if hotkey in running_task.assigned_miners:
                continue
            if running_task.create_time + self._config.task.organic.task_timeout < current_time:
                continue
            if not running_task.should_be_assigned(is_strong_miner, self._config.task.organic.assigned_miners_count):
                continue
            task = running_task
            bt.logging.info(f"[{miner_uid}] received active organic task ({running_task.prompt[:100]})")
            break

        if self._config.task.gateway.enabled and task is None and len(self._gateway_task_queue) > 0:
            task = self._gateway_task_queue.popleft()
            if task is not None:
                bt.logging.info(
                    f"[{miner_uid}] received new gateway organic task "
                    f"({task.prompt[:100]}){' (strong miner)' if is_strong_miner else ''}"
                )
                self._running_tasks[task.id] = task
                self._running_task_queue.append(task)

        if task is None and len(self._legacy_task_queue) > 0:
            task = self._legacy_task_queue.popleft()
            if task is not None:
                bt.logging.info(f"[{miner_uid}] received new legacy organic task ({task.prompt[:100]})")
                self._running_tasks[task.id] = task
                self._running_task_queue.append(task)

        if task is not None:
            task.strong_miner_assigned = task.strong_miner_assigned or is_strong_miner
            miner = AssignedMiner(uid=miner_uid, hotkey=hotkey, assign_time=int(current_time))
            task.assigned_miners[hotkey] = miner
            if task.assign_time == 0:
                bt.logging.debug(
                    f"{current_time - task.create_time} seconds between created and assigned ({task.prompt[:100]})"
                )
                task.assign_time = current_time
                if isinstance(task, LegacyOrganicTask):
                    # TODO: deprecated
                    set_future(task.start_future, miner)

        return task

    async def start_judging_results(self) -> None:
        await self._judge_service.start_judging_duels()
        self._process_next_finished_job = asyncio.create_task(self._process_next_judged_task())

    def stop_judging_results(self) -> None:
        self._judge_service.stop_judging_duels()
        if self._process_next_finished_job is not None:
            self._process_next_finished_job.cancel()

    async def _process_next_judged_task(self) -> None:
        while True:
            try:
                task = await asyncio.wait_for(self._finished_judgment_queue.get(), timeout=1.0)
                self._finished_judgment_queue.task_done()

                if len(task.results_to_judge) >= 2:
                    await self._pending_judgment_queue.put((task.judge_queue_priority(), task))

                if not task.finalized and task.all_work_done(copies=self._config.task.organic.assigned_miners_count):
                    self._submit_task_result(task=task)
            except TimeoutError:
                await asyncio.sleep(1)
                continue
            except Exception as e:
                bt.logging.error(f"Processing task with finished judgment failed with: {e}")
                self._finished_judgment_queue.task_done()

    async def fetch_gateway_tasks_cron(self) -> None:
        """Fetches tasks from the best gateway and adds them to the task registry."""

        if not self._config.task.gateway.enabled:
            return

        while True:
            try:
                await self._fetch_tasks_from_gateway()
            except Exception as e:
                bt.logging.error(f"TaskManager: failed fetching gateway tasks cron: {e}")
            finally:
                await asyncio.sleep(self._config.task.gateway.task_fetch_interval)

    def get_task_by_id(self, *, task_id: str) -> OrganicTask | None:
        """Returns the task by id. Used in tests."""
        return self._running_tasks.get(task_id, None)

    # TODO: deprecated
    async def get_best_results(self, *, task_id: str) -> AssignedMiner | None:
        task = self._running_tasks.get(task_id, None)
        if task is None or not isinstance(task, LegacyOrganicTask):
            return None
        try:
            result = await asyncio.wait_for(task.best_result_future, self._config.task.organic.send_result_timeout)
        except TimeoutError:
            task = self._running_tasks.get(task_id, None)
            if task is None:
                return None
            result = task.finalize()
        finally:
            self._cleanup_task(task=task)
        return result

    # TODO: deprecated
    def add_legacy_task(self, *, task: LegacyOrganicTask) -> LegacyOrganicTask:
        bt.logging.info(f"Task added to the legacy task queue ({task.prompt[:100]})")
        self._legacy_task_queue.append(task)
        return task

    def has_task(self, *, task_id: str) -> bool:
        return task_id in self._running_tasks

    async def submit_result(
        self, *, synapse: SubmitResults, validation_res: ValidationResponse, miner_uid: int
    ) -> None:
        """Submits the result."""
        current_time = int(time.time())
        task_id = synapse.task.id
        task_prompt = synapse.task.prompt
        hotkey = synapse.dendrite.hotkey

        task = self._running_tasks.get(task_id, None)
        if task is None:
            bt.logging.error(f"[{miner_uid}] failed to submit result: task not found ({task_prompt[:100]})")
            return
        assigned_miner = task.assigned_miners.get(hotkey, None)
        if assigned_miner is None:
            bt.logging.error(f"[{miner_uid}] failed to submit result: assigned miner not found ({task_prompt[:100]})")
            return

        assigned_miner.compressed_result = synapse.results
        assigned_miner.grid_preview = validation_res.grid_preview
        assigned_miner.score = validation_res.score
        assigned_miner.rating = self._ratings.get_miner_reward_rating(miner_uid)
        assigned_miner.submit_time = current_time
        assigned_miner.finished = True

        bt.logging.info(
            f"[{miner_uid}] completed organic task. Rating: {assigned_miner.rating:.2f}. "
            f"Score: {validation_res.score:.2f} ({task_prompt[:100]})"
        )

        task.update_best(assigned_miner)

        if not task.is_duplicate_result(assigned_miner):
            task.queue_for_judgment(assigned_miner)

        if task.first_result_time == 0:
            self._process_first_result(task=task, assigned_miner=assigned_miner)

        if len(task.results_to_judge) >= 2:
            await self._pending_judgment_queue.put((task.judge_queue_priority(), task))

        if not task.finalized and task.all_work_done(copies=self._config.task.organic.assigned_miners_count):
            self._submit_task_result(task=task)

    def fail_task(self, *, task_id: str, task_prompt: str, hotkey: str, miner_uid: int) -> None:
        task = self._running_tasks.get(task_id, None)
        if task is None:
            bt.logging.error(f"[{miner_uid}] failed to fail task: task not found ({task_prompt[:100]})")
            return
        assigned_miner = task.assigned_miners.get(hotkey, None)
        if assigned_miner is None:
            bt.logging.error(f"[{miner_uid}] assigned miner not found for task ({task_prompt[:100]})")
            return

        assigned_miner.submit_time = int(time.time())
        assigned_miner.finished = True

        bt.logging.debug(f"[{miner_uid}] failed organic task ({task_prompt[:100]})")

        if not task.finalized and task.all_work_done(copies=self._config.task.organic.assigned_miners_count):
            self._submit_task_result(task=task)

    async def _fetch_tasks_from_gateway(self) -> None:
        self._remove_expired_tasks()
        task_count = self._config.task.gateway.task_queue_size - len(self._gateway_task_queue)
        if task_count == 0:
            return
        url = self._get_best_gateway_url()
        result = await self._gateway_manager.get_tasks(
            gateway_host=url, validator_hotkey=self._wallet.hotkey, task_count=task_count
        )
        bt.logging.trace(f"Fetched {len(result.tasks)} tasks from the gateway: {url}")
        self._add_tasks(gateway_url=url, gateway_task_result=result)
        self._update_gateway_scores(result=result, url=url)

    def _remove_expired_tasks(self) -> None:
        """Removes tasks from queue that are expired or were already removed."""
        current_time = time.time()
        for queue, name in (
            (self._running_task_queue, "running task queue"),
            (self._gateway_task_queue, "gateway task queue"),
            (self._legacy_task_queue, "legacy task queue"),
        ):
            while queue:
                task = queue[0]
                if task.create_time + self._organic_task_expire_timeout >= current_time:
                    break
                queue.popleft()
                self._submit_task_result(task=task)
                bt.logging.trace(f"({task.prompt[:100]}): removed from {name}.")

    def _get_best_gateway_url(self) -> str:
        best_gateway = self._gateway_manager.get_best_gateway()
        if best_gateway is None:
            bt.logging.info(f"Using bootstrap gateway: {self._config.task.gateway.bootstrap_gateway}")
            return str(self._config.task.gateway.bootstrap_gateway)

        return best_gateway.url

    def _add_tasks(self, *, gateway_url: str, gateway_task_result: GetGatewayTasksResult) -> None:
        for task in gateway_task_result.tasks:
            data = task.model_dump()
            self._gateway_task_queue.append(
                GatewayOrganicTask.create_task(
                    id=data["id"],
                    prompt=data["prompt"],
                    gateway_url=gateway_url,
                )
            )
            bt.logging.info(f"({data['prompt'][:100]}): added to the gateway task queue from {gateway_url}")

    def _update_gateway_scores(self, *, result: GetGatewayTasksResult, url: str) -> None:
        if not result.tasks:
            bt.logging.trace(f"Gateway {url} is disabled for the next iteration: no tasks returned.")
            for gateway in result.gateways:
                if gateway.url == url:
                    gateway.disabled = True
                    break
        self._gateway_manager.update_gateways(gateways=result.gateways)

    def _process_first_result(self, *, task: OrganicTask, assigned_miner: AssignedMiner) -> None:
        task.first_result_time = time.time()
        bt.logging.debug(
            f"{task.first_result_time - task.create_time} seconds "
            f"between created and first acceptable results received ({task.prompt[:100]})"
        )
        if isinstance(task, LegacyOrganicTask):
            # TODO: deprecated
            set_future(task.first_result_future, assigned_miner)
        elif isinstance(task, GatewayOrganicTask):
            if task.result_future is None:
                task.result_future = asyncio.get_event_loop().create_future()
            asyncio.get_event_loop().create_task(self._best_result_timeout_handler(task=task))

    async def _best_result_timeout_handler(self, *, task: GatewayOrganicTask) -> None:
        """Sends a result of the task to the gateway."""
        try:
            if task.result_future is not None:
                _ = await asyncio.wait_for(task.result_future, self._config.task.organic.send_result_timeout)
        except asyncio.CancelledError:
            pass  # Do nothing because result was submitted to the gateway in other place
        except TimeoutError:
            bt.logging.warning(f"Timeout to receive or judge all results. Current best is used ({task.prompt[:100]})")
            await self._send_best_result(task=task)

    def _submit_task_result(self, *, task: OrganicTask) -> None:
        bt.logging.debug(f"All work for organic task done ({task.prompt[:100]})")

        if isinstance(task, LegacyOrganicTask):
            # TODO: deprecated
            best_result = task.finalize()
            set_future(task.start_future, None)
            set_future(task.first_result_future, best_result)
            set_future(task.best_result_future, best_result)
            self._cleanup_task(task=task)
            return
        elif isinstance(task, GatewayOrganicTask):
            if task.result_future is not None:
                task.result_future.cancel()
            asyncio.get_event_loop().create_task(self._send_best_result(task=task))

    async def _send_best_result(self, *, task: GatewayOrganicTask) -> None:
        try:
            best_result = task.finalize()
            if best_result is None:
                error = f"{len(task.assigned_miners)} miners submitted results but no one is good"
                await self._gateway_manager.add_result(validator_hotkey=self._wallet.hotkey, task=task, error=error)
                bt.logging.warning(f"({task.prompt[:100]}): sent error to the gateway {task.gateway_url}: {error}")
            else:
                asset = self._get_asset(result=best_result, task_id=task.id)
                await self._gateway_manager.add_result(
                    validator_hotkey=self._wallet.hotkey,
                    task=task,
                    miner_hotkey=best_result.hotkey,
                    miner_uid=best_result.uid,
                    miner_rating=best_result.rating,
                    score=best_result.score,
                    asset=asset,
                )
                bt.logging.debug(f"({task.prompt[:100]}): sent best result to the gateway {task.gateway_url}")
        except Exception as ex:
            bt.logging.error(
                f"({task.prompt[:100]}): failed to send result to the gateway {task.gateway_url}: "
                f"{ex}  {traceback.format_exc()}"
            )
        finally:
            self._cleanup_task(task=task)

    def _get_asset(self, *, result: AssignedMiner, task_id: str) -> bytes | None:
        try:
            if result is not None and result.compressed_result is not None:
                return pybase64.b64decode(result.compressed_result)
        except Exception:
            bt.logging.error(f"({task_id}): error compressing result: {traceback.format_exc()}")

    def _cleanup_task(self, *, task: OrganicTask | None) -> None:
        if task is None:
            return
        try:
            self._running_tasks.pop(task.id, None)
            self._running_task_queue.remove(task)
            bt.logging.trace(f"({task.id}): removed.")
        except Exception:
            bt.logging.trace(f"({task.id}): error cleaning up task")


def set_future(future: asyncio.Future[AssignedMiner | None] | None, miner: AssignedMiner | None) -> None:
    """Force sets result to the future."""

    def do_set(f: asyncio.Future[AssignedMiner | None], results: AssignedMiner | None) -> None:
        if not f.done() and not f.cancelled():
            f.set_result(results)

    if future is not None:
        future.get_loop().call_soon_threadsafe(do_set, future, miner)
