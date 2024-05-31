import asyncio
import time
import uuid
from collections import deque

import bittensor as bt
from pydantic import BaseModel, Field


class AssignedMiner(BaseModel):
    hotkey: str
    results: str | None = None
    score: float = 0
    finished: bool = False


class OrganicTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # Unique id
    create_time: float = Field(default_factory=time.time)
    # Task registration time.
    prompt: str
    # Prompt to use for generation.

    assigned: dict[str, AssignedMiner] = Field(default_factory=dict)
    # Miners assigned for the job.
    assigned_time: float = 0.0
    # First assignment time.

    strong_miner_assigned: bool = False
    # True if the strong miner is amongst assigned miners.
    first_results_time: float = 0

    # Time of the first acceptable results

    def get_best_results(self) -> AssignedMiner | None:
        best: AssignedMiner | None = None
        for miner in self.assigned.values():
            if miner.results is None:
                continue
            if best is None:
                best = miner
                continue
            if miner.score > best.score:
                best = miner
        return best

    def should_be_assigned(self, strong_miner: bool, copies: int) -> bool:
        return (strong_miner and not self.strong_miner_assigned) or len(self.assigned) < copies

    def all_miners_finished(self, copies: int) -> bool:
        return all(miner.finished for miner in self.assigned.values()) and not self.should_be_assigned(
            strong_miner=True, copies=copies
        )


def _set_future_result(future: asyncio.Future, results: AssignedMiner | None) -> None:
    if not future.done() and not future.cancelled():
        future.set_result(results)


class TaskRegistry:
    def __init__(
        self, queue_size: int = 0, copies: int = 4, wait_after_first_copy: int = 30, task_timeout: int = 300
    ) -> None:
        """
        Args:
        - queue_size (int): The maximum size of the task queue. A zero value implies an unbounded queue.
        - copies (int): Number of copies to generate to choose the best from.
        - wait_after_first_copy (int): Maximum wait time for the second results after the first acceptable results
            were generated.
        - task_timeout (int): Time limit for submitting tasks (in seconds).

        The TaskRegistry manages organic task received by this validator.
        """
        self._tasks: dict[str, OrganicTask] = {}
        self._first_results_futures: dict[str, asyncio.Future[AssignedMiner | None]] = {}
        # Set when the first acceptable results received. The first results are set.
        self._best_results_futures: dict[str, asyncio.Future[AssignedMiner | None]] = {}
        # Set when there will be no more results and the best results are picked. The best results are set.
        self._queue: deque[str] = deque()
        self._queue_size = queue_size
        self._copies = copies
        self._wait_after_first_copy = wait_after_first_copy
        self._task_timeout = task_timeout

    @property
    def is_queue_full(self) -> bool:
        """Returns True if the task queue is full, otherwise False."""
        return 0 < self._queue_size <= len(self._tasks)

    @property
    def wait_after_first_copy(self) -> int:
        return self._wait_after_first_copy

    def add_task(self, prompt: str) -> str:
        """
        Add new task to the registry.

        Args:
        - prompt: prompt to use for generation.
        """

        task = OrganicTask(prompt=prompt)
        self._tasks[task.id] = task

        first_future = asyncio.get_event_loop().create_future()
        self._first_results_futures[task.id] = first_future

        best_future = asyncio.get_event_loop().create_future()
        self._best_results_futures[task.id] = best_future
        self._queue.append(task.id)

        bt.logging.trace(f"New organic task added ({task.id}): {task.prompt}")

        return task.id

    async def get_first_results(self, task_id: str) -> AssignedMiner | None:
        """
        Wait for the first acceptable results and returns it.
        """
        if task_id not in self._first_results_futures:
            return None
        return await self._first_results_futures[task_id]

    async def get_best_results(self, task_id: str) -> AssignedMiner | None:
        """
        Wait for the best received results and returns it.
        """
        if task_id not in self._best_results_futures:
            return None

        try:
            return await asyncio.wait_for(self._best_results_futures[task_id], self._wait_after_first_copy)
        except TimeoutError:
            task = self._tasks.get(task_id, None)
            if task is None:
                return None
            return task.get_best_results()

    def get_next_task(self, hotkey: str, is_strong_miner: bool = True) -> OrganicTask | None:
        """
        Return the next in queue organic task.

        Args:
        - hotkey: hotkey of the miner requesting the task.
        - is_strong_miner: indicator whether miner is int the top miners.
        """
        self._remove_expired_tasks()

        current_time = time.time()

        for task_id in self._queue:
            task = self._tasks.get(task_id, None)
            if task is None:
                continue

            if task.create_time + self._task_timeout < current_time:
                # Too early to remove the task, too late to assign the task
                continue

            if not task.should_be_assigned(is_strong_miner, self._copies):
                continue

            if hotkey in task.assigned:
                continue

            task.strong_miner_assigned = task.strong_miner_assigned or is_strong_miner
            task.assigned[hotkey] = AssignedMiner(hotkey=hotkey)

            if task.assigned_time == 0:
                bt.logging.debug(f"{current_time - task.create_time} seconds between task created and task assigned")
                task.assigned_time = current_time

            bt.logging.trace(f"Next task to give: {task.id}")

            return task

        return None

    def _remove_expired_tasks(self) -> None:
        current_time = time.time()
        while self._queue:
            task = self._tasks.get(self._queue[0], None)
            if task is None:
                self._queue.popleft()
                continue

            if task.create_time + 2 * self._task_timeout >= current_time:
                break

            bt.logging.trace(f"Removing task from the registry: {task.id}")

            self.clean_task(task_id=task.id)
            self._queue.popleft()

    def complete_task(self, task_id: str, hotkey: str, results: str, score: float) -> None:
        """
        Miner finished the task.

        Args:
        - task_id (str): The unique identifier of the task.
        - hotkey (str): Hotkey of the miner.
        - results (str): encoded binary with the results.
        - score (float): validation score.
        """
        task = self._tasks.get(task_id, None)
        if task is None:
            return

        miner = task.assigned.get(hotkey, None)
        if miner is None:
            return

        bt.logging.trace(f"[{hotkey}] completed the task ({task_id}): {task.prompt}. Score: {score:.2f}")

        miner.results = results
        miner.finished = True

        if task.first_results_time == 0:
            task.first_results_time = time.time()
            bt.logging.debug(
                f"{task.first_results_time - task.create_time} "
                f"seconds between task started and first acceptable results received"
            )
            self._set_first_future(miner, task.id)

        if task.all_miners_finished(self._copies):
            bt.logging.debug(f"All miners submitted results for the task ({task_id}): {task.prompt}")
            best_results = task.get_best_results()
            self._set_best_future(best_results, task.id)

    def _set_first_future(self, miner: AssignedMiner | None, task_id: str) -> None:
        first_future = self._first_results_futures.get(task_id, None)
        if first_future is not None:
            first_future.get_loop().call_soon_threadsafe(_set_future_result, first_future, miner)

    def _set_best_future(self, miner: AssignedMiner | None, task_id: str) -> None:
        best_future = self._best_results_futures.get(task_id, None)
        if best_future is not None:
            best_future.get_loop().call_soon_threadsafe(_set_future_result, best_future, miner)

    def fail_task(self, task_id: str, hotkey: str) -> None:
        """
        Miner failed the task.

        Args:
        - task_id (str): The unique identifier of the task.
        - hotkey (str): Hotkey of the miner.
        """
        task = self._tasks.get(task_id, None)
        if task is None:
            return

        miner = task.assigned.get(hotkey, None)
        if miner is None:
            return

        bt.logging.trace(f"[{hotkey}] failed the task ({task_id}): {task.prompt}.")

        miner.finished = True
        if task.all_miners_finished(self._copies):
            best_results = task.get_best_results()
            self._set_first_future(best_results, task_id)
            self._set_best_future(best_results, task_id)

    def clean_task(self, task_id: str) -> None:
        """
        Removes task results as no longer needed.

        Args:
        - task_id (str): The unique identifier of the task.
        """
        self._tasks.pop(task_id, None)
        self._first_results_futures.pop(task_id, None)
        self._best_results_futures.pop(task_id, None)
