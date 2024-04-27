import time
import uuid
from collections import deque

import bittensor as bt
from api import TaskStatus
from pydantic import BaseModel, Field


class AssignedMiner(BaseModel):
    hotkey: str
    results: str | None = None
    score: float = 0


class OrganicTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # Unique id
    create_time: float = Field(default_factory=time.time)
    # Timestamp reflecting task creation.
    prompt: str
    # Prompt to use for generation.
    wallet: str
    # Task requester. Hot key.

    assigned: dict[str, AssignedMiner] = Field(default_factory=dict)
    # Miners assigned for the job.
    assigned_time: float = 0.0
    # First assignment time.

    strong_miner_assigned: bool = False
    # True if the strong miner is amongst assigned miners.
    acceptable_results_time: float = 0
    # Time of the first acceptable results

    last_status_check: float = 0.0
    # Last time task status was requested. Used for rate limiting.


def _get_best_results(task: OrganicTask) -> str | None:
    best: AssignedMiner | None = None
    for miner in task.assigned.values():
        if miner.results is None:
            continue
        if best is None:
            best = miner
            continue
        if miner.score > best.score:
            best = miner
    return best.results if best is not None else None


class TaskRegistry:
    def __init__(
        self,
        queue_size: int = 0,
        copies: int = 4,
        wait_after_first_copy: int = 60,
        task_timeout: int = 600,
        polling_interval: int = 30,
    ) -> None:
        """
        Args:
        - queue_size (int): The maximum size of the task queue. A zero value implies an unbounded queue.
        - copies (int): Number of copies to generate to chose the best from.
        - wait_after_first_copy (int): Maximum wait time for the second results after the first acceptable results
            were generated.
        - task_timeout (int): Time limit for submitting tasks (in seconds).
        - polling_interval (int): Minimum interval between status checks (in seconds).

        The TaskRegistry manages organic task received by this validator.
        """
        self._tasks: dict[str, OrganicTask] = {}
        self._queue: deque[str] = deque()
        self._queue_size = queue_size
        self._copies = copies
        self._wait_after_first_copy = wait_after_first_copy
        self._task_timeout = task_timeout
        self._polling_interval = polling_interval

    def is_queue_full(self) -> bool:
        """Returns True if the task queue is full, otherwise False."""
        return 0 < self._queue_size <= len(self._tasks)

    def add_task(self, prompt: str, wallet: str) -> str:
        """
        Add new task to the registry.

        Args:
        - prompt: prompt to use for generation.
        - wallet: task requester. Hot key.
        """

        task = OrganicTask(prompt=prompt, wallet=wallet)
        self._tasks[task.id] = task
        self._queue.append(task.id)

        bt.logging.trace(f"New organic task added: {task.id}")

        return task.id

    def get_next_task(self, hotkey: str, is_strong_miner: bool = True) -> OrganicTask | None:
        """
        Returns the next in queue organic task.

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

            if task.acceptable_results_time > 0:
                # At least one miner has generated acceptable results
                continue

            if task.create_time + self._task_timeout < current_time:
                # Too early to remove the task, too late to assign the task
                continue

            if (not is_strong_miner or task.strong_miner_assigned) and (len(task.assigned) >= self._copies):
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

            del self._tasks[task.id]
            self._queue.popleft()

    def fail_task(self, task_id: str, hotkey: str) -> None:
        """
        Miner failed to deliver acceptable results.

        Args:
        - task_id (str): The unique identifier of the task.
        - hotkey (str): Hotkey of the miner.
        """
        bt.logging.trace(f"Miner ({hotkey}) failed the task: {task_id}")

        task = self._tasks.get(task_id, None)
        if task is None:
            return
        task.assigned.pop(hotkey, None)

    def complete_task(self, task_id: str, hotkey: str, results: str, score: float) -> None:
        """
        Miner finished the task

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

        bt.logging.trace(f"[{hotkey}] completed the task: {task_id} with: {score:.2f}")

        miner.results = results
        if task.acceptable_results_time == 0:
            task.acceptable_results_time = time.time()
            bt.logging.debug(
                f"{task.acceptable_results_time - task.create_time} "
                f"seconds between task started and first acceptable results received"
            )

    def is_valid_requester(self, task_id: str, hotkey: str) -> bool:
        task = self._tasks.get(task_id, None)
        if task is None:
            return False
        if task.wallet != hotkey:
            return False
        return True

    def get_task_status(self, task_id: str, hotkey: str) -> tuple[TaskStatus, str | None]:
        task = self._tasks.get(task_id, None)
        if task is None:
            return TaskStatus.NOT_FOUND, None
        if task.wallet != hotkey:
            return TaskStatus.NOT_FOUND, None

        current_time = time.time()
        if task.last_status_check + self._polling_interval > current_time:
            return TaskStatus.RATE_LIMIT, None
        task.last_status_check = current_time
        if not task.assigned:
            return TaskStatus.IN_QUEUE, None
        if task.acceptable_results_time == 0:
            return TaskStatus.IN_PROGRESS, None
        if task.acceptable_results_time + self._wait_after_first_copy < current_time:
            return TaskStatus.DONE, _get_best_results(task)
        if all(miner.results is not None for miner in task.assigned.values()):
            return TaskStatus.DONE, _get_best_results(task)

        return TaskStatus.IN_PROGRESS, None
