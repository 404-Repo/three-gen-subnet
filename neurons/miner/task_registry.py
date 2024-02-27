import time
from queue import PriorityQueue

import bittensor as bt
from pydantic import BaseModel, Field

from common import synapses


class Task(BaseModel):
    id: str
    # ID sent from the validator.
    prompt: str
    # Prompt to use for generation.
    validator_hotkey: str
    # Hotkey of the validator sent the task.
    results: bytes | None = None
    # Generation results.
    in_progress: bool = False
    # Flag indicating whether task execution has started.
    failed: bool = False
    # Flag indicating whether task execution has failed.
    create_time: float = Field(default_factory=time.time)
    # Timestamp reflecting task creation.
    start_time: float = 0
    # Timestamp recording when task execution starts; initially zero.
    finish_time: float = 0
    # Timestamp recording when task execution completes; initially zero.


class TaskRegistry:
    def __init__(self, queue_size: int = 0):
        """
        Args:
        - queue_size (int): The maximum size of the task queue. A zero value implies an unbounded queue.

        The TaskRegistry manages task scheduling through a priority queue (asyncio.PriorityQueue),
        enabling priority-based task execution and supports tracking tasks from creation through completion.
        """
        self._tasks: dict[str, Task] = {}
        self._queue = PriorityQueue(maxsize=queue_size)

    def is_queue_full(self) -> bool:
        """Returns True if the task queue is full, otherwise False."""
        return self._queue.full()

    def add_task(self, syn: synapses.TGTaskV1, validator_stake: float):
        """
        Asynchronously registers a new task in the system with given parameters, using the validator's stake
        as a priority indicator, and enqueues it for execution.

        Args:
        - syn (synapses.TGTaskV1): synapse received from the validator
        - validator_stake (float): The validator's stake, used to prioritize the task in the queue.
        """

        # TODO: add tasks TTL

        task = Task(id=syn.task_id, prompt=syn.prompt, validator_hotkey=syn.dendrite.hotkey)
        if task.id in self._tasks:
            bt.logging.warning(f"Duplicate task id ({task.id}) received from {task.validator_hotkey}")
            return
        self._tasks[task.id] = task
        self._queue.put((-validator_stake, task.id))

    def start_next_task(self) -> Task:
        """
        Removes and returns the highest priority task from the queue, marking it as in progress.

        Returns:
        - Task: The task instance that is starting its execution.
        """

        task = None
        while task is None:
            _, task_id = self._queue.get()
            task = self._tasks.get(task_id, None)

        task.start_time = time.time()
        task.in_progress = True

        bt.logging.debug(f"{task.start_time - task.create_time} seconds between task received and task started")

        return task

    def fail_task(self, task_id: str) -> None:
        """
        Marks the specified task as failed.

        Args:
        - task_id (str): The unique identifier of the task to mark as complete.
        - results (bytes): The binary data produced by executing the task.

        If no task matches the given `task_id`, logs a warning.
        """
        task = self._tasks.get(task_id, None)
        if task is None:
            bt.logging.warning(f"Unknown task has failed ({task_id})")
            return
        task.in_progress = False
        task.failed = True

    def complete_task(self, task_id: str, results: bytes) -> None:
        """
        Marks the specified task as completed, recording its output and the completion timestamp.

        Args:
        - task_id (str): The unique identifier of the task to mark as complete.
        - results (bytes): The binary data produced by executing the task.

        If no task matches the given `task_id`, logs a warning.
        """

        task = self._tasks.get(task_id, None)
        if task is None:
            bt.logging.warning(f"Unknown task is completed ({task_id})")
            return
        task.in_progress = False
        task.results = results
        task.finish_time = time.time()

        bt.logging.debug(f"{task.finish_time - task.start_time} seconds between task started and task finished")
        bt.logging.debug(f"{task.finish_time - task.create_time} seconds between task received and task finished")

    def remove_task(self, task_id: str) -> None:
        """
        Removes a task from the registry, typically after its results have been
        successfully sent to the validator or if the task should be canceled.

        Args:
        - task_id (str): The unique identifier of the task to remove.
        """
        self._tasks.pop(task_id, None)

    def get_task(self, task_id: str) -> Task | None:
        """
        Returns the registered task, if available.

        Args:
        - task_id (str): The unique identifier of the task.

        Returns:
        - Task: Task or a None if the task is not found.
        """
        return self._tasks.get(task_id, None)
