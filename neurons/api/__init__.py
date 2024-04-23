from enum import IntEnum

import bittensor as bt


class TaskStatus(IntEnum):
    """Status codes indicating the progression state of a task."""

    IN_QUEUE = 100  # Task is queued.
    IN_PROGRESS = 102  # Task is currently being processed.
    DONE = 200  # Task completed successfully.
    FAILED = 500  # Task failed due to an error.
    NOT_FOUND = 404  # No task found with the given ID.
    RATE_LIMIT = 429  # Rate limit exceeded, too many requests.


class Generate(bt.Synapse):
    """Subnet client requesting the generation."""

    prompt: str  # Prompt to use for 3D generation.
    task_id: str | None = None  # Task identifier assigned by the validator.
    polling_interval: int = 30  # Allowed interval for polling results, in seconds.


class StatusCheck(bt.Synapse):
    """Subnet client querying the task status."""

    task_id: str
    status: TaskStatus | None = None  # Current task status
    results: str | None = None  # Generated assets, provided when the task is done (TaskStatus.Done).
