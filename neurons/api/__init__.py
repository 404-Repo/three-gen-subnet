from enum import IntEnum

import bittensor as bt


class PollStatus(IntEnum):
    NOT_FOUND = 404
    RATE_LIMIT = 429
    IN_QUEUE = 100
    # I couldn't resist the temptation to use HTTP Status Codes as values. Couldn't find a better code for `IN_QUEUE`.
    IN_PROGRESS = 102
    DONE = 200
    FAILED = 500


class Generate(bt.Synapse):
    """Subnet client requesting the generation."""

    prompt: str
    # Prompt to use for 3D generation.
    task_id: str | None = None
    # ID assigned to the task by a validator.
    poll_interval: int = 30
    # Allowed interval for polling task results, seconds. Defined by a validator.


class PollResults(bt.Synapse):
    """Subnet client querying the task status."""

    task_id: str
    status: PollStatus | None = None
    results: str | None = None
    # Binary content of the generated 3D asset, provided when task is done (PollStatus.Done).
