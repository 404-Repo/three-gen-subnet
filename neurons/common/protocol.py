from typing import Literal

import bittensor as bt


class TGTask(bt.Synapse):
    """
    Represents a task for generating 3D content, assigned from a validator to a miner.
    """

    prompt: str  # Prompt to use for 3D generation
    task_id: str  # Task identifier for tracking and status updates.
    status: Literal["IN QUEUE", "ERROR_QUEUE_FULL"] | None = None


class TGPoll(bt.Synapse):
    """
    Used for querying the current status of a 3D generation task.
    This request facilitates periodic checking by the validator to monitor task progression.
    """

    task_id: str
    status: Literal["NOT FOUND", "FORBIDDEN", "IN QUEUE", "IN PROGRESS", "DONE", "FAILED"] | None = None
    results: str | None = None  # Binary content of the generated 3D model, provided when task is "DONE".


class TGJobDoneCallback(bt.Synapse):
    """
    Notification sent from a miner to a validator to signal the completion of a 3D generation task.
    This callback mechanism allows for asynchronous alerting upon task finalization.
    """

    task_id: str  # The identifier of the completed task.
