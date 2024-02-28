from typing import Literal

import bittensor as bt


class TGHandshakeV1(bt.Synapse):
    """
    Serves as both the initial and periodic communication between a Miner and a Validator.
    Facilitates the discovery of active neurons within the subnet and assesses the operational
    capacity concerning the number of generation endpoints maintained by the miner.
    Note: Only a single generation endpoint is supported in the current implementation.
    """

    active_generation_endpoints: int | None = None  # The number of active generation endpoints managed by the miner.


class TGTaskV1(bt.Synapse):
    """
    Represents a task for generating 3D content, assigned from a validator to a miner.
    """

    prompt: str  # Prompt to use for 3D generation
    task_id: str  # Task identifier for tracking and status updates.
    error: Literal["QUEUE_FULL"] | None = None


class TGPollV1(bt.Synapse):
    """
    Used for querying the current status of a 3D generation task.
    This request facilitates periodic checking by the validator to monitor task progression.
    """

    task_id: str
    status: Literal["NOT FOUND", "FORBIDDEN", "IN QUEUE", "IN PROGRESS", "DONE", "FAILED"] | None = None
    results: bytes | None = None  # Binary content of the generated 3D model, provided when task is "DONE".


class TGJobDoneCallbackV1(bt.Synapse):
    """
    Notification sent from a miner to a validator to signal the completion of a 3D generation task.
    This callback mechanism allows for asynchronous alerting upon task finalization.
    """

    task_id: str  # The identifier of the completed task.
