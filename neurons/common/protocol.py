import uuid

import bittensor as bt
from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Unique task identifier.
    prompt: str = Field(default="")  # Prompt to use for 3D generation.


class Feedback(BaseModel):
    validation_failed: bool = False  # Set if validation failed.
    task_fidelity_score: float = 0.0  # Calculated fidelity score for the given task.
    average_fidelity_score: float = 0.0  # Average of all computed fidelity scores.
    generations_within_the_window: int = (
        0  # Total accepted generations (non-zero fidelity score) within the last 4 hours.
    )
    current_miner_reward: float = 0  # Recent miners reward value.


class PullTask(bt.Synapse):
    """Miner requesting a new task from the validator."""

    task: Task | None = None  # Task assigned by validator to be completed by miner.

    # Minimum score required for task results to be accepted by validators.
    # Results below this threshold are rejected, penalizing the miner.
    # Miners can submit empty results to avoid penalties if unable to meet threshold.
    validation_threshold: float = 0.6

    # Minimum expected time (in seconds) for task completion.
    # Used to calculate effective cooldown:
    # - Faster completion results in longer cooldown
    # - Cooldown is reduced by actual completion time, up to throttle_period
    # Example: With 60s cooldown and 20s throttle_period:
    # - 5s completion -> 55s cooldown (reduced by 5s)
    # - 15s completion -> 45s cooldown (reduced by 15s)
    # - 20s completion -> 40s cooldown (reduced by full throttle_period)
    # - 30s completion -> 40s cooldown (still reduced by throttle_period max)
    throttle_period: int = 0

    cooldown_until: int = 0  # Unix timestamp indicating when miner can pull the next task from this validator.
    cooldown_violations: int = 0  # Count of miner's failures to respect the mandatory cooling period.


class SubmitResults(bt.Synapse):
    """Miner submitting generation results."""

    task: Task = Field(default_factory=Task)  # The original task miner is submitting results for.
    results: str  # Generated assets, encoded as a string.

    data_format: str = "ply"  # Reserved for future use.
    data_ver: int = 0  # Reserved for future use.
    compression: int = 0  # Experimental feature.

    submit_time: int  # time.time_ns()

    # Miner signature:
    # b64encode(sign(f'{MINER_LICENSE_CONSENT_DECLARATION}{submit_time}{prompt}{validator.hotkey}{miner.hotkey}'))
    signature: str

    feedback: Feedback | None = None  # Feedback provided by a validator.
    cooldown_until: int = 0  # UTC time indicating when the miner is allowed to pull the next task from this validator.


class GetVersion(bt.Synapse):
    """Neuron version request."""

    version: int = 0  # neuron version
    validation_version: str = ""  # validation endpoint version
