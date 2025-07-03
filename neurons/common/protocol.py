import uuid

import bittensor as bt
from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique task identifier automatically generated using UUID4",
    )
    prompt: str = Field(default="", description="Prompt text to use for 3D model generation")


class Feedback(BaseModel):
    """Feedback model for miner task validation and performance tracking."""

    validation_failed: bool = Field(
        default=False, description="Indicates whether the validation process failed for this task"
    )
    task_fidelity_score: float = Field(
        default=0.0, description="Task fidelity score (0.0-1.0) measuring how faithfully the miner completed the task."
    )
    average_fidelity_score: float = Field(
        default=0.0, description="Exponential moving average of all computed fidelity scores for this miner"
    )
    generations_within_the_window: int = Field(
        default=0, description="Total accepted generations within the last 4 hours"
    )
    current_duel_rating: float = Field(default=0.0, description="Glicko2 duels rating")
    current_miner_reward: float = Field(default=0.0, description="Most recent reward value calculated for the miner")


class PullTask(bt.Synapse):
    """Miner requesting a new task from the validator."""

    task: Task | None = Field(default=None, description="Task assigned by validator to be completed by miner")
    validation_threshold: float = Field(
        default=0.6,
        description="Minimum fidelity score required for task results to be accepted by validators. "
        "Results below this threshold are rejected, penalizing the miner. "
        "Miners can submit empty results to avoid penalties if unable to meet threshold",
    )
    throttle_period: int = Field(
        default=0,
        description="Minimum expected time (in seconds) for task completion. "
        "Used to calculate effective cooldown: "
        "    - Faster completion results in longer cooldown "
        "    - Cooldown is reduced by actual completion time, up to throttle_period "
        "Example: With 60s cooldown and 20s throttle_period: "
        "    - 5s completion -> 55s cooldown (reduced by 5s) "
        "    - 15s completion -> 45s cooldown (reduced by 15s) "
        "    - 20s completion -> 40s cooldown (reduced by full throttle_period) "
        "    - 30s completion -> 40s cooldown (still reduced by throttle_period max)",
    )
    cooldown_until: int = Field(
        default=0, description="Unix timestamp indicating when miner can pull the next task from this validator"
    )
    cooldown_violations: int = Field(
        default=0, description="Count of miner's failures to respect the mandatory cooling period"
    )


class SubmitResults(bt.Synapse):
    """Miner submitting generation results."""

    task: Task = Field(default_factory=Task, description="The original task miner is submitting results for")
    results: str = Field(description="Generated assets, encoded as a string")
    data_format: str = Field(default="ply", description="Reserved for future use")
    data_ver: int = Field(default=0, description="Reserved for future use")
    compression: int = Field(
        default=2,
        description="Reserved for future use. Only spz compressed data is allowed (https://github.com/404-Repo/spz)",
    )
    submit_time: int = Field(description="Submission timestamp in nanoseconds (time.time_ns())")
    signature: str = Field(
        description="Miner signature: "
        "b64encode(sign("
        "f'{MINER_LICENSE_CONSENT_DECLARATION}{submit_time}{prompt}{validator.hotkey}{miner.hotkey}'"
        "))"
    )
    feedback: Feedback | None = Field(default=None, description="Feedback provided by a validator")
    cooldown_until: int = Field(
        default=0, description="UTC time indicating when the miner is allowed to pull the next task from this validator"
    )


class GetVersion(bt.Synapse):
    """Neuron version request."""

    version: int = Field(default=0, description="Neuron version")
    validation_version: str = Field(default="", description="Validation endpoint version")
