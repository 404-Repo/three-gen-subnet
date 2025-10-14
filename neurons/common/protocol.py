import uuid
from functools import cached_property
from typing import Annotated, Any, Literal

import bittensor as bt
import blake3
from pydantic import BaseModel, Field


class TextTask(BaseModel):
    type: Literal["text"] = "text"
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this task",
    )
    prompt: str = Field(default="", description="Prompt text to use for 3D model generation")

    @cached_property
    def log_id(self) -> str:
        if len(self.prompt) <= 100:
            return self.prompt
        return f"{self.prompt[:97]}..."


class ImageTask(BaseModel):
    type: Literal["image"] = "image"
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this task",
    )
    prompt: str = Field(
        default="",
        description="Base64-encoded image data for 3D model generation",
    )

    @cached_property
    def prompt_hash(self) -> str:
        return blake3.blake3(self.prompt.encode()).hexdigest()

    @cached_property
    def log_id(self) -> str:
        return f"image:{self.prompt_hash}"


ProtocolTask = TextTask | ImageTask

# ProtocolTask = Annotated[TextTask | ImageTask, Field(discriminator="type")]

ProtocolTaskType = Literal["text", "image"]


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

    task: ProtocolTask | None = Field(default=None, description="Task assigned by validator to be completed by miner")
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

    task: ProtocolTask | None = Field(
        default=None,
        description="[DEPRECATED] The original task miner is submitting results for. "
        "Will be removed in the next release.",
    )
    task_id: str = Field(default="", description="Unique identifier of the task these results are for")
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

    def model_post_init(self, __context: Any) -> None:
        # Forward compatibility: if task_id not provided, extract from task
        if self.task_id == "" and self.task is not None:
            self.task_id = self.task.id


class GetVersion(bt.Synapse):
    """Neuron version request."""

    version: int = Field(default=0, description="Neuron version")
    validation_version: str = Field(default="", description="Validation endpoint version")
