import uuid
from typing import Literal

import bittensor as bt
from pydantic import BaseModel, Field


class Version(BaseModel):
    major: int
    minor: int
    patch: int


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Unique task identifier.
    prompt: str  # Prompt to use for 3D generation.


class Feedback(BaseModel):
    fidelity_score: float  # Calculated fidelity score.
    avg_fidelity_score: float  # Average fidelity score.


class PollTask(bt.Synapse):
    """Miner requesting a new task from the validator."""

    version: Version | None = None
    # Current miner version
    task: Task | None = None
    # Task to be filled by validator.


class SubmitResults(bt.Synapse):
    """Miner submitting generation results."""

    task: Task | None
    # The original task associated with these results.
    results: str
    #  Binary content of the generated 3D model, encoded as a string.
    feedback: Feedback | None = None
    # Feedback provided by a validator.
    cooldown_until: int = 0
    # UTC time indicating when the miner is allowed to pull the next task from this validator.


class Generate(bt.Synapse):
    """Subnet client requesting the generation."""

    prompt: str
    # Prompt to use for 3D generation.
    task_id: str | None = None
    # ID assigned to the task by the validator.
    poll_interval: int = 30
    # Allowed interval for polling task results, seconds. Defined by the validator.


class PollResults(bt.Synapse):
    """Subnet client querying the task status."""

    task_id: str
    status: Literal["NOT FOUND", "RATE LIMIT", "IN QUEUE", "IN PROGRESS", "DONE", "FAILED"] | None = None
    results: str | None = None
    # Binary content of the generated 3D model, provided when task is "DONE"
