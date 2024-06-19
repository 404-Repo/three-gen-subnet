import uuid

import bittensor as bt
from pydantic import BaseModel, Field


class Version(BaseModel):
    major: int
    minor: int
    patch: int

    def __int__(self) -> int:
        return self.patch + self.minor * 100 + self.major * 10000


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Unique task identifier.
    prompt: str  # Prompt to use for 3D generation.


class Feedback(BaseModel):
    validation_failed: bool = False  # Set if validation failed.
    task_fidelity_score: float = 0.0  # Calculated fidelity score for the given task.
    average_fidelity_score: float = 0.0  # Average of all computed fidelity scores.
    generations_within_8_hours: int = 0  # Total accepted generations (non-zero fidelity score) within the last 8 hours.
    current_miner_reward: float = 0  # Recent miners reward value.


class PullTask(bt.Synapse):
    """Miner requesting a new task from the validator."""

    version: Version | None = None  # Current validator version.
    task: Task | None = None  # Task to be filled by validator.
    submit_before: int = 0  # Allocated time to submit the results.

    cooldown_until: int = 0  # UTC time indicating when the miner is allowed to pull the next task from this validator.
    cooldown_violations: int = 0  # Number of times a miner has failed to respect the mandatory cooling period.


class SubmitResults(bt.Synapse):
    """Miner submitting generation results."""

    task: Task | None  # The original task miner is submitting results for.
    results: str  # Generated assets, encoded as a string.

    data_format: str = "hdf5"  # Results data format. Supported formats: "hdf5" and "ply".
    data_ver: int = 0  # Version of the hdf5 data format. 0 - default (dream gaussian).

    submit_time: int  # time.time_ns()
    signature: str  # Miner signature: b64encode(sign(f'{submit_time}{prompt}{validator.hotkey}{miner.hotkey}'))

    feedback: Feedback | None = None  # Feedback provided by a validator.
    cooldown_until: int = 0  # UTC time indicating when the miner is allowed to pull the next task from this validator.


class GetVersion(bt.Synapse):
    """Neuron version request."""

    version: int = 0  # neuron version
    validation_version: str = ""  # validation endpoint version
