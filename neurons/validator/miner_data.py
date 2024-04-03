import time
from collections import deque

import bittensor as bt
from common.protocol import Task
from pydantic import BaseModel, Field


class MinerData(BaseModel):
    uid: int

    observations: deque[int] = Field(default_factory=deque)
    """Observation window containing task finish times (seconds), limited to the last 8 hours."""

    fidelity_score: float = 0.0
    """Exponential moving average (EMA) of the fidelity score."""

    assigned_task: Task | None = None
    """The task currently assigned to the miner, if any."""

    assignment_time: float | None = None
    """Task assignment time."""

    def reset_task(self) -> None:
        self.assigned_task = None
        self.assignment_time = None

    def assign_task(self, task: Task) -> None:
        self.assigned_task = task
        self.assignment_time = time.time()

    def add_observation(self, task_finish_time: int, fidelity_score: float, moving_average_alpha: float) -> None:
        self.observations.append(task_finish_time)
        prev_fidelity_score = self.fidelity_score
        self.fidelity_score = prev_fidelity_score * (1 - moving_average_alpha) + moving_average_alpha * fidelity_score

        bt.logging.trace(
            f"[{self.uid}] score: {prev_fidelity_score:.2f} -> {self.fidelity_score:.2f}."
            f" Observations: {len(self.observations)}"
        )

    def is_task_expired(self, expiration_time: int) -> bool:
        if self.assignment_time is None:
            return False  # No task assigned, hence no expiry.
        return time.time() > self.assignment_time + expiration_time

    def is_on_cooldown(self, cooldown_time: int) -> bool:
        if not self.observations:
            return False  # No previous history
        return time.time() < self.observations[-1] + cooldown_time

    def _expire_observations(self, current_time: int, observation_window: int = 8 * 60 * 60) -> None:
        expiration_threshold = current_time - observation_window
        while self.observations and self.observations[0] < expiration_threshold:
            self.observations.popleft()

    def calculate_reward(self, current_time: int, observation_window: int = 8 * 60 * 60) -> float:
        self._expire_observations(current_time, observation_window)
        return len(self.observations) * self.fidelity_score
