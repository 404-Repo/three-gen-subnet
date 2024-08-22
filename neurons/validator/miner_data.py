import time
from collections import deque

import bittensor as bt
from common.protocol import Task
from pydantic import BaseModel, Field


class MinerData(BaseModel):
    uid: int

    hotkey: str | None = None
    """Miner hotkey."""

    observations: deque[int] = Field(default_factory=deque)
    """Observation window containing task finish times (seconds), limited to the last 8 hours."""

    fidelity_score: float = 1.0
    """Exponential moving average (EMA) of the fidelity score."""

    assigned_task: Task | None = None
    """The task currently assigned to the miner, if any."""

    assignment_time: float | None = None
    """Task assignment time."""

    cooldown_until: int = 0
    """Miner aren't allowed to pull tasks form this validator during the cooldown."""

    cooldown_violations: int = 0
    """Number of times a miner has failed to respect the mandatory cooling period."""

    def reset_task(self, cooldown: int | None = None) -> None:
        self.assigned_task = None
        self.assignment_time = None

        if cooldown is not None:
            self.cooldown_until = int(time.time()) + cooldown

    def assign_task(self, task: Task) -> None:
        self.assigned_task = task
        self.assignment_time = time.time()

    def add_observation(self, task_finish_time: int, fidelity_score: float, moving_average_alpha: float) -> None:
        self.observations.append(task_finish_time)
        prev_fidelity_score = self.fidelity_score
        self.fidelity_score = prev_fidelity_score * (1 - moving_average_alpha) + moving_average_alpha * fidelity_score

        bt.logging.debug(
            f"[{self.uid}] score: {prev_fidelity_score:.2f} -> {self.fidelity_score:.2f}."
            f" Observations (8h): {len(self.observations)}"
        )

    def is_task_expired(self, expiration_time: int) -> bool:
        if self.assignment_time is None:
            return False  # No task assigned, hence no expiry.
        return time.time() > self.assignment_time + expiration_time

    def is_on_cooldown(self) -> bool:
        if self.cooldown_until == 0:
            return False
        return time.time() < self.cooldown_until

    def cooldown_left(self) -> int:
        return 0 if self.cooldown_until == 0 else self.cooldown_until - int(time.time())

    def _expire_observations(self, current_time: int, observation_window: int = 8 * 60 * 60) -> None:
        expiration_threshold = current_time - observation_window
        while self.observations and self.observations[0] < expiration_threshold:
            self.observations.popleft()

    def calculate_reward(self, current_time: int, observation_window: int = 8 * 60 * 60) -> float:
        self._expire_observations(current_time, observation_window)
        return len(self.observations) * self.fidelity_score
