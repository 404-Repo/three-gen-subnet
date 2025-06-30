import time
from collections import deque

import bittensor as bt
from common.protocol import Task
from pydantic import BaseModel, Field


class MinerData(BaseModel):
    """
    Data about a miner performing tasks for this validator. It contains miner's keys in the network,
    times of task finish, miner's fidelity score and assigned task. It also contains information about
    cooldown that controls how often the miner can pull tasks from this validator.
    """

    uid: int

    hotkey: str | None = None
    """Miner hotkey."""

    observations: deque[int] = Field(default_factory=deque)
    """Observation window containing task finish times (seconds), limited to the last 4 hours."""

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

    validation_locked_until: int = 0
    """Miner is locked for validation until this timestamp."""

    last_submit_time: int = 0
    """Timestamp of the last successful miner submit."""

    def reset_task(self, throttle_period: int, cooldown: int) -> None:
        """
        Reset the task and cooldown.
        Sets cooldown_until that is a timestamp of when the miner can pull tasks again.
        """
        if self.assignment_time is None:
            self.cooldown_until = int(time.time()) + cooldown
        else:
            self.cooldown_until = int(max(time.time() + cooldown - throttle_period, self.assignment_time + cooldown))

        self.assigned_task = None
        self.assignment_time = None

    def assign_task(self, task: Task) -> None:
        """
        Assign a task to the miner.
        Sets time when the task was assigned.
        """
        self.assigned_task = task
        self.assignment_time = time.time()

    def add_observation(self, task_finish_time: int, fidelity_score: float, moving_average_alpha: float) -> None:
        """
        Add a task finish time to the miner's observations.
        Updates the miner's fidelity score using formula (1-alpha)*prev_score + alpha*new_score.
        """
        self.observations.append(task_finish_time)
        prev_fidelity_score = self.fidelity_score
        self.fidelity_score = prev_fidelity_score * (1 - moving_average_alpha) + moving_average_alpha * fidelity_score

        bt.logging.debug(
            f"[{self.uid}] score: {fidelity_score}. Avg score: {prev_fidelity_score:.2f} -> {self.fidelity_score:.2f}."
            f" Observations (4h): {len(self.observations)}"
        )

    def is_on_cooldown(self) -> bool:
        """
        Check if the miner is on cooldown. It means that the miner is not allowed to pull tasks from this validator.
        """
        if self.cooldown_until == 0:
            return False
        return time.time() < self.cooldown_until

    def cooldown_left(self) -> int:
        """
        Return the remaining time when miner can't pull tasks from this validator.
        """
        return 0 if self.cooldown_until == 0 else self.cooldown_until - int(time.time())

    def _expire_observations(self, current_time: int, observation_window: int = 4 * 60 * 60) -> None:
        """
        Remove old observations from the miner's observations.
        Observations that are older than observation_window are removed.
        """
        expiration_threshold = current_time - observation_window
        while self.observations and self.observations[0] < expiration_threshold:
            self.observations.popleft()

    def calculate_reward(self, current_time: int, observation_window: int = 4 * 60 * 60) -> float:
        """
        Calculate the reward for the miner.
        It is calculated by multiplying fidelity score by the number of not-outdated observations.
        """
        self._expire_observations(current_time, observation_window)
        return len(self.observations) * self.fidelity_score
