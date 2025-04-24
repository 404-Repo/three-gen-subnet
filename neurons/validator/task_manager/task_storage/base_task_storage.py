from abc import ABC, abstractmethod
from typing import Any

import bittensor as bt

from validator.task_manager.task import ValidatorTask


class BaseTaskStorage(ABC):
    """Abstract base class for task registries.
    Defines the interface for managing tasks in the system."""

    @abstractmethod
    def __init__(self, *, config: bt.config, wallet: bt.wallet) -> None:
        self._config = config
        self._wallet = bt.wallet(config=config) if wallet is None else wallet

    @abstractmethod
    def get_next_task(self, *args: Any, **kwargs: Any) -> ValidatorTask | None:
        """Returns the next task from the queue."""
        pass

    @abstractmethod
    def submit_result(self, *args: Any, **kwargs: Any) -> None:
        """Submits the result."""
        pass

    def has_task(self, *, task_id: str) -> bool:
        """Returns True if the task exists."""
        return False

    def has_tasks(self) -> bool:
        """Returns True if there are any tasks in the queue."""
        return False

    def fail_task(self, *, task_id: str, task_prompt: str, hotkey: str, miner_uid: int) -> None:  # noqa: B027
        """Fails the task."""
        pass
