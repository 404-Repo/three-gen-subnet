from abc import ABC, abstractmethod

import bittensor as bt


class BaseTaskStorage(ABC):
    """Abstract base class for task registries.
    Defines the interface for managing tasks in the system."""

    @abstractmethod
    def __init__(self, *, config: bt.config, wallet: bt.wallet | None) -> None:
        self._config = config
        self._wallet = bt.wallet(config=config) if wallet is None else wallet

    def has_task(self, *, task_id: str) -> bool:
        """Returns True if the task exists."""
        return False
