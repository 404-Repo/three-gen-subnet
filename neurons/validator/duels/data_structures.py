from pydantic import BaseModel

from validator.duels.ratings import Rating
from validator.task_manager.task import DuelTask


class MinerResults(BaseModel):
    """Results and metadata for a miner participating in a duel."""

    view: bytes | None
    """Rendered duel view or None if miner failed the task."""
    score: float
    """Validation score."""
    hotkey: str
    """Miner hotkey (filled when results are received, to handle the case of the ownership change)."""
    coldkey: str
    """Miner coldkey (filled when results are received, to handle the case of the ownership change)."""
    rating: Rating
    """Miner rating. Reference, so will have up-to-date information, even if miner ownership changes."""


class MinerInDuel(BaseModel):
    """Represents a miner's participation state and results in a duel."""

    uid: int
    """Miner uid."""
    started: bool = False
    """Whether miner pulled the task or not."""
    results: MinerResults | None = None
    """Miner's results after completing the duel task."""


class Duel(BaseModel):
    """Represents a duel between two miners with a specific task."""

    timestamp_nonce: int
    """Multipurpose field:
    - reflects the creation time, although it might be some seconds bigger,
    - unique among duels and used to identify the duel in grafana,
    - nonce for the hash used to select miners.
    """
    task: DuelTask
    """Task to give to miners."""
    left: MinerInDuel
    """First participant in the duel."""
    right: MinerInDuel
    """Second participant in the duel."""
    last_miner_pull_time: float = 0.0
    """Time the duel was pulled by the last miner."""
    failed_by_validator: bool = False
    """True, if duel failed and it's not a miner's fault."""
