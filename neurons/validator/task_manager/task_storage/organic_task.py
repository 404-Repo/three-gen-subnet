import asyncio
import heapq
import time
from enum import IntEnum
from typing import NamedTuple

import bittensor as bt
import pybase64
import pyspz
from common.protocol import TextTask
from pydantic import BaseModel, ConfigDict, Field

from validator.api.protocol import MinerStatistics, TaskStatistics
from validator.task_manager.validator_task import ValidatorTask


class DuelStatus(IntEnum):
    """Duel status enumeration with priority-based ordering."""

    # WARNING: Values are priority-ordered (lower = higher priority)
    # These values are used directly in tuple comparisons for sorting.
    # Changing them will break ResultPriority and all duel sorting logic!
    WIN = 0
    DRAW = 1
    NOT_JUDGED = 2
    LOSS = 3


class ResultPriority(NamedTuple):
    """
    Result priority tuple for comparison (lower value = higher priority).

    Priority order (highest to lowest):
    1. Wins over losses (DuelStatus.WON < DuelStatus.LOST)
    2. Higher miner rating (stored as negative for natural ordering)
    3. Higher validation score (stored as negative for natural ordering)
    4. Earlier submit time (older submissions prioritized)

    Note: Rating and score are stored as negative values to achieve
    descending sort with Python's natural tuple comparison.
    """

    status: DuelStatus
    neg_rating: float
    neg_validation_score: float
    submit_time: int


class AssignedMiner(BaseModel):
    uid: int
    """Neuron UID of the miner."""
    hotkey: str
    """Neuron hotkey of the miner."""
    assign_time: int
    """When the task was assigned to the miner."""
    compressed_result: str | None = None
    """Submitted results."""
    grid_preview: str | None = None
    """Base64-encoded PNG of 2x2 grid showing multiple angles/views."""
    score: float = 0
    """Validation score of the miner results."""
    rating: float = 0
    """Miner's duel rating."""
    submit_time: int = 0
    """When task was submitted by the miner."""
    finished: bool = False
    """Status whether assigned miner is finished with the task."""
    duel_status: DuelStatus = DuelStatus.NOT_JUDGED
    """Status whether results where compared with other results using the judge service."""

    def miner_stats(self) -> MinerStatistics:
        return MinerStatistics(
            hotkey=self.hotkey,
            assign_time=self.assign_time,
            data_format="ply",
            score=self.score,
            submit_time=self.submit_time,
        )

    def decompress_results(self) -> str | None:
        if self.compressed_result is None:
            return None
        return str(
            pybase64.b64encode(
                pyspz.decompress(pybase64.b64decode(self.compressed_result), include_normals=False)
            ).decode(encoding="utf-8")
        )

    def result_priority(self) -> ResultPriority:
        """Returns the priority of the miner result (lower value - higher priority)."""
        return ResultPriority(self.duel_status, -self.rating, -self.score, self.submit_time)


class OrganicTaskJudgeQueuePriority(NamedTuple):
    """
    Task priority in the judge service queue (lower value = higher priority).

    Priority order (highest to lowest):
    1. Fewer duels performed (ensures all tasks get initial judgments)
    2. Earlier registration time (older tasks prioritized)
    """

    num_duels: int
    registration_time: float


class OrganicTask(ValidatorTask):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    assigned_miners: dict[str, AssignedMiner] = Field(default_factory=dict)
    """Miners assigned for the job."""
    create_time: float = Field(default_factory=time.time)
    """Task registration time."""
    assign_time: float = 0.0
    """First assignment time."""
    strong_miner_assigned: bool = False
    """True if the strong miner is amongst assigned miners."""
    first_result_time: float = 0.0
    """Time of the first acceptable results."""
    best_result: AssignedMiner | None = None
    """Current best result."""
    results_to_judge: list[tuple[ResultPriority, AssignedMiner]] = Field(default_factory=list, exclude=True)
    """Results pending judgment. Prioritized by `result_priority()`.
    Once judged, result might be enqueued again to compare with another result."""
    num_results_judged: int = 0
    """Number of results judged. Used to prioritize task in the judge queue."""
    num_results_being_judged: int = 0
    """Counter for the results queued for judgement."""
    finalized: bool = False
    """
    True when task processing is complete and result has been sent to gateway.
    Occurs when either: 
    - Best possible result is obtained
    - 30-second timeout reached (sends best available result)
    """

    def get_stats(self) -> TaskStatistics:
        return TaskStatistics(
            create_time=int(self.create_time), miners=[miner.miner_stats() for miner in self.assigned_miners.values()]
        )

    def judge_queue_priority(self) -> OrganicTaskJudgeQueuePriority:
        """Returns the priority of the task in the judge service queue (lower value - higher priority)."""
        return OrganicTaskJudgeQueuePriority(self.num_results_judged, self.create_time)

    def should_be_assigned(self, strong_miner: bool, copies: int) -> bool:
        """Return True if the task should be further assigned to miners.
        It occurs when:
        - the number of assigned miners is less than the number of required copies;
        - the strong miner can take the task in this case if all previous miners are not strong.
        If strong miner was assigned then no new miners will be assigned.
        """
        return (strong_miner and not self.strong_miner_assigned) or len(self.assigned_miners) < copies

    def all_miners_finished(self, copies: int) -> bool:
        """Return True if all assigned miners finished the task and no new miners should be assigned.
        If no strong miners were assigned to the task it returns False.
        """
        return all(miner.finished for miner in self.assigned_miners.values()) and not self.should_be_assigned(
            strong_miner=True, copies=copies
        )

    def all_work_done(self, copies: int) -> bool:
        """Return True if strong miner was assigned, all miners finished and all results are judged."""
        return (
            self.num_results_being_judged == 0 and len(self.results_to_judge) < 2 and self.all_miners_finished(copies)
        )

    def is_duplicate_result(self, miner: AssignedMiner) -> bool:
        """Check if miner's result matches any other miner's submission."""
        for other in self.assigned_miners.values():
            if other.uid != miner.uid and other.grid_preview == miner.grid_preview:
                bt.logging.debug(f"[{miner.uid}] and [{other.uid}] sent identical for organic task ({self.log_id}).")
                return True
        return False

    def queue_for_judgment(self, miner: AssignedMiner) -> None:
        """Add miner to judgment queue"""
        heapq.heappush(self.results_to_judge, (miner.result_priority(), miner))

    def update_best(self, miner: AssignedMiner) -> None:
        """Update best result if this has higher priority."""
        if self.best_result is None:
            self.best_result = miner
            return

        if miner.result_priority() < self.best_result.result_priority():
            self.best_result = miner

    def finalize(self) -> AssignedMiner | None:
        """Finalize the task and return the best result."""
        self.finalized = True
        return self.best_result


class GatewayOrganicTask(OrganicTask):
    """Class that represents a task to generate 3D assets from a gateway."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gateway_url: str
    result_future: asyncio.Future[AssignedMiner | None] | None = None
    """Future is set when the task result is ready to be sent to the gateway."""


class LegacyOrganicTask(OrganicTask):
    """Class that represents a task to generate 3D assets from a prompt that came from the
    legacy public API."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start_future: asyncio.Future[AssignedMiner | None] = Field(default_factory=asyncio.get_event_loop().create_future)
    """Future is set when the task gets assigned the first time."""
    first_result_future: asyncio.Future[AssignedMiner | None] = Field(
        default_factory=asyncio.get_event_loop().create_future
    )
    """Future is set when the first acceptable results are generated."""
    best_result_future: asyncio.Future[AssignedMiner | None] = Field(
        default_factory=asyncio.get_event_loop().create_future
    )
    """Future is set when all assigned miners submit their results."""

    @classmethod
    def create_task(cls, *, id: str, prompt: str) -> "LegacyOrganicTask":
        task = cls(
            protocol=TextTask(id=id, prompt=prompt),
            start_future=asyncio.get_event_loop().create_future(),
            first_result_future=asyncio.get_event_loop().create_future(),
            best_result_future=asyncio.get_event_loop().create_future(),
        )
        return task
