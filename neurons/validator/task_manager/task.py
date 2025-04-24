import asyncio
import time
import uuid

import pybase64
import pyspz
from pydantic import BaseModel, Field

from validator.api.protocol import MinerStatistics, TaskStatistics


class AssignedMiner(BaseModel):
    hotkey: str
    """Neuron hotkey of the miner."""
    assign_time: int
    """When the task was assigned to the miner."""
    compressed_result: str | None = None
    """Submitted results."""
    score: float = 0
    """Validation score of the miner results."""
    submit_time: int = 0
    """When task was submitted by the miner."""
    finished: bool = False
    """Status whether assigned miner is finished with the task."""

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


class ValidatorTask(BaseModel):
    id: str = str(uuid.uuid4())
    """Unique id."""
    prompt: str
    """Prompt to use for generation."""


class SyntheticTask(ValidatorTask):
    pass


class OrganicTask(ValidatorTask):
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

    def get_stats(self) -> TaskStatistics:
        return TaskStatistics(
            create_time=int(self.create_time), miners=[miner.miner_stats() for miner in self.assigned_miners.values()]
        )

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

    def get_best_result(self) -> AssignedMiner | None:
        """Returns miner with the highest validation score."""
        best: AssignedMiner | None = None
        for miner in self.assigned_miners.values():
            if miner.compressed_result is None:
                continue
            if best is None:
                best = miner
                continue
            if miner.score > best.score:
                best = miner
        return best


class GatewayOrganicTask(OrganicTask):
    """Class that represents a task to generate 3D assets from a gateway.."""

    gateway_url: str
    result_future: asyncio.Future[AssignedMiner | None] | None = None
    """Future is set when the task result is ready to be sent to the gateway."""

    @classmethod
    def create_task(cls, *, id: str, prompt: str, gateway_url: str) -> "GatewayOrganicTask":
        task = cls(
            id=id,
            prompt=prompt,
            gateway_url=gateway_url,
        )
        return task

    class Config:
        arbitrary_types_allowed = True


class LegacyOrganicTask(OrganicTask):
    """Class that represents a task to generate 3D assets from a prompt that came from the
    legacy public API."""

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

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create_task(cls, *, id: str, prompt: str) -> "LegacyOrganicTask":
        task = cls(
            id=id,
            prompt=prompt,
            start_future=asyncio.get_event_loop().create_future(),
            first_result_future=asyncio.get_event_loop().create_future(),
            best_result_future=asyncio.get_event_loop().create_future(),
        )
        return task


def set_future(future: asyncio.Future[AssignedMiner | None] | None, miner: AssignedMiner | None) -> None:
    """Forcely sets result to the future."""

    def do_set(f: asyncio.Future[AssignedMiner | None], results: AssignedMiner | None) -> None:
        if not f.done() and not f.cancelled():
            f.set_result(results)

    if future is not None:
        future.get_loop().call_soon_threadsafe(do_set, future, miner)
