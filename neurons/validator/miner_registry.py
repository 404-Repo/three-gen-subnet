from pydantic import BaseModel


class MinerInfo(BaseModel):
    uid: int
    score: float
    total_tasks: int
    tasks_24h: list[int]  # length 24


class MinerRegistry:
    unknown_miners: list[int]
    active_miners: list[int]
