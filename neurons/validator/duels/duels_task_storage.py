import asyncio
import hashlib
import json
import time
import uuid
from collections import deque
from typing import Any

import aiohttp
import bittensor as bt
from common.protocol import SubmitResults
from pydantic import BaseModel

from validator.config import config
from validator.duels.ranks import DuelRanks, Rank, duel_ranks, update_ranks
from validator.task_manager.task import DuelTask
from validator.task_manager.task_storage.base_task_storage import BaseTaskStorage
from validator.task_manager.task_storage.synthetic_task_storage import SyntheticTaskStorage, synthetic_task_storage
from validator.validation_service import ValidationResponse, ValidationService, validation_service


NEURONS_LIMIT = 256
PENDING_DUELS_SIZE = NEURONS_LIMIT * 4
GARBAGE_COLLECTION_CYCLE = 60 * 60  # 1 hour


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
    rank: Rank
    """Miner rank. Reference, so will have up-to-date information, even if miner ownership changes."""


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


class JudgeResponse(BaseModel):
    """Response from a judge evaluating a duel between two miners."""

    winner: int
    """Judge's decision: 1, 2 or 0 (draw)"""
    explanation: str
    """Explanation of the scoring decision."""


class DuelsTaskStorage(BaseTaskStorage):
    """Task storage implementation for managing duels between miners."""

    def __init__(
        self,
        *,
        config: bt.config,
        wallet: bt.wallet | None,
        validation_service: ValidationService,
        synthetic_task_storage: SyntheticTaskStorage,
        ranks: DuelRanks,
    ) -> None:
        super().__init__(config=config, wallet=wallet)

        self._synthetic_task_storage = synthetic_task_storage
        self._validation_service = validation_service
        self._ranks = ranks

        self._duels_start: float = time.time() + self._config.duels.start_delay
        """Time to start duels, necessary delay to fill `_last_pull_time`."""

        self._last_timestamp_nonce: int = 0
        """See `Duel.timestamp_nonce` and `_fill_pending_duels`."""

        self._last_pull_time: dict[int, float] = {}
        """Last time each miner called a `get_next_task`."""

        self._pending_duels: dict[int, Duel] = {}
        """Pre-filled duels waiting to be pulled.

        The idea behind the `_pending_duels` is we want to have a transparent and verifiable miner selection
        for the duel.
        Duel results are shared with the prompt given and nonce used. Miner selection could be verified.
        """

        self._pending_duels_by_miner: dict[int, deque[Duel]] = {}
        """Same pending duels but grouped by miner uid."""

        self._active_duels: dict[str, Duel] = {}
        """Duels started by at least one miner (task id -> duel)."""

        self._pending_judgement: asyncio.Queue[Duel] = asyncio.Queue()
        """Duels with the results from both miners, awaiting to be judged."""

        self._duels_judged: int = 0
        """Total number of duels judged during a validator runtime."""

        self._duels_per_second_limit: float = min(1.0, self._config.duels.duels_per_minute / 60.0)
        """Rate limit (enforced when using pay-per-request billing)."""

        self._garbage_collecting_job: asyncio.Task | None = None
        """Periodic job to check the active duels and recycle the expired ones."""

        self._judge_workers: list[asyncio.Task] = []
        """Worker tasks for judging completed duels."""

    async def start_garbage_collection_cron(self) -> None:
        self._garbage_collecting_job = asyncio.create_task(self._garbage_collection_cron())

    async def _garbage_collection_cron(self) -> None:
        while True:
            await asyncio.sleep(GARBAGE_COLLECTION_CYCLE)
            await self._collect_garbage()

    async def _collect_garbage(self) -> None:
        current_time = time.time()
        expired_duels = [
            duel
            for duel in self._active_duels.values()
            if duel.last_miner_pull_time + GARBAGE_COLLECTION_CYCLE < current_time
        ]
        for duel in expired_duels:
            bt.logging.debug(f"[{duel.left.uid}] vs [{duel.right.uid}] duel was recycled")
            self._active_duels.pop(duel.task.id, None)

    def get_next_task(self, *, miner_uid: int) -> DuelTask | None:
        current_time = time.time()
        self._last_pull_time[miner_uid] = current_time

        if current_time < self._duels_start:
            return None

        self._fill_pending_duels(current_time)

        miner_duels = self._pending_duels_by_miner.get(miner_uid, None)

        if not miner_duels:
            return None

        duel = miner_duels.popleft()
        miner = duel.left if duel.left.uid == miner_uid else duel.right
        miner.started = True

        self._active_duels[duel.task.id] = duel
        duel.last_miner_pull_time = current_time

        if duel.left.started and duel.right.started:
            self._pending_duels.pop(duel.timestamp_nonce)

        bt.logging.debug(
            f"[{miner_uid}] received a duel: [{duel.left.uid}] vs [{duel.right.uid}] ({duel.task.prompt[:100]}). "
            f"Stats: {len(self._pending_duels)} ({len(miner_duels)}) + {len(self._active_duels)} "
            f"+ {self._pending_judgement.qsize()}"
        )

        return duel.task

    def _fill_pending_duels(self, current_time: float) -> None:
        total_duels = len(self._pending_duels) + self._pending_judgement.qsize()
        if total_duels >= PENDING_DUELS_SIZE:
            return

        inactivity_threshold = current_time - self._config.duels.inactivity_time
        attempts = max(3, PENDING_DUELS_SIZE - total_duels)
        for _ in range(attempts):
            timestamp_nonce = max(int(current_time), self._last_timestamp_nonce + 1)
            self._last_timestamp_nonce = timestamp_nonce

            prompt = self._synthetic_task_storage.get_random_prompt()
            hash_bytes = hashlib.sha256(f"{timestamp_nonce}:{prompt}".encode()).digest()
            left_uid = int.from_bytes(hash_bytes[0:4], "big") % NEURONS_LIMIT
            right_uid = int.from_bytes(hash_bytes[4:8], "big") % NEURONS_LIMIT

            if left_uid == right_uid:
                continue
            if self._last_pull_time.get(left_uid, 0) < inactivity_threshold:
                continue
            if self._last_pull_time.get(right_uid, 0) < inactivity_threshold:
                continue

            duel = Duel(
                timestamp_nonce=timestamp_nonce,
                left=MinerInDuel(
                    uid=left_uid,
                ),
                right=MinerInDuel(
                    uid=right_uid,
                ),
                task=DuelTask(id=str(uuid.uuid4()), prompt=prompt),
            )

            self._pending_duels[duel.timestamp_nonce] = duel
            self._pending_duels_by_miner.setdefault(left_uid, deque()).append(duel)
            self._pending_duels_by_miner.setdefault(right_uid, deque()).append(duel)

    async def submit_result(
        self, synapse: SubmitResults, validation_res: ValidationResponse, miner_uid: int, metagraph: bt.metagraph
    ) -> None:
        duel = self._active_duels.get(synapse.task.id, None)
        if duel is None:  # pragma: no cover
            bt.logging.warning(
                f"[{miner_uid}] submitted duel results but duel was not found ({synapse.task.prompt[:100]})"
            )
            return

        bt.logging.debug(
            f"[{miner_uid}] submitted a duel: [{duel.left.uid}] vs [{duel.right.uid}] ({synapse.task.prompt[:100]})"
        )

        axon = metagraph.axons[miner_uid]
        results = MinerResults(
            view=None,
            score=validation_res.score,
            coldkey=axon.coldkey,
            hotkey=axon.hotkey,
            rank=self._ranks.get_miner_rank(miner_uid),
        )
        await self._render_and_submit(
            synapse=synapse,
            miner_uid=miner_uid,
            duel=duel,
            results=results,
        )

    async def _render_and_submit(
        self,
        synapse: SubmitResults,
        miner_uid: int,
        duel: Duel,
        results: MinerResults,
    ) -> None:
        results.view = await self._validation_service.render_duel_view(synapse=synapse, neuron_uid=miner_uid)
        if results.view is None:
            duel.failed_by_validator = True
        await self._submit_results(miner_uid=miner_uid, duel=duel, results=results)

    async def fail_task(self, *, task_id: str, task_prompt: str, miner_uid: int, metagraph: bt.metagraph) -> None:
        duel = self._active_duels.get(task_id, None)
        if duel is None:  # pragma: no cover
            bt.logging.warning(f"[{miner_uid}] failed a duel but duel was not found ({task_prompt[:100]})")
            return

        bt.logging.debug(f"[{miner_uid}] failed a duel: [{duel.left.uid}] vs [{duel.right.uid}] ({task_prompt[:100]})")

        axon = metagraph.axons[miner_uid]
        results = MinerResults(
            view=None,
            score=0.0,
            coldkey=axon.coldkey,
            hotkey=axon.hotkey,
            rank=self._ranks.get_miner_rank(miner_uid),
        )
        await self._submit_results(miner_uid=miner_uid, duel=duel, results=results)

    async def _submit_results(self, *, miner_uid: int, duel: Duel, results: MinerResults) -> None:
        if duel.left.uid == miner_uid:
            miner = duel.left
            opponent = duel.right
            miner_position = 1
            opponent_position = 2
        else:
            miner = duel.right
            opponent = duel.left
            miner_position = 2
            opponent_position = 1

        miner.results = results

        if opponent.results is None:
            # Waiting for another miner
            return

        self._active_duels.pop(duel.task.id, None)

        if duel.failed_by_validator:
            bt.logging.warning(f"[{miner.uid}] vs [{opponent.uid}] duel failed (validator fault)")
            return

        if miner.results.view is None and opponent.results.view is None:
            winner = 0
        elif miner.results.view is None:
            winner = opponent_position
        elif opponent.results.view is None:
            winner = miner_position
        else:
            winner = None

        if winner is not None:
            await self._process_duel_results(
                duel, judgement=JudgeResponse(winner=winner, explanation="One or both miners failed to generate")
            )
            return

        await self._pending_judgement.put(duel)

    async def _process_duel_results(self, duel: Duel, judgement: JudgeResponse) -> None:
        if duel.left.results is None or duel.right.results is None:  # pragma: no cover
            bt.logging.error(
                f"[{duel.left.uid}] vs [{duel.right.uid}] duel undefined behaviour, "
                f"no results when results are expected"
            )
            return

        def _fill_duel_results(results: MinerResults, rank_before: Rank, rank_after: Rank) -> dict[str, str | float]:
            return {
                "hotkey": results.hotkey,
                "coldkey": results.coldkey,
                "elo_before": rank_before.elo.rank,
                "elo_after": rank_after.elo.rank,
                "glicko_before": rank_before.glicko.rank,
                "glicko_after": rank_after.glicko.rank,
                "trueskill_before": rank_before.trueskill.mu,
                "trueskill_after": rank_after.trueskill.mu,
            }

        left_rank_before = self._ranks.get_miner_rank(duel.left.uid).model_copy(deep=True)
        right_rank_before = self._ranks.get_miner_rank(duel.right.uid).model_copy(deep=True)
        update_ranks(duel.left.results.rank, duel.right.results.rank, winner=judgement.winner)

        bt.logging.debug(
            f"[{duel.left.uid}] vs [{duel.right.uid}] duel processed. "
            f"Winner: {judgement.winner} ({duel.task.prompt[:100]})"
        )

        duel_results = {
            "timestamp_nonce": duel.timestamp_nonce,
            "prompt": duel.task.prompt,
            "winner": judgement.winner,
            "explanation": judgement.explanation,
            "left": _fill_duel_results(duel.left.results, left_rank_before, duel.left.results.rank),
            "right": _fill_duel_results(duel.right.results, right_rank_before, duel.right.results.rank),
        }
        asyncio.create_task(
            self._publish_results(
                results=duel_results, preview1=duel.left.results.view, preview2=duel.right.results.view
            )
        )

    def has_task(self, *, task_id: str) -> bool:
        return task_id in self._active_duels

    async def start_judging_duels(self) -> None:
        for i in range(self._config.duels.judge_workers):
            self._judge_workers.append(asyncio.create_task(self._judge_next_duel(i)))

    async def _judge_next_duel(self, worker_id: int) -> None:
        while True:
            try:
                since_start = max(1.0, time.time() - self._duels_start)
                current_judge_rate: float = self._duels_judged / since_start
                if current_judge_rate > self._duels_per_second_limit:
                    delay = self._duels_judged / self._duels_per_second_limit - since_start
                    bt.logging.debug(
                        f"Judge worker {worker_id}: {current_judge_rate} duels/second. Delay: {delay} seconds"
                    )
                    await asyncio.sleep(delay)

                duel = await asyncio.wait_for(self._pending_judgement.get(), timeout=1.0)

                judgement = await self._request_duel(duel)
                self._duels_judged += 1

                if judgement is None:
                    bt.logging.warning(f"[{duel.left.uid}] vs [{duel.right.uid}] duel failed (validator fault)")
                else:
                    await self._process_duel_results(duel=duel, judgement=judgement)

                self._pending_judgement.task_done()
            except TimeoutError:
                continue
            except Exception as e:
                bt.logging.error(f"Judge worker {worker_id} failed with: {e}")
                self._pending_judgement.task_done()

    async def _request_duel(self, duel: Duel) -> JudgeResponse | None:
        if duel.left.results is None or duel.right.results is None:  # pragma: no cover
            bt.logging.error(
                f"[{duel.left.uid}] and [{duel.right.uid}] duel undefined behaviour, "
                f"no results when results are expected"
            )
            return None

        async with aiohttp.ClientSession() as session:
            try:
                start_time = time.time()
                endpoint = self._config.duels.judge_endpoint
                form_data = aiohttp.FormData()
                form_data.add_field("prompt", duel.task.prompt)
                form_data.add_field(
                    "preview1", duel.left.results.view, filename="preview1.png", content_type="image/png"
                )
                form_data.add_field(
                    "preview2", duel.right.results.view, filename="preview2.png", content_type="image/png"
                )
                async with session.post(
                    endpoint,
                    data=form_data,
                ) as response:
                    if response.status == 200:
                        judgement = JudgeResponse.model_validate(await response.json())
                        bt.logging.debug(
                            f"[{duel.left.uid}] and [{duel.right.uid}] duel evaluation finished "
                            f"in {time.time() - start_time:.4f} seconds. "
                            f"Result: {judgement.winner}. Prompt: {duel.task.prompt[:100]}"
                        )
                        return judgement
                    else:
                        bt.logging.error(
                            f"Duel evaluation failed: "
                            f"[{response.status}] {response.reason}. Prompt: {duel.task.prompt[:100]}"
                        )
            except aiohttp.ClientConnectorError:
                bt.logging.error(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {endpoint}.")
            except TimeoutError:
                bt.logging.error(f"The request to the endpoint timed out: {endpoint}")
            except aiohttp.ClientError as e:
                bt.logging.error(f"An unexpected client error occurred: {e} ({endpoint})")
            except Exception as e:
                bt.logging.error(f"An unexpected error occurred: {e} ({endpoint})")

    async def _publish_results(self, results: dict[str, Any], preview1: bytes | None, preview2: bytes | None) -> None:
        async with aiohttp.ClientSession() as session:
            try:
                endpoint = self._config.duels.duel_saver_endpoint
                form_data = aiohttp.FormData()
                form_data.add_field("results", json.dumps(results))
                form_data.add_field("preview1", preview1 or b"", filename="preview1.png", content_type="image/png")
                form_data.add_field("preview2", preview2 or b"", filename="preview2.png", content_type="image/png")
                async with session.post(
                    endpoint,
                    data=form_data,
                ) as response:
                    response.raise_for_status()
            except aiohttp.ClientConnectorError:
                bt.logging.error(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {endpoint}.")
            except TimeoutError:
                bt.logging.error(f"The request to the endpoint timed out: {endpoint}")
            except aiohttp.ClientError as e:
                bt.logging.error(f"An unexpected client error occurred: {e} ({endpoint})")
            except Exception as e:
                bt.logging.error(f"An unexpected error occurred: {e} ({endpoint})")


duel_task_storage = DuelsTaskStorage(
    config=config,
    wallet=None,
    synthetic_task_storage=synthetic_task_storage,
    validation_service=validation_service,
    ranks=duel_ranks,
)
