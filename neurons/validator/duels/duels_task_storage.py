import asyncio
import hashlib
import json
import time
import uuid
from collections import deque
from typing import Any

import aiohttp
import bittensor as bt
import pybase64
from common.protocol import SubmitResults

from validator.config import config
from validator.duels.data_structures import Duel, MinerInDuel, MinerResults
from validator.duels.judge_service import JudgeResponse, JudgeService
from validator.duels.ratings import DuelRatings, Rating, duel_ratings, update_ratings
from validator.task_manager.task import DuelTask
from validator.task_manager.task_storage.base_task_storage import BaseTaskStorage
from validator.task_manager.task_storage.synthetic_task_storage import SyntheticTaskStorage, synthetic_task_storage
from validator.validation_service import ValidationResponse, ValidationService, validation_service


NEURONS_LIMIT = 256
PENDING_DUELS_SIZE = NEURONS_LIMIT * 4
GARBAGE_COLLECTION_CYCLE = 60 * 60  # 1 hour


class DuelsTaskStorage(BaseTaskStorage):
    """Task storage implementation for managing duels between miners."""

    def __init__(
        self,
        *,
        config: bt.config,
        wallet: bt.wallet | None,
        validation_service: ValidationService,
        synthetic_task_storage: SyntheticTaskStorage,
        ratings: DuelRatings,
    ) -> None:
        super().__init__(config=config, wallet=wallet)

        self._synthetic_task_storage = synthetic_task_storage
        """Reference to the synthetic task storage. Used to get random prompts for the duels."""

        self._validation_service = validation_service
        """Reference to the validation service. Used to request rendering results view for the judge."""

        self._ratings = ratings
        """Reference to the ratings structure that manages all miner ratings."""

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
        
        Duel is removed when both miners pull a task.
        """

        self._pending_duels_by_miner: dict[int, deque[Duel]] = {}
        """Same pending duels but grouped by miner uid.
        
        Duel is removed when miner pulls a task.
        """

        self._active_duels: dict[str, Duel] = {}
        """Duels started by at least one miner (task id -> duel)."""

        self._pending_judgment_queue: asyncio.Queue[Duel] = asyncio.Queue()
        """Duels with the results from both miners, awaiting to be judged."""

        self._process_results_queue: asyncio.Queue[tuple[Duel, JudgeResponse]] = asyncio.Queue()
        """Duels evaluated by the judge, awaiting to be processed."""

        self._process_results_job: asyncio.Task | None = None
        """Async task that processes results from the `_process_results_queue`"""

        self._garbage_collecting_job: asyncio.Task | None = None
        """Periodic job to check the active duels and recycle the expired ones."""

        self._judge_service: JudgeService = JudgeService(
            config=config,
            pending_judgment_queue=self._pending_judgment_queue,
            process_results_queue=self._process_results_queue,
            validation_service=validation_service,
        )
        """Wrapper that communicates with the vllm server."""

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
            self._pending_duels.pop(duel.timestamp_nonce, None)
            if duel in self._pending_duels_by_miner[duel.left.uid]:
                self._pending_duels_by_miner[duel.left.uid].remove(duel)
            if duel in self._pending_duels_by_miner[duel.right.uid]:
                self._pending_duels_by_miner[duel.right.uid].remove(duel)

    def get_next_task(self, *, miner_uid: int) -> DuelTask | None:
        if self._config.duels.disabled:
            return None

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
            f"+ {self._pending_judgment_queue.qsize()}"
        )

        return duel.task

    def _fill_pending_duels(self, current_time: float) -> None:
        total_duels = len(self._pending_duels) + self._pending_judgment_queue.qsize()
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
            rating=self._ratings.get_miner_rating(miner_uid),
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
            rating=self._ratings.get_miner_rating(miner_uid),
        )
        await self._submit_results(miner_uid=miner_uid, duel=duel, results=results)

    async def _submit_results(self, *, miner_uid: int, duel: Duel, results: MinerResults) -> None:
        if duel.left.uid == miner_uid:
            miner = duel.left
            opponent = duel.right
            miner_position = 1
        else:
            miner = duel.right
            opponent = duel.left
            miner_position = 2

        miner.results = results

        if opponent.results is None:
            # Waiting for another miner
            return

        self._active_duels.pop(duel.task.id, None)

        if duel.failed_by_validator:
            bt.logging.warning(f"[{miner.uid}] vs [{opponent.uid}] duel failed (validator fault)")
            return

        if miner.results.view is None and opponent.results.view is None:
            worst = 0
        elif miner.results.view is None:
            worst = miner_position
        elif opponent.results.view is None:
            worst = 3 - miner_position
        else:
            worst = None

        if worst is not None:
            await self._process_duel_results(
                duel, judgement=JudgeResponse(worst=worst, issues="One or both miners failed to generate")
            )
            return

        await self._pending_judgment_queue.put(duel)

    def has_task(self, *, task_id: str) -> bool:
        return task_id in self._active_duels

    async def start_judging_duels(self) -> None:
        await self._judge_service.start_judging_duels()
        self._process_results_job = asyncio.create_task(self._process_next_duel_results())

    def stop_judging_duels(self) -> None:
        self._judge_service.stop_judging_duels()
        if self._process_results_job is not None:
            self._process_results_job.cancel()

    async def _process_next_duel_results(self) -> None:
        while True:
            try:
                duel, judgement = await asyncio.wait_for(self._process_results_queue.get(), timeout=1.0)

                await self._process_duel_results(duel, judgement)

                self._process_results_queue.task_done()
            except TimeoutError:
                await asyncio.sleep(1)
                continue
            except Exception as e:
                bt.logging.error(f"Processing duel results failed with: {e}")
                self._process_results_queue.task_done()

    async def _process_duel_results(self, duel: Duel, judgement: JudgeResponse) -> None:
        if duel.left.results is None or duel.right.results is None:  # pragma: no cover
            bt.logging.error(
                f"[{duel.left.uid}] vs [{duel.right.uid}] duel undefined behaviour, "
                f"no results when results are expected"
            )
            return

        def _fill_duel_results(
            results: MinerResults, rating_before: Rating, rating_after: Rating
        ) -> dict[str, str | float]:
            return {
                "hotkey": results.hotkey,
                "coldkey": results.coldkey,
                "glicko_before": rating_before.glicko.rating,
                "glicko_after": rating_after.glicko.rating,
                "glicko_rd": rating_after.glicko.rd,
                "glicko_vol": rating_after.glicko.vol,
            }

        if judgement.worst == 1:
            winner = 2
        elif judgement.worst == 2:
            winner = 1
        else:
            winner = 0

        left_r_before = self._ratings.get_miner_rating(duel.left.uid).model_copy(deep=True)
        right_r_before = self._ratings.get_miner_rating(duel.right.uid).model_copy(deep=True)
        update_ratings(duel.left.results.rating, duel.right.results.rating, winner=winner)

        bt.logging.debug(
            f"[{duel.left.uid}] rating: {left_r_before.glicko.rating:.1f} "
            f"-> {duel.left.results.rating.glicko.rating:.1f} "
            f"({duel.task.prompt[:100]})"
        )
        bt.logging.debug(
            f"[{duel.right.uid}] rating: {right_r_before.glicko.rating:.1f} "
            f"-> {duel.right.results.rating.glicko.rating:.1f} "
            f"({duel.task.prompt[:100]})"
        )

        validator_hotkey = self._wallet.hotkey.ss58_address
        message_to_sign = f"duel_saver{duel.timestamp_nonce}{validator_hotkey}"
        signature = pybase64.b64encode(self._wallet.hotkey.sign(message_to_sign)).decode(encoding="utf-8")
        duel_results = {
            "validator_hotkey": validator_hotkey,
            "signature": signature,
            "finish_time": time.time(),
            "timestamp_nonce": duel.timestamp_nonce,
            "prompt": duel.task.prompt,
            "winner": winner,
            "explanation": judgement.issues,
            "left": _fill_duel_results(duel.left.results, left_r_before, duel.left.results.rating),
            "right": _fill_duel_results(duel.right.results, right_r_before, duel.right.results.rating),
        }
        asyncio.create_task(
            self._publish_results(
                left_uid=duel.left.uid,
                right_uid=duel.right.uid,
                results=duel_results,
                preview1=duel.left.results.view,
                preview2=duel.right.results.view,
            )
        )

    async def _publish_results(
        self, left_uid: int, right_uid: int, results: dict[str, Any], preview1: bytes | None, preview2: bytes | None
    ) -> None:
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
                    response_data = await response.json()
                    duel_id = response_data.get("duel_id", "unknown")
                bt.logging.debug(
                    f"[{left_uid}] vs [{right_uid}] duel published. Duel ID: {duel_id} " f"({results['prompt'][:100]})"
                )
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
    ratings=duel_ratings,
)
