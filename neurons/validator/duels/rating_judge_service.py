import asyncio
import time

import bittensor as bt

from validator.duels.base_judge_service import BaseJudgeService, JudgeResponse
from validator.duels.data_structures import Duel
from validator.validation_service import ValidationService


class RatingJudgeService(BaseJudgeService):
    def __init__(
        self,
        *,
        config: bt.config,
        pending_judgment_queue: asyncio.Queue[Duel],
        process_results_queue: asyncio.Queue[tuple[Duel, JudgeResponse]],
        validation_service: ValidationService,
    ) -> None:
        super().__init__(
            judge_endpoint=config.duels.judge_endpoint,
            judge_api_key=config.duels.judge_api_key,
            num_judge_workers=config.duels.judge_workers,
        )

        self._config = config
        self._pending_judgment_queue: asyncio.Queue[Duel] = pending_judgment_queue
        self._process_results_queue: asyncio.Queue[tuple[Duel, JudgeResponse]] = process_results_queue

        self._target_duel_time: float = max(1.0, 60 / self._config.duels.duels_per_minute)
        """
        Target average time spent per duel, in seconds (inference + delay).

        Used to:
        - Control costs in pay-per-duel setups.
        - Limit duel throughput to reduce load on a shared GPU.
        """

        self._average_duel_time: float = self._target_duel_time
        # Overriding the initial average duel time value.

        self._current_duel_delay: float = 0.0
        """
        Current enforced delay between duels, in seconds, to maintain
        the target average duel time.
        """

        self._validation_service = validation_service
        """Reference to the validation service. Used to get the peak validation time."""

    async def _process_duels_loop(self, worker_id: int) -> None:
        """Take the next duel from the queue and send it to the judge service."""

        while True:
            try:
                duel_start_time = time.time()

                await self._delay_next_duel(worker_id)

                duel = await asyncio.wait_for(self._pending_judgment_queue.get(), timeout=1.0)

                judgement = await self._request_rating_duel(duel)

                duel_time = time.time() - duel_start_time
                self.track_average_duel_time(duel_time)

                if judgement is None:
                    bt.logging.warning(f"[{duel.left.uid}] vs [{duel.right.uid}] duel failed (validator error)")
                else:
                    bt.logging.debug(
                        f"[{duel.left.uid}] vs [{duel.right.uid}] duel judged. "
                        f"Duel time: {duel_time:.2f} sec ({self._average_duel_time:.2f} sec). "
                        f"Worst: {judgement.worst} ({duel.task.log_id})."
                    )

                    await self._process_results_queue.put((duel, judgement))

                self._pending_judgment_queue.task_done()
            except TimeoutError:
                await asyncio.sleep(1)
                continue
            except Exception as e:
                bt.logging.error(f"Judge worker {worker_id} failed with: {e}")
                self._pending_judgment_queue.task_done()

    async def _delay_next_duel(self, worker_id: int) -> None:
        """
        Maintain configured duel rate by delaying the next duel.

        Applies two types of delays:
        1. Validation backpressure - pause when validation is slow
        2. Rate limiting - maintain configured duels per second limit
        """

        await self._apply_validation_backpressure_delay(worker_id)
        await self._apply_rate_limiting_delay(worker_id)

    async def _apply_validation_backpressure_delay(self, worker_id: int) -> None:
        """Pause duels when validation time exceeds threshold to prevent GPU overload."""
        if not self._config.duels.pause_duels_on_slow_validation:
            return

        while self._validation_service.peak_validation_time() > self._config.duels.slow_validation_threshold:
            bt.logging.debug(f"Worker {worker_id}. High validation time. Duels paused")
            await asyncio.sleep(5)

    async def _apply_rate_limiting_delay(self, worker_id: int) -> None:
        """Apply rate limiting delay to maintain configured duels per second limit."""

        current_rate = self._average_duel_time / self._config.duels.judge_workers
        if 0.95 <= current_rate / self._target_duel_time <= 1.05:  # +/- 5% difference is acceptable.
            await asyncio.sleep(self._current_duel_delay)
            return

        prev_duel_delay = self._current_duel_delay
        self._current_duel_delay += (self._target_duel_time - current_rate) / 2
        # Gradually reaching the target

        bt.logging.debug(
            f"Maintaining duel rates. Worker: {worker_id}. "
            f"Current rate: {current_rate:.2f} sec/duel. "
            f"Target rate: {self._target_duel_time:.2f} sec/duel. "
            f"Delay update: {prev_duel_delay:.1f} sec -> {self._current_duel_delay:.1f} sec"
        )

        await asyncio.sleep(self._current_duel_delay)

    async def _request_rating_duel(self, duel: Duel) -> JudgeResponse | None:
        if duel.left.results is None or duel.right.results is None:  # pragma: no cover
            bt.logging.error(
                f"[{duel.left.uid}] and [{duel.right.uid}] duel undefined behavior, "
                f"no results when results are expected"
            )
            return None

        if duel.left.results.grid_preview is None or duel.right.results.grid_preview is None:  # pragma: no cover
            bt.logging.error(
                f"[{duel.left.uid}] and [{duel.right.uid}] duel undefined behavior, "
                f"no generated view when both views are expected"
            )
            return None

        if duel.task.protocol.type == "text":
            return await self._request_duel_text_prompt(
                prompt=duel.task.prompt,
                left_grid_preview_encoded=duel.left.results.grid_preview,
                right_grid_preview_encoded=duel.right.results.grid_preview,
                seed=duel.timestamp_nonce,
            )
        elif duel.task.protocol.type == "image":
            return await self._request_duel_image_prompt(
                prompt_encoded=duel.task.prompt,
                left_grid_preview_encoded=duel.left.results.grid_preview,
                right_grid_preview_encoded=duel.right.results.grid_preview,
                seed=duel.timestamp_nonce,
            )
        else:
            raise RuntimeError(f"Unknown protocol type: {duel.task.protocol.type}")
