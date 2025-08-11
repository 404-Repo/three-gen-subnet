import asyncio
import time

import bittensor as bt
import httpx
import pybase64
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from pydantic import BaseModel

from validator.duels.data_structures import Duel
from validator.validation_service import ValidationService


SYSTEM_PROMPT = "You are a specialized 3D model evaluation system. Analyze visual quality and prompt adherence with expert precision. Always respond with valid JSON only."  # noqa: E501

USER_PROMPT = """Evaluate two 3D models (4 renders each) generated from: "{prompt}"
Find the WORST one.

Problems to spot:
* Missing/extra objects from prompt
* Doesn't match what was requested
* Style mismatch (ONLY if prompt requests specific style - otherwise ANY style is fine)
* Major quality issues (artifacts, very low detail, broken geometry)

Compare BOTH directions: What's wrong with 1? What's wrong with 2?
Pick the WORST: 1, 2, or 0 (tie).

Output: {{"issues": "<1's problems> vs <2's problems>", "worst": <1/2/0>}}"""


DUEL_TIME_EMA_ALPHA = 0.2


class JudgeResponse(BaseModel):
    """Response from a judge evaluating a duel between two miners."""

    worst: int
    """Judge's decision: 1, 2 or 0 (draw)"""
    issues: str
    """Explanation of the scoring decision."""


class JudgeService:
    def __init__(
        self,
        *,
        config: bt.config,
        pending_judgment_queue: asyncio.Queue[Duel],
        process_results_queue: asyncio.Queue[tuple[Duel, JudgeResponse]],
        validation_service: ValidationService,
    ) -> None:
        self._config = config
        self._pending_judgment_queue: asyncio.Queue[Duel] = pending_judgment_queue
        self._process_results_queue: asyncio.Queue[tuple[Duel, JudgeResponse]] = process_results_queue

        self._client = AsyncOpenAI(
            base_url=self._config.duels.judge_endpoint,
            api_key=self._config.duels.judge_api_key,
            timeout=20.0,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_keepalive_connections=self._config.duels.judge_workers,
                    max_connections=self._config.duels.judge_workers * 2,
                )
            ),
        )
        """OpenAI client to communicate with the judge."""

        self._target_duel_time: float = max(1.0, 60 / self._config.duels.duels_per_minute)
        """
        Target average time spent per duel, in seconds (inference + delay).

        Used to:
        - Control costs in pay-per-duel setups.
        - Limit duel throughput to reduce load on a shared GPU.
        """

        self._current_duel_time: float = self._target_duel_time
        """
        Current average time spent per duel, in seconds (inference + delay),
        calculated as an exponential moving average (EMA).
        """

        self._current_duel_delay: float = 0.0
        """
        Current enforced delay between duels, in seconds, to maintain
        the target average duel time.
        """

        self._judge_workers: list[asyncio.Task] = []
        """Worker tasks for judging completed duels."""

        self._validation_service = validation_service
        """Reference to the validation service. Used to get the peak validation time."""

    async def start_judging_duels(self) -> None:
        for worker_id in range(self._config.duels.judge_workers):
            self._judge_workers.append(asyncio.create_task(self._judge_next_duel(worker_id)))

    def stop_judging_duels(self) -> None:
        for worker in self._judge_workers:
            worker.cancel()

    def judge_performance(self) -> float:
        if self._config.duels.judge_workers == 0:
            return 0.0
        perf: float = self._current_duel_time / self._config.duels.judge_workers
        return perf

    async def _judge_next_duel(self, worker_id: int) -> None:
        """Take the next duel from the queue and send it to the judge service."""

        while True:
            try:
                duel_start_time = time.time()

                await self._delay_next_duel(worker_id)

                duel = await asyncio.wait_for(self._pending_judgment_queue.get(), timeout=1.0)

                judgement = await self._request_duel(duel)

                duel_time = time.time() - duel_start_time
                self._current_duel_time = (
                    DUEL_TIME_EMA_ALPHA * duel_time + (1 - DUEL_TIME_EMA_ALPHA) * self._current_duel_time
                )

                if judgement is None:
                    bt.logging.warning(f"[{duel.left.uid}] vs [{duel.right.uid}] duel failed (validator error)")
                else:
                    bt.logging.debug(
                        f"[{duel.left.uid}] vs [{duel.right.uid}] duel judged. "
                        f"Duel time: {duel_time:.2f} sec ({self._current_duel_time:.2f} sec). "
                        f"Worst: {judgement.worst} ({duel.task.prompt[:100]})."
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

        current_rate = self._current_duel_time / self._config.duels.judge_workers
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

    async def _request_duel(self, duel: Duel) -> JudgeResponse | None:
        if duel.left.results is None or duel.right.results is None:  # pragma: no cover
            bt.logging.error(
                f"[{duel.left.uid}] and [{duel.right.uid}] duel undefined behavior, "
                f"no results when results are expected"
            )
            return None

        if duel.left.results.view is None or duel.right.results.view is None:  # pragma: no cover
            bt.logging.error(
                f"[{duel.left.uid}] and [{duel.right.uid}] duel undefined behavior, "
                f"no generated view when both views are expected"
            )
            return None

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT.format(prompt=duel.task.prompt)},
                    {"type": "text", "text": "First 3D model (multiple views):"},
                    {"type": "image_url", "image_url": {"url": image_to_base64(duel.left.results.view)}},
                    {"type": "text", "text": "Second 3D model (multiple views):"},
                    {"type": "image_url", "image_url": {"url": image_to_base64(duel.right.results.view)}},
                ],
            },
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge-response",
                "schema": JudgeResponse.model_json_schema(),
            },
        }
        completion: ChatCompletion = await self._client.chat.completions.create(
            model="THUDM/GLM-4.1V-9B-Thinking",
            messages=messages,  # type: ignore
            temperature=0.1,
            seed=duel.timestamp_nonce,
            max_tokens=1024,
            response_format=response_format,  # type: ignore
            timeout=20.0,
        )
        if not completion.choices:
            raise ValueError("Empty list of completion choices")

        message: ChatCompletionMessage = completion.choices[0].message

        if message.content is None:
            raise ValueError("Empty message content")

        return JudgeResponse.model_validate_json(message.content)


def image_to_base64(image: bytes) -> str:
    img_str = pybase64.b64encode(image).decode()
    return f"data:image/png;base64,{img_str}"
