import asyncio
import time

import bittensor as bt
import httpx
import pybase64
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from pydantic import BaseModel

from validator.duels.data_structures import Duel


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
        pending_judgement_queue: asyncio.Queue[Duel],
        process_results_queue: asyncio.Queue[tuple[Duel, JudgeResponse]],
    ) -> None:
        self._config = config
        self._pending_judgement_queue: asyncio.Queue[Duel] = pending_judgement_queue
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
        """Open AI client to communicate with the judge."""

        self._duels_start: float = 0
        """First duel start time."""

        self._duels_judged: int = 0
        """Total number of duels judged during a validator runtime."""

        self._duels_per_second_limit: float = min(1.0, self._config.duels.duels_per_minute / 60.0)
        """Rate limit (enforced when using pay-per-request billing)."""

        self._judge_workers: list[asyncio.Task] = []
        """Worker tasks for judging completed duels."""

    async def start_judging_duels(self) -> None:
        for worker_id in range(self._config.duels.judge_workers):
            self._judge_workers.append(asyncio.create_task(self._judge_next_duel(worker_id)))

    def stop_judging_duels(self) -> None:
        for worker in self._judge_workers:
            worker.cancel()

    async def _judge_next_duel(self, worker_id: int) -> None:
        """Take the next duel from the queue and send it to the judge service."""

        while True:
            try:
                await self._delay_next_duel(worker_id)

                duel = await asyncio.wait_for(self._pending_judgement_queue.get(), timeout=1.0)

                if self._duels_start == 0:
                    self._duels_start = time.time()

                judgement = await self._request_duel(duel)
                self._duels_judged += 1

                if judgement is None:
                    bt.logging.warning(f"[{duel.left.uid}] vs [{duel.right.uid}] duel failed (validator fault)")
                else:
                    judge_perfomance: float = (time.time() - self._duels_start) / self._duels_judged
                    bt.logging.debug(
                        f"[{duel.left.uid}] vs [{duel.right.uid}] duel judged ({judge_perfomance:.2f} sec). "
                        f"Worst: {judgement.worst} ({duel.task.prompt[:100]})."
                    )

                    await self._process_results_queue.put((duel, judgement))

                self._pending_judgement_queue.task_done()
            except TimeoutError:
                await asyncio.sleep(1)
                continue
            except Exception as e:
                bt.logging.error(f"Judge worker {worker_id} failed with: {e}")
                self._pending_judgement_queue.task_done()

    async def _delay_next_duel(self, worker_id: int) -> None:
        """Maintain a configured duel rate by delaying the next duel."""

        since_start = max(1.0, time.time() - self._duels_start)
        current_judge_rate: float = self._duels_judged / since_start
        if current_judge_rate > self._duels_per_second_limit:
            delay = self._duels_judged / self._duels_per_second_limit - since_start
            bt.logging.debug(f"Judge worker {worker_id}: {current_judge_rate} duels/second. Delay: {delay} seconds")
            await asyncio.sleep(delay)

    async def _request_duel(self, duel: Duel) -> JudgeResponse | None:
        if duel.left.results is None or duel.right.results is None:  # pragma: no cover
            bt.logging.error(
                f"[{duel.left.uid}] and [{duel.right.uid}] duel undefined behaviour, "
                f"no results when results are expected"
            )
            return None

        if duel.left.results.view is None or duel.right.results.view is None:  # pragma: no cover
            bt.logging.error(
                f"[{duel.left.uid}] and [{duel.right.uid}] duel undefined behaviour, "
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
