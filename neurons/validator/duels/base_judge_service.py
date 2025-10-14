import asyncio
from abc import ABC, abstractmethod

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageParam
from pydantic import BaseModel


SYSTEM_PROMPT = (
    "You are a specialized 3D model evaluation system. "
    "Analyze visual quality and prompt adherence with expert precision. "
    "Always respond with valid JSON only."
)

USER_PROMPT_TEXT = """Evaluate two 3D models (4 renders each) generated from: "{prompt}"
Find the WORST one.

Problems to spot:
* Missing/extra objects from prompt
* Doesn't match what was requested
* Style mismatch (ONLY if prompt requests specific style - otherwise ANY style is fine)
* Major quality issues (artifacts, very low detail, broken geometry)

Compare BOTH directions: What's wrong with 1? What's wrong with 2?
Pick the WORST: 1, 2, or 0 (tie).

Output: {{"issues": "<1's problems> vs <2's problems>", "worst": <1/2/0>}}"""

USER_PROMPT_IMAGE = """Which of the 3D models does match the object in the image prompt less?
Pick the WORST: 1, 2, or 0 (tie).

Output: {"issues": "<1's problems> vs <2's problems>", "worst": <1/2/0> }"""

DUEL_TIME_EMA_ALPHA = 0.2
BASE_DUEL_TIMEOUT_SECONDS = 20


class JudgeResponse(BaseModel):
    """Response from a judge evaluating a duel between two 3D model renders.

    A duel compares two sets of rendered views to determine which model
    has worse quality based on prompt adherence and visual criteria.
    """

    worst: int
    """Judge's decision: 1 (first is worst), 2 (second is worst), or 0 (tie)."""
    issues: str
    """Explanation of the scoring decision."""


class BaseJudgeService(ABC):
    """Abstract base service for evaluating 3D model quality through duels.

    Manages concurrent workers that compare pairs of 3D model renders using
    a vision-language model. A "duel" evaluates which of two models has worse
    quality based on prompt adherence and visual criteria.

    **Subclasses must implement:**
    - `_process_duels_loop(worker_id)`: Define how each worker acquires duel
      requests (e.g., from a queue), processes them via `_request_duel()`,
      and handles results. Include any metrics tracking needed.

    **Common patterns:**
    - Rate-limited proxy: Apply delays, track throughput, pause/resume
    - Task processor: Compare multiple outputs per task, find best result

    Attributes:
        _client: OpenAI client for judge API communication.
        _num_judge_workers: Number of concurrent worker tasks.
        _judge_worker_tasks: Active asyncio Task objects.
    """

    def __init__(
        self,
        *,
        judge_endpoint: str,
        judge_api_key: str,
        num_judge_workers: int,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=judge_endpoint,
            api_key=judge_api_key,
            timeout=BASE_DUEL_TIMEOUT_SECONDS,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_keepalive_connections=num_judge_workers,
                    max_connections=num_judge_workers * 2,
                )
            ),
        )
        """OpenAI client to communicate with the judge."""

        self._average_duel_time: float = 0
        """Exponential moving average (EMA) of duel time, in seconds."""

        self._num_judge_workers = num_judge_workers
        """Number of concurrent workers requesting duels."""

        self._judge_worker_tasks: list[asyncio.Task] = []
        """Active worker tasks running `_process_duels_loop`."""

    async def start_judging_duels(self) -> None:
        """Start all judge worker tasks.

        Spawns the configured number of worker tasks, each executing the
        abstract _process_duels_loop method. Workers run concurrently and
        continue until explicitly stopped via stop_judging_duels().
        """

        for worker_id in range(self._num_judge_workers):
            self._judge_worker_tasks.append(asyncio.create_task(self._process_duels_loop(worker_id)))

    def stop_judging_duels(self) -> None:
        """Stop all judge worker tasks.

        Cancels all running worker tasks. Any in-flight duel requests
        will be interrupted. Pending items in queues will not be processed.
        """

        for worker in self._judge_worker_tasks:
            worker.cancel()

    def judge_performance(self) -> float:
        """Calculate the average time per duel across all workers."""

        if self._num_judge_workers == 0:
            return 0.0
        perf: float = self._average_duel_time / self._num_judge_workers
        return perf

    def track_average_duel_time(self, duel_time: float) -> None:
        self._average_duel_time = DUEL_TIME_EMA_ALPHA * duel_time + (1 - DUEL_TIME_EMA_ALPHA) * self._average_duel_time

    @abstractmethod
    async def _process_duels_loop(self, worker_id: int) -> None:
        """Main processing loop for a worker task.

        This method defines how each worker acquires and processes duels.
        It should run continuously until the task is cancelled.

        **Implementation Guidelines:**

        - Use an infinite loop with proper asyncio cancellation handling
        - Acquire duel requests from the appropriate source (queue, etc.)
        - Call _request_duel methods to evaluate model pairs
        - Handle results according to the service's purpose
        - Implement any necessary metrics tracking
        - Include error handling and logging as appropriate
        """
        raise NotImplementedError()

    async def _request_duel_text_prompt(
        self, *, prompt: str, left_grid_preview_encoded: str, right_grid_preview_encoded: str, seed: int
    ) -> JudgeResponse:
        """Request judgment for a duel between two 3D models generated from a text prompt.

        Sends the text prompt and rendered views to the judge model for evaluation.
        The judge compares both models and determines which one is worse based on prompt adherence and visual quality.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT_TEXT.format(prompt=prompt)},
                    {"type": "text", "text": "First 3D model (4 different views):"},
                    {"type": "image_url", "image_url": {"url": image_to_uri(left_grid_preview_encoded)}},
                    {"type": "text", "text": "Second 3D model (4 different views):"},
                    {"type": "image_url", "image_url": {"url": image_to_uri(right_grid_preview_encoded)}},
                ],
            },
        ]
        return await self._request_duel(messages=messages, seed=seed, max_tokens=1024)  # type: ignore

    async def _request_duel_image_prompt(
        self, *, prompt_encoded: str, left_grid_preview_encoded: str, right_grid_preview_encoded: str, seed: int
    ) -> JudgeResponse:
        """Request judgment for a duel between two 3D models generated from an image prompt.

        Sends the text prompt and rendered views to the judge model for evaluation.
        The judge compares both models and determines which one is worse based on prompt adherence and visual quality.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image prompt to generate 3D model:"},
                    {"type": "image_url", "image_url": {"url": image_to_uri(prompt_encoded)}},
                    {"type": "text", "text": "First 3D model (4 different views):"},
                    {"type": "image_url", "image_url": {"url": image_to_uri(left_grid_preview_encoded)}},
                    {"type": "text", "text": "Second 3D model (4 different views):"},
                    {"type": "image_url", "image_url": {"url": image_to_uri(right_grid_preview_encoded)}},
                    {"type": "text", "text": USER_PROMPT_IMAGE},
                ],
            },
        ]
        return await self._request_duel(messages=messages, seed=seed, max_tokens=1024)  # type: ignore

    async def _request_duel(self, *, messages: ChatCompletionMessageParam, seed: int, max_tokens: int) -> JudgeResponse:
        """Base method for requesting judgment from the judge model.

        Called internally by _request_duel_text_prompt and _request_duel_image_prompt.
        Handles the API call, response validation, and JSON parsing.

        Not intended to be called directly - use the specific text or image methods instead.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "judge-response",
                "schema": JudgeResponse.model_json_schema(),
            },
        }
        completion: ChatCompletion = await self._client.chat.completions.create(
            model="THUDM/GLM-4.1V-9B-Thinking",
            messages=messages,
            temperature=0.1,
            seed=seed,
            max_tokens=max_tokens,
            response_format=response_format,  # type: ignore
            timeout=20.0,
        )
        if not completion.choices:
            raise ValueError("Empty list of completion choices")

        message: ChatCompletionMessage = completion.choices[0].message

        if message.content is None:
            raise ValueError("Empty message content")

        return JudgeResponse.model_validate_json(message.content)


def image_to_uri(image_encoded: str) -> str:
    """Convert image bytes to base64 data URI."""
    return f"data:image/png;base64,{image_encoded}"
