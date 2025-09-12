import asyncio
import heapq
import time

import bittensor as bt
from common.protocol import ProtocolTask

from validator.duels.base_judge_service import BaseJudgeService, JudgeResponse
from validator.task_manager.task_storage.organic_task import DuelStatus, OrganicTask, OrganicTaskJudgeQueuePriority


class OrganicJudgeService(BaseJudgeService):
    def __init__(
        self,
        *,
        config: bt.config,
        pending_judgment_queue: asyncio.PriorityQueue[tuple[OrganicTaskJudgeQueuePriority, OrganicTask]],
        finished_judgment_queue: asyncio.Queue[OrganicTask],
    ) -> None:
        super().__init__(
            judge_endpoint=config.duels.judge_endpoint,
            judge_api_key=config.duels.judge_api_key,
            num_judge_workers=config.duels.judge_workers,
        )

        self._config = config
        self._pending_judgment_queue = pending_judgment_queue
        self._finished_judgment_queue = finished_judgment_queue

    async def _process_duels_loop(self, worker_id: int) -> None:
        """Take the next duel from the queue and send it to the judge service."""

        while True:
            await self._process_next_duel(worker_id)

    async def _process_next_duel(self, worker_id: int) -> None:
        try:
            _, task = await asyncio.wait_for(self._pending_judgment_queue.get(), timeout=1.0)
            self._pending_judgment_queue.task_done()

            if task.finalized:
                bt.logging.debug(f"Organic task is finalized. Skipping the task ({task.log_id}).")
                return

            if len(task.results_to_judge) < 2:
                bt.logging.debug(f"Not enough results to judge. Skipping the task ({task.log_id}).")
                return

            task.num_results_being_judged += 2

            _, left = heapq.heappop(task.results_to_judge)
            _, right = heapq.heappop(task.results_to_judge)

            if left.grid_preview is None:
                bt.logging.error(
                    f"[{left.uid}] Undefined behaviour. No grid preview for organic task ({task.log_id}).]"
                )
                judgement = JudgeResponse(worst=1, issues="No grid preview")
            elif right.grid_preview is None:
                bt.logging.error(
                    f"[{right.uid}] Undefined behaviour. No grid preview for organic task ({task.log_id}).]"
                )
                judgement = JudgeResponse(worst=2, issues="No grid preview")
            else:
                judgement = await self._request_organic_duel(
                    task=task.protocol,
                    left_grid_preview=left.grid_preview,
                    right_grid_preview=right.grid_preview,
                    seed=int(task.create_time),
                )
                task.num_results_judged += 2

            task.num_results_being_judged -= 2

            if judgement.worst == 0:
                left.duel_status = DuelStatus.DRAW
                right.duel_status = DuelStatus.DRAW
                best = left
            elif judgement.worst == 1:
                left.duel_status = DuelStatus.LOSS
                right.duel_status = DuelStatus.WIN
                best = right
            else:
                left.duel_status = DuelStatus.WIN
                right.duel_status = DuelStatus.LOSS
                best = left

            task.update_best(best)
            task.queue_for_judgment(best)
            await self._finished_judgment_queue.put(task)
        except TimeoutError:
            await asyncio.sleep(1)
        except Exception as e:
            bt.logging.error(f"Judge worker {worker_id} failed with: {e}")

    async def _request_organic_duel(
        self, *, task: ProtocolTask, left_grid_preview: str, right_grid_preview: str, seed: int
    ) -> JudgeResponse:
        try:
            duel_start_time = time.time()
            if task.type == "text":
                judgement = await self._request_duel_text_prompt(
                    prompt=task.prompt,
                    left_grid_preview_encoded=left_grid_preview,
                    right_grid_preview_encoded=right_grid_preview,
                    seed=seed,
                )
            elif task.type == "image":
                judgement = await self._request_duel_image_prompt(
                    prompt_encoded=task.prompt,
                    left_grid_preview_encoded=left_grid_preview,
                    right_grid_preview_encoded=right_grid_preview,
                    seed=seed,
                )
            else:
                raise RuntimeError(f"Unknown task type: {task.type}")
            duel_time = time.time() - duel_start_time
            self.track_average_duel_time(duel_time)
            bt.logging.debug(
                f"Organic task duel judged. "
                f"Duel time: {duel_time:.2f} sec ({self._average_duel_time:.2f} sec). "
                f"Worst: {judgement.worst} ({task.log_id})."
            )
        except Exception as e:
            bt.logging.warning(f"Organic task duel failed (validator error) with {e} ({task.log_id})")
            judgement = JudgeResponse(worst=0, issues="Duel failed")

        return judgement
