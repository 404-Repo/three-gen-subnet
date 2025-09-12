import asyncio
from typing import Any, cast

import bittensor as bt
import pybase64
import pytest
import time_machine
from asyncpg.pgproto.pgproto import timedelta
from pytest_httpserver import HTTPServer
from werkzeug import Request

from common.protocol import TextTask
from validator.duels.organic_judge_service import OrganicJudgeService
from validator.task_manager.task_storage.organic_task import OrganicTask, AssignedMiner, OrganicTaskJudgeQueuePriority
from validator.validator import Validator

from tests.test_validator.conftest import (
    ASSIGNED_MINERS_COUNT,
    GatewayApiMock,
    create_pull_task,
    create_submit_result,
    FROZEN_TIME,
    TASK_TIMEOUT,
    SECOND_MINER_TIMEOUT,
)


class TestGatewayTasks:

    @staticmethod
    def get_grid_preview_from_openai_request(request: Request, left: bool = False, right: bool = False) -> str:
        assert left != right
        idx = 2 if left else 4
        return pybase64.b64decode(
            request.json["messages"][1]["content"][idx]["image_url"]["url"][22:]  # type: ignore[index]
        ).decode()

    @pytest.mark.asyncio
    async def test_all_strong_miners_submit_on_time(
        self,
        time_travel: time_machine.Coordinates,
        validation_server: HTTPServer,
        judge_server: HTTPServer,
        validator: Validator,
        wallets: list[bt.Wallet],
    ) -> None:
        """
        Base scenario: all miners are strong, they pull the same task, compete, the best result is sent.
        """

        organic_task_storage = validator.task_manager._organic_task_storage

        await organic_task_storage._fetch_tasks_from_gateway()
        assert len(organic_task_storage._gateway_task_queue) > 0
        # validator fixture uses task_manager fixture;
        # task_manager fixture uses GatewayApiMock mock from the conftest.py;
        # GatewayApiMock is created with GATEWAY_TASK_COUNT (e.g. 100) predefined tasks.

        assert len(organic_task_storage._running_tasks) == 0
        assert len(organic_task_storage._running_task_queue) == 0

        strong_miners = list(validator.metagraph_sync._strong_miners)[:ASSIGNED_MINERS_COUNT]
        # Most probably strong miner uids will be 100-199 but why risk?

        pull = await validator.pull_task(create_pull_task(strong_miners[0], wallets))
        assert pull.task is not None
        assert organic_task_storage.has_task(task_id=pull.task.id)
        organic_task = organic_task_storage._running_task_queue[0]
        assert len(organic_task_storage._running_tasks) == 1
        assert len(organic_task_storage._running_task_queue) == 1
        assert organic_task.should_be_assigned(strong_miner=True, copies=ASSIGNED_MINERS_COUNT)

        for uid in strong_miners[1:]:
            other = await validator.pull_task(create_pull_task(uid, wallets))
            assert other.task == pull.task

        assert not organic_task.should_be_assigned(strong_miner=True, copies=ASSIGNED_MINERS_COUNT)
        assert len(organic_task.assigned_miners) == ASSIGNED_MINERS_COUNT

        for uid in strong_miners:
            validator.ratings._ratings[uid].glicko.rating = 1500 + uid
        # Setting miner ratings to get the initial order of the results.
        # Each next miner is better than all the previous ones.

        for idx, uid in enumerate(strong_miners, start=1):
            validation_server.expect_oneshot_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json(
                {
                    "score": 0.6 + idx / 100.0,
                    "grid_preview": pybase64.b64encode(f"grid preview {idx}".encode()).decode(),
                }
            )
            await validator.submit_results(create_submit_result(uid, pull.task, wallets))

            await asyncio.sleep(0)
            # Sleep to switch to the submit task coroutine.

            assert len(organic_task.results_to_judge) == idx
            assert organic_task.results_to_judge[0][1] == organic_task.assigned_miners[wallets[uid].hotkey.ss58_address]
            assert organic_task.best_result == organic_task.assigned_miners[wallets[uid].hotkey.ss58_address]

        assert organic_task.all_miners_finished(ASSIGNED_MINERS_COUNT)
        assert not organic_task.all_work_done(ASSIGNED_MINERS_COUNT)

        await organic_task_storage.start_judging_results()

        # Expected order of duels (based on the glicko rating set above):
        #   miner 4 vs miner 3 (winner 4, see the judge_server mock)
        #   miner 4 vs miner 2 (winner 4)
        #   miner 4 vs miner 1 (winner 4).

        while not organic_task.finalized:
            await asyncio.sleep(0)

        duel_requests = [req for req, res in judge_server.log if req.path == "/v1/chat/completions"]
        for idx, req in enumerate(duel_requests, start=1):
            left_preview = self.get_grid_preview_from_openai_request(req, left=True)
            right_preview = self.get_grid_preview_from_openai_request(req, right=True)

            assert left_preview == f"grid preview {ASSIGNED_MINERS_COUNT}"  # last one is always the best one
            assert right_preview == f"grid preview {ASSIGNED_MINERS_COUNT - idx}"  # 3, 2, 1

        assert organic_task.best_result == organic_task.assigned_miners[wallets[strong_miners[-1]].hotkey.ss58_address]
        assert organic_task.all_work_done(ASSIGNED_MINERS_COUNT)
        assert organic_task.finalized

        organic_task_storage.stop_judging_results()

        sent_results = cast(
            GatewayApiMock, validator.task_manager._organic_task_storage._gateway_manager._gateway_api
        ).results
        assert len(sent_results) == 1
        assert sent_results[organic_task.id].uid == strong_miners[-1]  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_assigned_miners_count(
        self,
        time_travel: time_machine.Coordinates,
        validation_server: HTTPServer,
        validator: Validator,
        wallets: list[bt.Wallet],
    ) -> None:
        """
        Scenario: only ASSIGNED_MINERS_COUNT weak miners pull the task.
        Scenario: strong miner can additionally pull the task but only one strong miner.
        """
        organic_task_storage = validator.task_manager._organic_task_storage

        await organic_task_storage._fetch_tasks_from_gateway()
        assert len(organic_task_storage._gateway_task_queue) > 0
        # validator fixture uses task_manager fixture;
        # task_manager fixture uses GatewayApiMock mock from the conftest.py;
        # GatewayApiMock is created with GATEWAY_TASK_COUNT (e.g. 100) predefined tasks.

        strong_miners = list(validator.metagraph_sync._strong_miners)
        weak_miners = [uid for uid in range(len(wallets)) if uid not in validator.metagraph_sync._strong_miners]

        pull = await validator.pull_task(create_pull_task(weak_miners[0], wallets))
        assert pull.task is not None

        for idx in range(1, ASSIGNED_MINERS_COUNT):
            other = await validator.pull_task(create_pull_task(weak_miners[idx], wallets))
            assert other.task == pull.task

        organic_task = organic_task_storage._running_task_queue[0]
        assert organic_task.should_be_assigned(strong_miner=True, copies=ASSIGNED_MINERS_COUNT)

        other = await validator.pull_task(create_pull_task(weak_miners[ASSIGNED_MINERS_COUNT], wallets))
        assert other.task != pull.task
        # When ASSIGNED_MINERS_COUNT weak miners pull the same task, next miner must pull another task.

        other = await validator.pull_task(create_pull_task(strong_miners[0], wallets))
        assert other.task == pull.task
        assert not organic_task.should_be_assigned(strong_miner=True, copies=ASSIGNED_MINERS_COUNT)
        # The first strong miner must pull the same task.

        other = await validator.pull_task(create_pull_task(strong_miners[1], wallets))
        assert other.task != pull.task
        # The second strong miner must pull another task.

    @pytest.mark.asyncio
    async def test_task_lifecycle(
        self,
        time_travel: time_machine.Coordinates,
        validation_server: HTTPServer,
        validator: Validator,
        wallets: list[bt.Wallet],
    ) -> None:
        """
        Scenario: task is no longer assigned if timeout passed.
        Scenario: task is recycled when expired.
        """
        organic_task_storage = validator.task_manager._organic_task_storage
        await organic_task_storage._fetch_tasks_from_gateway()

        pull = await validator.pull_task(create_pull_task(1, wallets))
        assert pull.task is not None

        organic_task = organic_task_storage._running_task_queue[0]
        assert organic_task.should_be_assigned(strong_miner=True, copies=ASSIGNED_MINERS_COUNT)

        time_travel.move_to(FROZEN_TIME + timedelta(seconds=TASK_TIMEOUT + 1))
        other = await validator.pull_task(create_pull_task(2, wallets))
        assert other.task != pull.task

        time_travel.move_to(FROZEN_TIME + timedelta(seconds=2 * TASK_TIMEOUT + 1))
        synthetic = await validator.pull_task(create_pull_task(3, wallets))
        # Pulling task triggers the task recycling.

        assert synthetic.task is not None
        assert not validator.task_manager._organic_task_storage.has_task(task_id=synthetic.task.id)
        assert validator.task_manager._synthetic_task_storage.has_task(task_id=synthetic.task.id)
        assert len(organic_task_storage._running_task_queue) == 0
        assert len(organic_task_storage._gateway_task_queue) == 0

        await asyncio.sleep(0)  # Sleep to switch to the task timeout coroutine.
        assert len(organic_task_storage._running_tasks) == 0

    @pytest.mark.asyncio
    async def test_task_timeouts(
        self,
        time_travel: time_machine.Coordinates,
        validation_server: HTTPServer,
        validator: Validator,
        wallets: list[bt.Wallet],
    ) -> None:
        """
        Scenario: intermediate task results are sent in case of the task timeout.
        """
        organic_task_storage = validator.task_manager._organic_task_storage
        await organic_task_storage._fetch_tasks_from_gateway()

        pull = await validator.pull_task(create_pull_task(1, wallets))
        assert pull.task is not None

        await validator.submit_results(create_submit_result(1, pull.task, wallets))
        await asyncio.sleep(0)  # sleep to switch to the taskmanager submit
        await asyncio.sleep(0)  # sleep to let wait_for start

        sent_results = cast(
            GatewayApiMock, validator.task_manager._organic_task_storage._gateway_manager._gateway_api
        ).results
        assert len(sent_results) == 0
        # No results are sent yet.

        time_travel.move_to(FROZEN_TIME + timedelta(seconds=SECOND_MINER_TIMEOUT + 2), tick=True)
        await asyncio.sleep(0.01)  # Sleep to switch to the task timeout coroutine.
        assert len(sent_results) == 1
        assert sent_results[pull.task.id].uid == 1  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_best_result_based_on_rating(
        self,
        wallets: list[bt.Wallet],
    ) -> None:
        organic_task = OrganicTask(protocol=TextTask(prompt="prompt"))
        for uid in range(1, 5):
            miner = AssignedMiner(uid=uid, hotkey=wallets[uid].hotkey.ss58_address, assign_time=0)
            organic_task.assigned_miners[miner.hotkey] = miner

        for hotkey, miner in organic_task.assigned_miners.items():
            miner.score = 1.0
            miner.rating = 1500 + miner.uid
            organic_task.update_best(miner)

        assert organic_task.best_result.rating == 1500 + 4  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_best_result_based_on_score(
        self,
        wallets: list[bt.Wallet],
    ) -> None:
        organic_task = OrganicTask(protocol=TextTask(prompt="prompt"))
        for uid in range(1, 5):
            miner = AssignedMiner(uid=uid, hotkey=wallets[uid].hotkey.ss58_address, assign_time=0)
            organic_task.assigned_miners[miner.hotkey] = miner

        for hotkey, miner in organic_task.assigned_miners.items():
            miner.score = 0.6 + miner.uid / 1000
            miner.rating = 1500
            organic_task.update_best(miner)

        assert organic_task.best_result.score == 0.6 + 4 / 1000  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_all_results_are_the_same(
        self,
        time_travel: time_machine.Coordinates,
        validation_server: HTTPServer,
        judge_server: HTTPServer,
        validator: Validator,
        wallets: list[bt.Wallet],
    ) -> None:
        """
        Scenario: all miners returned the same result, no judgement needed.
        """
        organic_task_storage = validator.task_manager._organic_task_storage
        await organic_task_storage._fetch_tasks_from_gateway()

        strong_miners = list(validator.metagraph_sync._strong_miners)[:ASSIGNED_MINERS_COUNT]

        pull = await validator.pull_task(create_pull_task(strong_miners[0], wallets))
        assert pull.task is not None
        organic_task = organic_task_storage._running_task_queue[0]

        for uid in strong_miners[1:]:
            await validator.pull_task(create_pull_task(uid, wallets))

        assert not organic_task.should_be_assigned(strong_miner=True, copies=ASSIGNED_MINERS_COUNT)

        for uid in strong_miners:
            await validator.submit_results(create_submit_result(uid, pull.task, wallets))

            await asyncio.sleep(0)
            # Sleep to switch to the submit task coroutine.

        assert organic_task.all_miners_finished(ASSIGNED_MINERS_COUNT)
        assert organic_task.all_work_done(ASSIGNED_MINERS_COUNT)

        while not organic_task.finalized:
            await asyncio.sleep(0)

        sent_results = cast(
            GatewayApiMock, validator.task_manager._organic_task_storage._gateway_manager._gateway_api
        ).results
        assert len(sent_results) == 1
        assert sent_results[organic_task.id].uid == strong_miners[0]  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_duel_skip_for_finalized_task(
        self,
        time_travel: time_machine.Coordinates,
        validation_server: HTTPServer,
        judge_server: HTTPServer,
        validator: Validator,
        wallets: list[bt.Wallet],
    ) -> None:
        """
        Scenario: organic task duel is pulled when task is being sent/expired already.
        """
        organic_task_storage = validator.task_manager._organic_task_storage
        await organic_task_storage._fetch_tasks_from_gateway()

        pull = await validator.pull_task(create_pull_task(1, wallets))
        assert pull.task is not None
        await validator.pull_task(create_pull_task(2, wallets))

        organic_task = organic_task_storage._running_task_queue[0]

        for uid in [1, 2]:
            validation_server.expect_oneshot_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json(
                {
                    "score": 0.6 + uid / 10.0,
                    "grid_preview": pybase64.b64encode(f"grid preview {uid}".encode()).decode(),
                }
            )
            await validator.submit_results(create_submit_result(uid, pull.task, wallets))

            await asyncio.sleep(0)
            # Sleep to switch to the submit task coroutine.

        time_travel.move_to(FROZEN_TIME + timedelta(seconds=SECOND_MINER_TIMEOUT + 2), tick=True)
        await asyncio.sleep(0.01)  # Sleep to switch to the task timeout coroutine.

        sent_results = cast(
            GatewayApiMock, validator.task_manager._organic_task_storage._gateway_manager._gateway_api
        ).results
        assert len(sent_results) == 1

        assert organic_task.finalized

        assert len(organic_task.results_to_judge) == 2  # Still some results to judge
        assert organic_task_storage._pending_judgment_queue.qsize() == 1  # Still one task enqueued.

        await organic_task_storage.start_judging_results()

        await organic_task_storage._pending_judgment_queue.join()

        duel_requests = [req for req, res in judge_server.log if req.path == "/v1/chat/completions"]
        assert len(duel_requests) == 0

        organic_task_storage.stop_judging_results()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("no_preview_miner_score", [0.7, 0.9])
    async def test_judge_no_preview(
        self,
        config: bt.config,
        wallets: list[bt.Wallet],
        judge_server: HTTPServer,
        no_preview_miner_score: float,
    ) -> None:
        """
        - Scenario: one of the miner has no preview
            miner with no preview has 0.7 score - miner is the second priority miner (right preview),
            miner with no preview has 0.9 score - miner is the first priority miner (left preview),
        """
        pending_judgment_queue: asyncio.PriorityQueue[tuple[OrganicTaskJudgeQueuePriority, OrganicTask]] = (
            asyncio.PriorityQueue()
        )
        finished_judgment_queue: asyncio.Queue[OrganicTask] = asyncio.Queue()
        judge_service = OrganicJudgeService(
            config=config,
            pending_judgment_queue=pending_judgment_queue,
            finished_judgment_queue=finished_judgment_queue,
        )
        organic_task = OrganicTask(protocol=TextTask(prompt="prompt"))
        miner = AssignedMiner(
            uid=1, hotkey=wallets[1].hotkey.ss58_address, assign_time=0, score=0.8, grid_preview="preview"
        )
        organic_task.assigned_miners[miner.hotkey] = miner

        no_preview_miner = AssignedMiner(
            uid=2, hotkey=wallets[2].hotkey.ss58_address, score=no_preview_miner_score, assign_time=0
        )
        organic_task.assigned_miners[no_preview_miner.hotkey] = no_preview_miner

        organic_task.queue_for_judgment(miner)
        organic_task.queue_for_judgment(no_preview_miner)

        await pending_judgment_queue.put((organic_task.judge_queue_priority(), organic_task))

        await judge_service._process_next_duel(worker_id=0)

        duel_requests = [req for req, res in judge_server.log if req.path == "/v1/chat/completions"]
        assert len(duel_requests) == 0

        assert len(organic_task.results_to_judge) == 1
        assert organic_task.results_to_judge[0][1].uid == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "judge_decision,best_uid",
        [
            (0, 1),  # Draw - the fist miner should win
            (1, 2),  # Left is worst - the second miner wins
            (2, 1),  # Right is worst - the first miner wins
            ("failure", 1),  # Failure=draw - the first miner wins.
        ],
    )
    async def test_judge_different_winner(
        self,
        config: bt.config,
        wallets: list[bt.Wallet],
        judge_server: HTTPServer,
        judge_decision: int,
        best_uid: int,
    ) -> None:
        pending_judgment_queue: asyncio.PriorityQueue[tuple[OrganicTaskJudgeQueuePriority, OrganicTask]] = (
            asyncio.PriorityQueue()
        )
        finished_judgment_queue: asyncio.Queue[OrganicTask] = asyncio.Queue()
        judge_service = OrganicJudgeService(
            config=config,
            pending_judgment_queue=pending_judgment_queue,
            finished_judgment_queue=finished_judgment_queue,
        )
        organic_task = OrganicTask(protocol=TextTask(prompt="prompt"))
        miner_1 = AssignedMiner(
            uid=1, hotkey=wallets[1].hotkey.ss58_address, assign_time=0, score=0.9, grid_preview="preview 1"
        )
        organic_task.assigned_miners[miner_1.hotkey] = miner_1

        miner_2 = AssignedMiner(
            uid=2, hotkey=wallets[2].hotkey.ss58_address, assign_time=0, score=0.8, grid_preview="preview 2"
        )
        organic_task.assigned_miners[miner_2.hotkey] = miner_2

        organic_task.queue_for_judgment(miner_1)
        organic_task.queue_for_judgment(miner_2)

        await pending_judgment_queue.put((organic_task.judge_queue_priority(), organic_task))

        judge_server.expect_oneshot_request("/v1/chat/completions", method="POST").respond_with_json(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            f"content": f'{{"issues": "issues", "worst": {judge_decision}}}',
                        },
                    }
                ]
            }
        )

        await judge_service._process_next_duel(worker_id=0)

        duel_requests = [req for req, res in judge_server.log if req.path == "/v1/chat/completions"]
        assert len(duel_requests) == 1

        assert len(organic_task.results_to_judge) == 1
        assert organic_task.results_to_judge[0][1].uid == best_uid

    @pytest.mark.asyncio
    async def test_all_miners_fail(
        self,
        time_travel: time_machine.Coordinates,
        validation_server: HTTPServer,
        judge_server: HTTPServer,
        validator: Validator,
        wallets: list[bt.Wallet],
    ) -> None:
        """
        Scenario: all miners are strong, they pull the same task and fail.
        """
        organic_task_storage = validator.task_manager._organic_task_storage
        await organic_task_storage._fetch_tasks_from_gateway()

        strong_miners = list(validator.metagraph_sync._strong_miners)[:ASSIGNED_MINERS_COUNT]

        pull = await validator.pull_task(create_pull_task(strong_miners[0], wallets))
        assert pull.task is not None

        organic_task = organic_task_storage._running_task_queue[0]
        for uid in strong_miners[1:]:
            await validator.pull_task(create_pull_task(uid, wallets))

        for idx, uid in enumerate(strong_miners, start=1):
            validation_server.expect_oneshot_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json(
                {
                    "score": 0.0,
                }
            )
            await validator.submit_results(create_submit_result(uid, pull.task, wallets))

            await asyncio.sleep(0)
            # Sleep to switch to the submit task coroutine.

        assert organic_task.all_miners_finished(ASSIGNED_MINERS_COUNT)
        assert organic_task.all_work_done(ASSIGNED_MINERS_COUNT)
        assert organic_task.best_result is None

        time_travel.move_to(FROZEN_TIME, tick=True)
        await asyncio.sleep(0.01)

        sent_results = cast(
            GatewayApiMock, validator.task_manager._organic_task_storage._gateway_manager._gateway_api
        ).results
        assert len(sent_results) == 1
        assert sent_results[organic_task.id] == "4 miners submitted results but no one is good"

    @pytest.mark.asyncio
    async def test_last_miners_fail(
        self,
        time_travel: time_machine.Coordinates,
        validation_server: HTTPServer,
        judge_server: HTTPServer,
        validator: Validator,
        wallets: list[bt.Wallet],
    ) -> None:
        """
        Scenario: all miners are strong, they pull the same task, two miners succeed, other fail.
        """
        organic_task_storage = validator.task_manager._organic_task_storage
        await organic_task_storage._fetch_tasks_from_gateway()

        strong_miners = list(validator.metagraph_sync._strong_miners)[:ASSIGNED_MINERS_COUNT]
        pull = await validator.pull_task(create_pull_task(strong_miners[0], wallets))
        assert pull.task is not None
        organic_task = organic_task_storage._running_task_queue[0]

        for uid in strong_miners[1:]:
            await validator.pull_task(create_pull_task(uid, wallets))

        for uid in strong_miners[:2]:
            validation_server.expect_oneshot_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json(
                {
                    "score": 0.6 + uid / 1000.0,
                    "grid_preview": pybase64.b64encode(f"grid preview {uid}".encode()).decode(),
                }
            )
            await validator.submit_results(create_submit_result(uid, pull.task, wallets))

            await asyncio.sleep(0)
            # Sleep to switch to the submit task coroutine.

        await organic_task_storage._judge_service._process_next_duel(worker_id=0)

        duel_requests = [req for req, res in judge_server.log if req.path == "/v1/chat/completions"]
        assert len(duel_requests) == 1
        left_preview = self.get_grid_preview_from_openai_request(duel_requests[0], left=True)
        right_preview = self.get_grid_preview_from_openai_request(duel_requests[0], right=True)
        assert left_preview == f"grid preview {strong_miners[1]}"
        assert right_preview == f"grid preview {strong_miners[0]}"

        for uid in strong_miners[2:]:
            validation_server.expect_oneshot_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json(
                {
                    "score": 0.0,
                }
            )
            await validator.submit_results(create_submit_result(uid, pull.task, wallets))

            await asyncio.sleep(0)
            # Sleep to switch to the submit task coroutine.

        assert organic_task.all_miners_finished(ASSIGNED_MINERS_COUNT)
        assert organic_task.all_work_done(ASSIGNED_MINERS_COUNT)

        time_travel.move_to(FROZEN_TIME, tick=True)
        await asyncio.sleep(0.01)

        sent_results = cast(
            GatewayApiMock, validator.task_manager._organic_task_storage._gateway_manager._gateway_api
        ).results
        assert len(sent_results) == 1
        assert sent_results[organic_task.id].uid == strong_miners[1]  # type: ignore[union-attr]
