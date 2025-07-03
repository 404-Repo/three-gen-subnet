import asyncio
import random as rd
import time
from typing import Any, cast
from uuid import uuid4

import bittensor as bt
import pytest
import time_machine
from pytest_httpserver import HTTPServer

from validator.duels.ratings import DuelRatings
from validator.gateway.gateway_api import GatewayApi
from validator.gateway.http3_client.http3_client import Http3Client, Http3Response
from validator.task_manager.task import (
    AssignedMiner,
    GatewayOrganicTask,
    LegacyOrganicTask,
    SyntheticTask,
    ValidatorTask,
)
from validator.task_manager.task_manager import TaskManager
from validator.task_manager.task_storage.organic_task_storage import OrganicTask
from validator.validation_service import ValidationService
from validator.validator import Validator

from tests.test_validator.conftest import (
    ASSIGNED_MINERS_COUNT,
    FROZEN_TIME,
    STRONG_MINER_COUNT,
    TASK_TIMEOUT,
    GatewayApiMock,
    create_pull_task,
    create_submit_result,
    get_validator_with_available_organic_tasks,
)
from tests.test_validator.subtensor_mocks import WALLETS


class TestGatewayTasks:

    @pytest.mark.asyncio
    async def test_task_order(
        self,
        reset_validation_server: ValidationService,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        Test that the task order is correct.
        First active organic tasks.
        Then gateway organic tasks.
        Then legacy organic tasks.
        Then synthetic tasks.

        All miners are strong. Initially we should take all gateway tasks (4 miners per each).
        Then we should take legacy organic tasks (4 miners per each).
        Then we should take synthetic tasks (1 miner per each).
        """
        async with get_validator_with_available_organic_tasks(
            config=config,
            subtensor=subtensor,
            task_manager=task_manager,
            ratings=ratings,
        ) as validator:
            with time_machine.travel(FROZEN_TIME, tick=False):
                tasks: list[ValidatorTask] = []
                time = time_machine.time()
                for idx in range(STRONG_MINER_COUNT):
                    time += 1
                    with time_machine.travel(time, tick=False):
                        pull_task = create_pull_task(idx)
                        pull_task = await validator.pull_task(pull_task)
                        assert pull_task.task is not None
                        is_synthetic = validator.task_manager._synthetic_task_storage.has_task(
                            task_id=pull_task.task.id
                        )
                        if not is_synthetic:
                            task = validator.task_manager._organic_task_storage._running_tasks.get(
                                pull_task.task.id, None
                            )
                            assert task is not None
                            tasks.append(task)
                        else:
                            tasks.append(SyntheticTask(id=pull_task.task.id, prompt=pull_task.task.prompt))
                        await validator.submit_results(synapse=create_submit_result(idx, pull_task.task))

            assert len(tasks) == STRONG_MINER_COUNT

            # Check the order of the tasks: initially GatewayOrganicTasks, then LegacyOrganicTasks, then SyntheticTasks.
            is_gateway_organic_task = True
            is_legacy_organic_task = False
            for t in tasks:
                if is_gateway_organic_task:
                    assert isinstance(t, GatewayOrganicTask | LegacyOrganicTask)
                    if isinstance(t, LegacyOrganicTask):
                        is_gateway_organic_task = False
                        is_legacy_organic_task = True
                elif is_legacy_organic_task:
                    assert isinstance(t, LegacyOrganicTask | SyntheticTask)
                    if isinstance(t, SyntheticTask):
                        is_legacy_organic_task = False
                else:
                    assert isinstance(t, SyntheticTask)

            # Check that each organic task has the same number of miners assigned to it.
            for t in tasks:
                if isinstance(t, GatewayOrganicTask | LegacyOrganicTask):
                    assert len(t.assigned_miners) == ASSIGNED_MINERS_COUNT

    @pytest.mark.asyncio
    async def test_active_legacy_task_is_prioritized(
        self,
        reset_validation_server: ValidationService,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        If a legacy task was selected by miner and gateway task was added, then
        active legacy task is prioritized.
        """
        async with get_validator_with_available_organic_tasks(
            config=config,
            subtensor=subtensor,
            task_manager=task_manager,
            ratings=ratings,
        ) as validator:
            with time_machine.travel(FROZEN_TIME, tick=False):
                # Remove all gateway tasks.
                validator.task_manager._organic_task_storage._gateway_task_queue.clear()

                # Pull legacy task
                pull_task = create_pull_task(1)
                pull_task = await validator.pull_task(pull_task)
                assert pull_task is not None
                assert pull_task.task is not None
                task = validator.task_manager._organic_task_storage._running_tasks.get(pull_task.task.id, None)
                assert task is not None
                assert isinstance(task, LegacyOrganicTask)

                # Add a gateway task to the queue manually
                validator.task_manager._organic_task_storage._gateway_task_queue.append(
                    GatewayOrganicTask.create_task(id=str(uuid4()), prompt="prompt", gateway_url="host")
                )

                # Pull task and check that is legacy.
                pull_task_2 = create_pull_task(2)
                pull_task_2 = await validator.pull_task(pull_task_2)
                assert pull_task_2.task is not None
                assert pull_task_2.task.id == pull_task.task.id
                assert pull_task_2.task.prompt == pull_task.task.prompt

    @pytest.mark.asyncio
    async def test_active_synthetic_tasks_are_not_prioritized(
        self,
        reset_validation_server: ValidationService,
        subtensor: bt.MockSubtensor,
        validator: Validator,
    ) -> None:
        """
        If a synthetic task was selected by miner and organic task was added, then
        active synthetic task is not prioritized.
        """
        with time_machine.travel(FROZEN_TIME, tick=False):
            pull_task = create_pull_task(1)
            pull_task = await validator.pull_task(pull_task)
            assert pull_task is not None
            assert pull_task.task is not None
            assert validator.task_manager._synthetic_task_storage.has_task(task_id=pull_task.task.id)

            # Add the gateway task to the queue manually
            validator.task_manager._organic_task_storage._gateway_task_queue.append(
                GatewayOrganicTask.create_task(id=str(uuid4()), prompt="prompt", gateway_url="host")
            )

            # Pull task and check that is organic.
            pull_task_2 = create_pull_task(2)
            pull_task_2 = await validator.pull_task(pull_task_2)
            assert pull_task_2.task is not None
            assert pull_task_2.task.id != pull_task.task.id
            task = validator.task_manager._organic_task_storage._running_tasks.get(pull_task_2.task.id, None)
            assert task is not None
            assert isinstance(task, GatewayOrganicTask)

    @pytest.mark.asyncio
    async def test_task_lifecycle(
        self,
        reset_validation_server: ValidationService,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        Test that task lifecycle is correct.
        Task can be assigned to the miner during task timeout.
        Task can't be assigned to the miner after task timeout.
        Task is deleted after double task timeout.
        """
        async with get_validator_with_available_organic_tasks(
            config=config,
            subtensor=subtensor,
            task_manager=task_manager,
            ratings=ratings,
        ) as validator:
            now = time.time()
            with time_machine.travel(now, tick=True) as travel:
                pull_task = create_pull_task(1)
                pull_task = await validator.pull_task(pull_task)
                assert pull_task is not None
                assert pull_task.task is not None
                assert not validator.task_manager._synthetic_task_storage.has_task(task_id=pull_task.task.id)
                task = validator.task_manager._organic_task_storage._running_tasks.get(pull_task.task.id, None)
                assert task is not None

                # Task can be assigned to the miner before task timeout.
                travel.move_to(now + TASK_TIMEOUT - 1)
                pull_task_2 = create_pull_task(2)
                pull_task_2 = await validator.pull_task(pull_task_2)
                assert pull_task_2 is not None
                assert pull_task_2.task is not None
                assert pull_task_2.task.id == pull_task.task.id

                # Task can't be assigned to the miner after task timeout.
                travel.move_to(now + TASK_TIMEOUT + 1)
                pull_task_3 = create_pull_task(3)
                pull_task_3 = await validator.pull_task(pull_task_3)
                assert pull_task_3 is not None
                assert pull_task_3.task is not None
                assert pull_task_3.task.id != pull_task.task.id

                # Task is deleted after double task timeout.
                travel.move_to(now + TASK_TIMEOUT * 2 + 1)
                pull_task_4 = create_pull_task(4)
                pull_task_4 = await validator.pull_task(pull_task_4)
                assert pull_task_4 is not None
                assert pull_task_4.task is not None
                assert pull_task_4.task.id != pull_task.task.id
                # Wait for the task to be removed from the running tasks.
                await asyncio.sleep(0.2)
                task = validator.task_manager._organic_task_storage._running_tasks.get(pull_task.task.id, None)
                assert task is None

    @pytest.mark.asyncio
    async def test_send_result_timeout_for_gateway_task(
        self,
        reset_validation_server: ValidationService,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        Test that task can be assigned to second miner during send_result_timeout.
        Then a task is not accessible, and the result is sent to the gateway.
        """
        config.task.organic.send_result_timeout = 1
        async with get_validator_with_available_organic_tasks(
            config=config,
            subtensor=subtensor,
            task_manager=task_manager,
            ratings=ratings,
        ) as validator:
            now = time.time()
            with time_machine.travel(now, tick=True):
                # First miner gets the task
                pull_task_1 = create_pull_task(1)
                pull_task_1 = await validator.pull_task(pull_task_1)
                assert pull_task_1 is not None
                assert pull_task_1.task is not None
                assert not validator.task_manager._synthetic_task_storage.has_task(task_id=pull_task_1.task.id)
                task = validator.task_manager._organic_task_storage._running_tasks.get(pull_task_1.task.id, None)
                assert task is not None
                assert isinstance(task, GatewayOrganicTask)

                # First miner submits the result
                await validator.submit_results(synapse=create_submit_result(1, pull_task_1.task))

                # Other miner can get the task before send_result_timeout expires
                pull_task_2 = create_pull_task(2)
                pull_task_2 = await validator.pull_task(pull_task_2)
                assert pull_task_2 is not None
                assert pull_task_2.task is not None
                assert pull_task_2.task.id == pull_task_1.task.id
                await asyncio.sleep(config.task.organic.send_result_timeout + 1)

                # Another miner can't get the task after send_result_timeout expires
                pull_task_3 = create_pull_task(10)
                pull_task_3 = await validator.pull_task(pull_task_3)
                assert pull_task_3 is not None
                assert pull_task_3.task is not None
                assert pull_task_3.task.id != pull_task_1.task.id

                # Task is removed from active tasks
                task = validator.task_manager._organic_task_storage._running_tasks.get(pull_task_1.task.id, None)
                assert task is None

                # The result should be in mock gateway storage.
                task_id = pull_task_1.task.id
                results = cast(
                    GatewayApiMock, validator.task_manager._organic_task_storage._gateway_manager._gateway_api
                ).results
                assert task_id in results

    @pytest.mark.asyncio
    async def test_send_result_timeout_for_legacy_task(
        self,
        reset_validation_server: ValidationService,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        Test that task can be assigned to second miner during send_result_timeout.
        Then a task is not accessible, and the result is sent to the gateway.
        """
        config.task.organic.send_result_timeout = 0.5
        async with get_validator_with_available_organic_tasks(
            config=config,
            subtensor=subtensor,
            task_manager=task_manager,
            ratings=ratings,
        ) as validator:
            # Remove all gateway tasks.
            validator.task_manager._organic_task_storage._gateway_task_queue.clear()

            now = time.time()
            with time_machine.travel(now, tick=True):
                # First miner gets the task
                pull_task_1 = create_pull_task(1)
                pull_task_1 = await validator.pull_task(pull_task_1)
                assert pull_task_1 is not None
                assert pull_task_1.task is not None
                assert not validator.task_manager._synthetic_task_storage.has_task(task_id=pull_task_1.task.id)
                task = validator.task_manager._organic_task_storage._running_tasks.get(pull_task_1.task.id, None)
                assert task is not None
                assert isinstance(task, LegacyOrganicTask)

                # First miner submits the result
                await validator.submit_results(synapse=create_submit_result(1, pull_task_1.task))

                # Other miner can get the task before send_result_timeout expires
                pull_task_2 = create_pull_task(2)
                pull_task_2 = await validator.pull_task(pull_task_2)
                assert pull_task_2 is not None
                assert pull_task_2.task is not None
                assert pull_task_2.task.id == pull_task_1.task.id
                result = await validator.task_manager._organic_task_storage.get_best_results(
                    task_id=pull_task_1.task.id
                )
                assert result is not None
                assert result.hotkey == WALLETS[1].hotkey.ss58_address

                # Other miner can't get the task after send_result_timeout expires
                pull_task_3 = create_pull_task(10)
                pull_task_3 = await validator.pull_task(pull_task_3)
                assert pull_task_3 is not None
                assert pull_task_3.task is not None
                assert pull_task_3.task.id != pull_task_1.task.id

                # Task is removed from active tasks
                task = validator.task_manager._organic_task_storage._running_tasks.get(pull_task_1.task.id, None)
                assert task is None

    @pytest.mark.asyncio
    async def test_task_count_limited_by_miner_count(
        self,
        reset_validation_server: ValidationService,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        Test that task count is limited by miner count.
        It is defined by config parameter `task.max_miners_per_task`.
        """
        async with get_validator_with_available_organic_tasks(
            config=config,
            subtensor=subtensor,
            task_manager=task_manager,
            ratings=ratings,
        ) as validator:
            task_id: str | None = None
            # The same task should be assigned to the miners below.
            for idx in range(config.task.organic.assigned_miners_count):
                pull_task = create_pull_task(idx)
                pull_task = await validator.pull_task(pull_task)
                assert pull_task is not None
                assert pull_task.task is not None
                if task_id is None:
                    task_id = pull_task.task.id
                assert pull_task.task.id == task_id

            # Then this task can't be assigned to the miner.
            pull_task = create_pull_task(config.task.organic.assigned_miners_count)
            pull_task = await validator.pull_task(pull_task)
            assert pull_task.task is not None
            assert pull_task.task.id != task_id

    @pytest.mark.asyncio
    async def test_task_can_be_assigned_to_strong_miner(
        self,
        reset_validation_server: ValidationService,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        Test that task can be assigned to a strong miner if it was not assigned to any miner before.
        """
        async with get_validator_with_available_organic_tasks(
            config=config,
            subtensor=subtensor,
            task_manager=task_manager,
            ratings=ratings,
        ) as validator:
            task_id: str | None = None
            # The same task should be assigned to the weak miners below.
            for idx in range(1, config.task.organic.assigned_miners_count + 1):
                pull_task = create_pull_task(idx)
                pull_task = await validator.pull_task(pull_task)
                assert pull_task is not None
                assert pull_task.task is not None
                if task_id is None:
                    task_id = pull_task.task.id
                assert pull_task.task.id == task_id

            # Then this task can't be assigned to the weak miner again.
            pull_task = create_pull_task(config.task.organic.assigned_miners_count + 2)
            pull_task = await validator.pull_task(pull_task)
            assert pull_task.task is not None
            assert pull_task.task.id != task_id

            # But we can assign this task to the strong miner.
            pull_task = create_pull_task(
                config.public_api.strong_miners_count + config.task.organic.assigned_miners_count + 1
            )
            pull_task = await validator.pull_task(pull_task)
            assert pull_task.task is not None
            assert pull_task.task.id == task_id

            # If strong miner was assigned then task became unavailable for strong miner again.
            pull_task = create_pull_task(
                config.public_api.strong_miners_count + config.task.organic.assigned_miners_count + 2
            )
            pull_task = await validator.pull_task(pull_task)
            assert pull_task.task is not None
            assert pull_task.task.id != task_id

    @pytest.mark.asyncio
    async def test_best_result_for_gateway_task_is_selected(
        self,
        reset_validation_server: HTTPServer,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        Test that best result is selected from the results.
        """
        config.task.organic.send_result_timeout = 1
        async with get_validator_with_available_organic_tasks(
            config=config,
            subtensor=subtensor,
            task_manager=task_manager,
            ratings=ratings,
        ) as validator:
            task: ValidatorTask | None = None
            random_ratings = [1500 + i for i in range(config.task.organic.assigned_miners_count)]
            rd.shuffle(random_ratings)
            # For 4 miners - shuffled list of [1500, 1501, 1502, 1503].

            random_scores = [(0.8 + i / 100) for i in range(config.task.organic.assigned_miners_count)]
            rd.shuffle(random_scores)

            # The same task should be assigned to the weak miners below.
            for idx in range(config.task.organic.assigned_miners_count):
                reset_validation_server.clear()
                reset_validation_server.expect_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json(
                    {"score": random_scores[idx]}
                )

                ratings._ratings[idx].glicko.rating = random_ratings[idx]

                # Pull task
                pull_task = create_pull_task(idx)
                pull_task = await validator.pull_task(pull_task)
                assert pull_task is not None
                assert pull_task.task is not None
                # Submit result
                result = await validator.submit_results(synapse=create_submit_result(idx, pull_task.task, full=True))
                assert result is not None
                assert result.feedback is not None
                assert result.feedback.validation_failed is False

                # Get task
                task = validator.task_manager._organic_task_storage._running_tasks.get(pull_task.task.id, None)
                assert task is not None
                miner_hotkey = WALLETS[idx].hotkey.ss58_address

                # Wait while background tasks in task_manager are finished.
                for _ in range(5):
                    await asyncio.sleep(config.task.organic.send_result_timeout / 100)
                assert task.assigned_miners[miner_hotkey].score == random_scores[idx]
                assert task.assigned_miners[miner_hotkey].finished

            # Wait for the results to be sent to the gateway mock.
            await asyncio.sleep(config.task.organic.send_result_timeout + 0.1)

            # Check that the best result is selected.
            assert task is not None
            best_result = cast(OrganicTask, task).get_best_result()
            assert best_result is not None
            max_miner_idx, max_rating = max(enumerate(random_ratings), key=lambda x: x[1])
            assert best_result.hotkey == WALLETS[max_miner_idx].hotkey.ss58_address
            assert best_result.rating == max_rating

            # Check that the result is sent to the gateway mock.
            assert pull_task.task is not None
            task_id = pull_task.task.id
            results = cast(
                GatewayApiMock, validator.task_manager._organic_task_storage._gateway_manager._gateway_api
            ).results
            assert task_id in results
            assert isinstance(results[task_id], AssignedMiner)
            assert cast(AssignedMiner, results[task_id]).hotkey == WALLETS[max_miner_idx].hotkey.ss58_address

    @pytest.mark.asyncio
    async def test_best_result_for_gateway_task_is_selected_same_rating(
        self,
        reset_validation_server: HTTPServer,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        Test that best result is selected from the results.
        """
        config.task.organic.send_result_timeout = 1
        async with get_validator_with_available_organic_tasks(
            config=config,
            subtensor=subtensor,
            task_manager=task_manager,
            ratings=ratings,
        ) as validator:
            task: ValidatorTask | None = None
            random_ratings = [1500 + i for i in range(config.task.organic.assigned_miners_count)]
            random_ratings[-1] = random_ratings[-2]
            rd.shuffle(random_ratings)
            # For 4 miners - shuffled list of [1500, 1501, 1502, 1502(!)].

            random_scores = [(0.8 + i / 100) for i in range(config.task.organic.assigned_miners_count)]
            rd.shuffle(random_scores)

            # The same task should be assigned to the weak miners below.
            for idx in range(config.task.organic.assigned_miners_count):
                reset_validation_server.clear()
                reset_validation_server.expect_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json(
                    {"score": random_scores[idx]}
                )

                ratings._ratings[idx].glicko.rating = random_ratings[idx]

                # Pull task
                pull_task = create_pull_task(idx)
                pull_task = await validator.pull_task(pull_task)
                assert pull_task is not None
                assert pull_task.task is not None
                # Submit result
                await validator.submit_results(synapse=create_submit_result(idx, pull_task.task, full=True))

                # Get task
                task = validator.task_manager._organic_task_storage._running_tasks[pull_task.task.id]
                miner_hotkey = WALLETS[idx].hotkey.ss58_address

                # Wait while background tasks in task_manager are finished.
                for _ in range(5):
                    await asyncio.sleep(config.task.organic.send_result_timeout / 100)
                assert task.assigned_miners[miner_hotkey].finished

            # Wait for the results to be sent to the gateway mock.
            await asyncio.sleep(config.task.organic.send_result_timeout + 0.1)

            # Check that the best result is selected.
            best_result = cast(OrganicTask, task).get_best_result()
            assert best_result is not None
            max_miner_idx, (max_rating, max_score) = max(
                enumerate(zip(random_ratings, random_scores)), key=lambda x: x[1]
            )
            assert best_result.hotkey == WALLETS[max_miner_idx].hotkey.ss58_address
            assert best_result.rating == max_rating
            assert best_result.score == max_score

            # Check that the result is sent to the gateway mock.
            assert pull_task.task is not None
            task_id = pull_task.task.id
            results = cast(
                GatewayApiMock, validator.task_manager._organic_task_storage._gateway_manager._gateway_api
            ).results
            assert cast(AssignedMiner, results[task_id]).hotkey == WALLETS[max_miner_idx].hotkey.ss58_address

    @pytest.mark.asyncio
    async def test_best_result_for_legacy_task_is_selected(
        self,
        reset_validation_server: HTTPServer,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        Test that best result is selected from the results.
        """
        config.task.organic.send_result_timeout = 1
        async with get_validator_with_available_organic_tasks(
            config=config,
            subtensor=subtensor,
            task_manager=task_manager,
            ratings=ratings,
        ) as validator:
            # Remove all gateway tasks.
            validator.task_manager._organic_task_storage._gateway_task_queue.clear()
            task: ValidatorTask | None = None
            random_ratings = [1500 + i for i in range(config.task.organic.assigned_miners_count)]
            rd.shuffle(random_ratings)

            random_scores = [(0.8 + i / 100) for i in range(config.task.organic.assigned_miners_count)]
            rd.shuffle(random_scores)

            # The same task should be assigned to the weak miners below.
            for idx in range(config.task.organic.assigned_miners_count):
                reset_validation_server.clear()
                reset_validation_server.expect_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json(
                    {"score": random_scores[idx]}
                )

                ratings._ratings[idx].glicko.rating = random_ratings[idx]

                # Pull task
                pull_task = create_pull_task(idx)
                pull_task = await validator.pull_task(pull_task)
                assert pull_task is not None
                assert pull_task.task is not None

                # Submit result
                result = await validator.submit_results(synapse=create_submit_result(idx, pull_task.task))
                assert result is not None
                assert result.feedback is not None
                assert result.feedback.validation_failed is False

                # Get task
                task = validator.task_manager._organic_task_storage._running_tasks.get(pull_task.task.id, None)
                assert task is not None
                miner_hotkey = WALLETS[idx].hotkey.ss58_address

                # Wait while background tasks in task_manager are finished.
                for _ in range(5):
                    await asyncio.sleep(config.task.organic.send_result_timeout / 100)
                assert task.assigned_miners[miner_hotkey].score == random_scores[idx]
                assert task.assigned_miners[miner_hotkey].finished

            # Wait for the best result
            assert pull_task.task is not None
            best_result = await validator.task_manager._organic_task_storage.get_best_results(task_id=pull_task.task.id)
            assert best_result is not None
            max_miner_idx, max_rating = max(enumerate(random_ratings), key=lambda x: x[1])
            assert best_result.hotkey == WALLETS[max_miner_idx].hotkey.ss58_address
            assert best_result.rating == max_rating

    @pytest.mark.asyncio
    async def test_incorrect_data_from_gateway(
        self,
        reset_validation_server: ValidationService,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        validator: Validator,
    ) -> None:
        """
        Test that task can be assigned to strong miner if it was not assigned to any miner before.
        """

        class Http3ClientMock(Http3Client):
            async def post(
                self,
                *,
                url: str,
                payload: dict[str, Any],
                headers: dict[str, Any] | None = None,
            ) -> Http3Response:
                bt.logging.error(f"Incorrect data from gateway: {payload}")
                return Http3Response(data="incorrect data", latency=0)

            async def get(
                self, *, url: str, headers: dict[str, Any] | None = None, payload: dict[str, Any] | None = None
            ) -> Http3Response:
                bt.logging.error(f"Incorrect data from gateway: {payload}")
                return Http3Response(data="incorrect data", latency=0)

        # Use mock because pytest doesn't support http3
        validator.task_manager._organic_task_storage._gateway_manager._gateway_api = GatewayApi()
        validator.task_manager._organic_task_storage._gateway_manager._gateway_api._http3_client = Http3ClientMock()
        asyncio.create_task(validator.task_manager._organic_task_storage.fetch_gateway_tasks_cron())
        for _ in range(20):
            await asyncio.sleep(0.1)

        # Pull task. It should work ok and be synthetic.
        pull_task = create_pull_task(1)
        pull_task = await validator.pull_task(pull_task)
        assert pull_task.task is not None
        assert validator.task_manager._synthetic_task_storage.has_task(task_id=pull_task.task.id)

        # Submit result. It should fail.
        result = await validator.submit_results(synapse=create_submit_result(1, pull_task.task))
        assert result is not None

    @pytest.mark.asyncio
    async def test_result_is_submitted_if_timeout_occurs(
        self,
        reset_validation_server: HTTPServer,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        Test that result is submitted if timeout occurs after first result.
        """
        config.task.organic.send_result_timeout = 1
        async with get_validator_with_available_organic_tasks(
            config=config,
            subtensor=subtensor,
            task_manager=task_manager,
            ratings=ratings,
        ) as validator:
            # Pull and submit the first task.
            reset_validation_server.clear()
            reset_validation_server.expect_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json(
                {"score": 0.9}
            )

            # Pull task
            pull_task = create_pull_task(1)
            pull_task = await validator.pull_task(pull_task)
            assert pull_task is not None
            assert pull_task.task is not None

            # Submit result
            result = await validator.submit_results(synapse=create_submit_result(1, pull_task.task))
            assert result is not None
            assert result.feedback is not None
            assert result.feedback.validation_failed is False

            # Wait for the best result
            for _ in range(11):
                await asyncio.sleep(config.task.organic.send_result_timeout / 10)
            gateway_result = cast(
                GatewayApiMock, validator.task_manager._organic_task_storage._gateway_manager._gateway_api
            ).results
            assert pull_task.task is not None
            assert pull_task.task.id in gateway_result
            assert isinstance(gateway_result[pull_task.task.id], AssignedMiner)
            assert cast(AssignedMiner, gateway_result[pull_task.task.id]).hotkey == WALLETS[1].hotkey.ss58_address
            assert cast(AssignedMiner, gateway_result[pull_task.task.id]).score == 0.9

    @pytest.mark.asyncio
    async def test_send_result_timeout_for_not_assigned_tasks(
        self,
        reset_validation_server: ValidationService,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        Test that task is removed when timeout occurs
        """
        task_manager._organic_task_storage._organic_task_expire_timeout = 0.2
        async with get_validator_with_available_organic_tasks(
            config=config,
            subtensor=subtensor,
            task_manager=task_manager,
            ratings=ratings,
        ) as validator:
            await asyncio.sleep(task_manager._organic_task_storage._organic_task_expire_timeout + 0.2)
            _ = await validator.pull_task(create_pull_task(1))
            # Wait for all old tasks are removed
            await asyncio.sleep(0.1)
            assert len(validator.task_manager._organic_task_storage._running_tasks) == 0
            assert len(validator.task_manager._organic_task_storage._gateway_task_queue) == 0
            assert len(validator.task_manager._organic_task_storage._legacy_task_queue) == 0

            assert (
                len(
                    cast(
                        GatewayApiMock,
                        validator.task_manager._organic_task_storage._gateway_manager._gateway_api,
                    ).results
                )
                > 0
            )
            for error in cast(
                GatewayApiMock,
                validator.task_manager._organic_task_storage._gateway_manager._gateway_api,
            ).results.values():
                assert type(error) is str
                assert "miners submitted results but no one is good" in error

    @pytest.mark.asyncio
    async def test_gateway_disabled_by_default(
        self,
        validator: Validator,
        reset_validation_server: ValidationService,
    ) -> None:
        """
        Test that gateway is disabled by default.
        """
        validator.task_manager._organic_task_storage._config.task.gateway.enabled = False

        asyncio.create_task(validator.task_manager._organic_task_storage.fetch_gateway_tasks_cron())
        for _ in range(10):
            await asyncio.sleep(0.1)

        assert len(validator.task_manager._organic_task_storage._running_tasks) == 0
        assert len(validator.task_manager._organic_task_storage._gateway_task_queue) == 0
