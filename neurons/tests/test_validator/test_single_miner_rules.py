import asyncio
import base64
import time
from collections import deque
from datetime import timedelta
from typing import cast

import bittensor as bt
import pybase64
import pyspz
import pytest
import time_machine
from common.protocol import Feedback, Task
from pytest_httpserver import HTTPServer

from validator.duels.ratings import DuelRatings
from validator.task_manager.task import AssignedMiner
from validator.task_manager.task_manager import TaskManager
from validator.validator import Validator

from tests.test_validator.conftest import (
    COOLDOWN_VIOLATION_PENALTY,
    COOLDOWN_VIOLATIONS_THRESHOLD,
    FROZEN_TIME,
    TASK_COOLDOWN,
    TASK_COOLDOWN_PENALTY,
    TASK_THROTTLE_PERIOD,
    TASK_TIMEOUT,
    VALIDATION_SCORE,
    GatewayApiMock,
    create_pull_task,
    create_submit_result,
    get_validator_with_available_organic_tasks,
)
from tests.test_validator.subtensor_mocks import WALLETS


class TestSingleMinerRules:

    @pytest.mark.asyncio
    async def test_pull_task(self, validator: Validator, reset_validation_server: HTTPServer) -> None:
        """
        Task for valid miner can't be None.
        """
        pull = await validator.pull_task(create_pull_task(1))
        assert pull.task is not None

    @pytest.mark.asyncio
    async def test_pull_task_from_unknown_miner(
        self, validator: Validator, reset_validation_server: HTTPServer
    ) -> None:
        """Unknown miner can't pull a task."""
        result = await validator.pull_task(create_pull_task(None))
        assert result.task is None

    @pytest.mark.asyncio
    async def test_pull_repeated_task(self, validator: Validator, reset_validation_server: HTTPServer) -> None:
        """If miner pulls the same task many times, then it will receive this task again."""

        now = time.time()
        with time_machine.travel(now, tick=False) as travel:
            pull = await validator.pull_task(create_pull_task(1))
            assert pull.task is not None
            task_id = pull.task.id

            for _ in range(10):
                # Initiate task expiration.
                validator.task_manager._organic_task_storage._remove_expired_tasks()
                now += 1
                travel.move_to(now)
                pull_next = await validator.pull_task(create_pull_task(1))
                assert pull_next.task is not None
                assert pull_next.task.id == task_id
                assert validator.miners[1].assigned_task is not None
                assert validator.miners[1].assigned_task.id == task_id

            synapse = create_submit_result(1, pull.task)
            await validator.submit_results(synapse)
            travel.move_to(synapse.cooldown_until + 1)
            pull_next = await validator.pull_task(create_pull_task(1))
            assert pull_next.task is not None
            assert pull_next.task.id != task_id
            assert validator.miners[1].assigned_task is not None
            assert validator.miners[1].assigned_task.id == pull_next.task.id

    @pytest.mark.asyncio
    async def test_pull_task_from_miner_on_cooldown(
        self,
        validator: Validator,
        time_travel: time_machine.travel,
        reset_validation_server: HTTPServer,
    ) -> None:
        """Miner can't pull a task if it is on cooldown.
        In this case, the task is not assigned and cooldown is increased.
        """
        existing_cooldown = int(time_machine.time()) + TASK_COOLDOWN
        validator.miners[1].cooldown_until = existing_cooldown
        pull = await validator.pull_task(create_pull_task(1))

        assert pull.task is None
        assert validator.miners[1].cooldown_violations == 1
        assert pull.cooldown_until == existing_cooldown

    async def test_cooldown_penalty(
        self, validator: Validator, time_travel: time_machine.travel, reset_validation_server: HTTPServer
    ) -> None:
        """If miner frequently pulled tasks being on cooldown, the cooldown penalty is applied."""

        # Set miner on cooldown
        existing_cooldown = int(time_machine.time()) + TASK_COOLDOWN
        validator.miners[1].cooldown_until = existing_cooldown
        validator.miners[1].cooldown_violations = COOLDOWN_VIOLATIONS_THRESHOLD - 1

        # Cooldown penalty is not applied but miner is on cooldown
        pull = await validator.pull_task(create_pull_task(1))
        assert pull.task is None
        assert pull.cooldown_until == existing_cooldown
        assert validator.miners[1].cooldown_until == existing_cooldown
        assert validator.miners[1].cooldown_violations == COOLDOWN_VIOLATIONS_THRESHOLD

        # Cooldown penalty is applied several times
        # It should lead to the increase of cooldown violations
        for idx in range(1, 4):
            pull = await validator.pull_task(create_pull_task(1))
            assert pull.task is None
            assert pull.cooldown_until == existing_cooldown + COOLDOWN_VIOLATION_PENALTY * idx
            assert validator.miners[1].cooldown_until == pull.cooldown_until
            assert validator.miners[1].cooldown_violations == COOLDOWN_VIOLATIONS_THRESHOLD + idx

        # After cooldown penalty period, miner can pull task again
        next_pull_time = existing_cooldown + COOLDOWN_VIOLATION_PENALTY * 3
        with time_machine.travel(next_pull_time, tick=False):
            pull = await validator.pull_task(create_pull_task(1))
            assert pull.task is not None
            assert pull.cooldown_until == 0
            assert validator.miners[1].cooldown_until == existing_cooldown + COOLDOWN_VIOLATION_PENALTY * 3
            assert validator.miners[1].cooldown_violations == COOLDOWN_VIOLATIONS_THRESHOLD + 3

            # If miner submits the result, then cooldown violations are decreased by 1
            await validator.submit_results(create_submit_result(1, pull.task))
            assert validator.miners[1].cooldown_violations == COOLDOWN_VIOLATIONS_THRESHOLD + 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "completion_time,expected_remaining_cooldown",
        [
            (5, 55),  # 60 - 5 = 55s cooldown
            (15, 45),  # 60 - 15 = 45s cooldown
            (20, 40),  # 60 - 20 = 40s cooldown
            (30, 40),  # Still 40s cooldown (capped at throttle period reduction)
        ],
    )
    async def test_miner_cooldown_after_successful_submission(
        self,
        validator: Validator,
        httpserver: HTTPServer,
        time_travel: time_machine.travel,
        completion_time: int,
        expected_remaining_cooldown: int,
        reset_validation_server: HTTPServer,
    ) -> None:
        """
        When miner submits a task successfully, cooldown id defined in MinerData.reset_task().

        TASK_COOLDOWN = 60
        TASK_THROTTLE_PERIOD = 20
        Case 1.
        The Task was assigned to miner. It did it 5 seconds that is less than a throttle period.
        So cooldown until is assigned time + cooldown = 60 seconds.
        The Remaining cooldown is 60-5=55 seconds.

        Case 2.
        The Task was assigned to miner. It did it 15 seconds that are less than a throttle period.
        So cooldown until is assigned time and cooldown 60 seconds.
        The Remaining cooldown is 60-15=45 seconds.

        Case 3.
        The Task was assigned to miner. It did it 20 seconds that is equal to a throttle period.
        So cooldown until is assigned time and cooldown 60 seconds.
        The Remaining cooldown is 60-20=40 seconds.

        Case 4.
        The Task was assigned to miner. It did it 30 seconds that is more than cooldown.
        Such a cooldown until is
        task submit time - throttle period and cooldown 60 seconds = 30-20+60=70 seconds.
        The Remaining cooldown is 70-30 = 40 seconds.
        """
        pull = await validator.pull_task(create_pull_task(1))
        assert pull.task is not None
        with time_machine.travel(FROZEN_TIME + timedelta(seconds=completion_time), tick=False):
            submit = await validator.submit_results(create_submit_result(1, pull.task))
            assert submit.cooldown_until == int(time_machine.time()) + expected_remaining_cooldown
            assert validator.miners[1].cooldown_until == submit.cooldown_until

    @pytest.mark.asyncio
    async def test_submit_task_unknown_neuron(self, validator: Validator, reset_validation_server: HTTPServer) -> None:
        """When miner is unknown, it can't submit successful task."""
        pull = await validator.pull_task(create_pull_task(1))
        assert pull.task is not None
        submit = await validator.submit_results(create_submit_result(None, pull.task))

        assert submit.feedback is None
        assert validator.miners[1].assigned_task == pull.task

    @pytest.mark.asyncio
    async def test_submit_task_wrong_task(self, validator: Validator, reset_validation_server: HTTPServer) -> None:
        """Miner can't submit an unknown task."""
        pull = await validator.pull_task(create_pull_task(1))
        submit = await validator.submit_results(create_submit_result(1, Task(prompt="invalid task")))

        assert submit.results == ""
        assert submit.feedback == Feedback(average_fidelity_score=1, current_duel_rating=1500)
        assert submit.feedback.task_fidelity_score == 0.0
        assert validator.miners[1].assigned_task == pull.task

    @pytest.mark.asyncio
    async def test_miner_skips_the_task(
        self, validator: Validator, time_travel: time_machine.travel, reset_validation_server: HTTPServer
    ) -> None:
        """Miner can skip the task."""
        pull = await validator.pull_task(create_pull_task(1))
        assert pull.task is not None
        synapse = create_submit_result(1, pull.task)
        synapse.results = ""  # Task skip
        submit = await validator.submit_results(synapse)

        assert submit.feedback == Feedback(average_fidelity_score=1, current_duel_rating=1500)
        assert validator.miners[1].assigned_task is None
        assert validator.miners[1].cooldown_until == int(time_machine.time()) + TASK_COOLDOWN

    @pytest.mark.asyncio
    async def test_submit_task_with_invalid_signature(
        self, validator: Validator, reset_validation_server: HTTPServer
    ) -> None:
        """
        Miner can't submit a task with invalid signature.
        In this case, additionally, TASK_COOLDOWN_PENALTY is applied.
        """
        pull = await validator.pull_task(create_pull_task(1))
        assert pull.task is not None
        synapse = create_submit_result(1, pull.task)
        synapse.signature = pybase64.b64encode(WALLETS[1].hotkey.sign("")).decode("utf-8")
        submit = await validator.submit_results(synapse)

        assert submit.feedback == Feedback(average_fidelity_score=1, current_duel_rating=1500)
        assert validator.miners[1].assigned_task is None
        assert validator.miners[1].cooldown_until == int(time_machine.time()) + TASK_COOLDOWN + TASK_COOLDOWN_PENALTY

    @pytest.mark.asyncio
    async def test_submit_task_validation_failure(
        self, validator: Validator, time_travel: time_machine.travel, validation_server: HTTPServer
    ) -> None:
        """
        When miner submits bad asset then validation gives bad score interpreted as 0.
        In this case cooldown penalty is applied to miner and the score is 0.
        """
        validation_server.clear()
        validation_server.expect_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json({"score": 0.23})

        pull = await validator.pull_task(create_pull_task(1))
        assert pull.task is not None
        submit = await validator.submit_results(create_submit_result(1, pull.task))

        assert submit.results == ""
        assert submit.feedback is not None
        assert submit.feedback.task_fidelity_score == 0
        assert not submit.feedback.validation_failed
        assert submit.cooldown_until == int(time_machine.time()) + TASK_COOLDOWN + TASK_COOLDOWN_PENALTY
        assert validator.miners[1].observations == deque([])

    @pytest.mark.asyncio
    async def test_submit_task_window_check(self, validator: Validator, reset_validation_server: HTTPServer) -> None:
        """
        When the miner submits the task, observations are added to his observation list.
        It contains the time of the task submission.
        It contains only tasks for the given window - last 4 hours.
        """
        with time_machine.travel(FROZEN_TIME, tick=False):
            first_time = int(time_machine.time())
            pull = await validator.pull_task(create_pull_task(1))
            assert pull.task is not None
            await validator.submit_results(create_submit_result(1, pull.task))
            assert validator.miners[1].observations == deque([first_time])

        with time_machine.travel(FROZEN_TIME + timedelta(hours=3, minutes=59), tick=False):
            second_time = int(time_machine.time())
            pull = await validator.pull_task(create_pull_task(1))
            assert pull.task is not None
            await validator.submit_results(create_submit_result(1, pull.task))
            assert validator.miners[1].observations == deque([first_time, second_time])

        with time_machine.travel(FROZEN_TIME + timedelta(hours=4, minutes=1), tick=False):
            third_time = int(time_machine.time())
            pull = await validator.pull_task(create_pull_task(1))
            assert pull.task is not None
            await validator.submit_results(create_submit_result(1, pull.task))
            assert validator.miners[1].observations == deque([second_time, third_time])

    @pytest.mark.asyncio
    async def test_miner_reward_over_multiple_submissions(
        self,
        config: bt.config,
        validator: Validator,
        time_travel: time_machine.travel,
        reset_validation_server: HTTPServer,
    ) -> None:
        """
        Tests how miner rewards changes over 10 consecutive submissions, one hour apart.

        This test:
        1. Verifies the reward calculation formula (observations * fidelity_score)
        2. Track how the average fidelity score (EMA) evolves with each submission
        3. Confirms that observations outside the 4-hour window are expired
        """
        # Set up validation server to return a constant score
        reset_validation_server.clear()
        reset_validation_server.expect_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json(
            {"score": 0.8}
        )

        # Verify initial state
        default_reward_rating = 1500.0
        assert validator.ratings.get_miner_reward_rating(1) == default_reward_rating
        assert validator.miners[1].fidelity_score == 1.0
        assert len(validator.miners[1].observations) == 0

        # Perform 10 submissions, 1 hour apart
        # | - 1 hour - | - 1 hour - | - 1 hour - | - 1 hour - | - 1 hour - |
        # 4 hours window, not including the left value if it's exactly on the line.
        moving_average_alpha = config.validation.moving_average_alpha
        expected_fidelity = 1.0  # Initial fidelity score
        base_time = FROZEN_TIME
        for i in range(10):
            # Move time forward by one hour for each submission
            with time_machine.travel(base_time + timedelta(hours=i), tick=False):
                pull = await validator.pull_task(create_pull_task(1))
                assert pull.task is not None
                submit = await validator.submit_results(create_submit_result(1, pull.task))

                # Check fidelity
                assert submit.feedback is not None
                current_fidelity = submit.feedback.average_fidelity_score
                expected_fidelity = expected_fidelity * (1 - moving_average_alpha) + moving_average_alpha * 0.8
                assert current_fidelity == pytest.approx(expected_fidelity, abs=0.01)

                # Check observations
                # 1 submission  - 1 observation (start of the observation time)
                # 2 submissions - 2 observations (1 hour passed)
                # 3 submissions - 3 observations (2 hour passed)
                # 4 submissions - 4 observations (3 hour passed)
                # 5 submissions - 4 observations (4 hour passed, first submission dropped)
                expected_observations = min(i + 1, 4)
                expected_reward = expected_observations * default_reward_rating
                assert len(validator.miners[1].observations) == expected_observations

                # Check reward
                reward = submit.feedback.current_miner_reward
                assert reward == pytest.approx(expected_reward, abs=0.01)

                # After 5th submission, verify oldest submissions are dropped
                if i >= 4:
                    oldest_expected_time = base_time.timestamp() + (i - 4) * 60 * 60
                    oldest_actual_time = validator.miners[1].observations[0]
                    assert oldest_actual_time >= oldest_expected_time

    @pytest.mark.asyncio
    async def test_submit_task_after_timeout(
        self, validator: Validator, time_travel: time_machine.travel, reset_validation_server: HTTPServer
    ) -> None:
        """
        Tests behavior when a miner submits results after the task timeout has expired.
        In this case, miner should receive reward as usual.
        """
        # Get task assignment time
        _ = await validator.pull_task(create_pull_task(1))
        assert validator.miners[1].assigned_task is not None
        task = validator.miners[1].assigned_task
        assert validator.miners[1].observations == deque([])
        assert len(validator.miners[1].observations) == 0

        # Move time forward beyond task timeout
        with time_machine.travel(FROZEN_TIME + timedelta(minutes=TASK_TIMEOUT + 1), tick=False):
            # Pull another task in order to cleanup outdated tasks.
            _ = await validator.pull_task(create_pull_task(2))

            # Submit task after timeout and check that it was not submitted.
            submit = await validator.submit_results(create_submit_result(1, task))
            assert submit.feedback is not None
            assert submit.feedback.task_fidelity_score == VALIDATION_SCORE
            assert validator.miners[1].cooldown_until == int(time_machine.time()) + TASK_COOLDOWN - TASK_THROTTLE_PERIOD
            assert len(validator.miners[1].observations) == 1

    @pytest.mark.asyncio
    async def test_submit_task_uncompressed(
        self,
        reset_validation_server: HTTPServer,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        Tests behavior when a miner submits a task with a uncompressed result.
        """
        config.task.organic.send_result_timeout = 0.1
        config.task.organic.assigned_miners_count = 1
        with time_machine.travel(FROZEN_TIME, tick=True):
            async with get_validator_with_available_organic_tasks(
                config=config,
                subtensor=subtensor,
                task_manager=task_manager,
                ratings=ratings,
            ) as validator:
                _ = await validator.pull_task(create_pull_task(1))
                assert validator.miners[1].assigned_task is not None
                task = validator.miners[1].assigned_task
                assert validator.miners[1].observations == deque([])
                assert len(validator.miners[1].observations) == 0
                res = create_submit_result(1, task, full=True)
                res.compression = 0
                submit = await validator.submit_results(res)
                for _ in range(5):
                    await asyncio.sleep(0.1)
                assert submit.feedback is not None
                assert submit.feedback.task_fidelity_score == VALIDATION_SCORE
                assert len(validator.miners[1].observations) == 1
                assert (
                    task.id
                    in cast(
                        GatewayApiMock, validator.task_manager._organic_task_storage._gateway_manager._gateway_api
                    ).results
                )
                assert isinstance(
                    cast(
                        GatewayApiMock,
                        validator.task_manager._organic_task_storage._gateway_manager._gateway_api,
                    ).results[task.id],
                    AssignedMiner,
                )

    @pytest.mark.asyncio
    async def test_submit_task_pyspz_compressed(
        self,
        reset_validation_server: HTTPServer,
        config: bt.config,
        subtensor: bt.MockSubtensor,
        task_manager: TaskManager,
        validator: Validator,
        ratings: DuelRatings,
    ) -> None:
        """
        Tests behavior when a miner submits a task with a pyspz compressed result.
        """
        config.task.organic.send_result_timeout = 0.1
        config.task.organic.assigned_miners_count = 1
        with time_machine.travel(FROZEN_TIME, tick=True):
            async with get_validator_with_available_organic_tasks(
                config=config,
                subtensor=subtensor,
                task_manager=task_manager,
                ratings=ratings,
            ) as validator:
                _ = await validator.pull_task(create_pull_task(1))
                assert validator.miners[1].assigned_task is not None
                task = validator.miners[1].assigned_task
                assert validator.miners[1].observations == deque([])
                assert len(validator.miners[1].observations) == 0
                result = create_submit_result(1, task, full=True)
                input = base64.b64decode(result.results)
                compressed_input = pyspz.compress(input, workers=-1)
                result.results = str(base64.b64encode(compressed_input).decode(encoding="utf-8"))
                submit = await validator.submit_results(result)
                for _ in range(5):
                    await asyncio.sleep(0.1)
                assert submit.feedback is not None
                assert submit.feedback.task_fidelity_score == VALIDATION_SCORE
                assert len(validator.miners[1].observations) == 1
                assert (
                    task.id
                    in cast(
                        GatewayApiMock, validator.task_manager._organic_task_storage._gateway_manager._gateway_api
                    ).results
                )
                assert isinstance(
                    cast(
                        GatewayApiMock,
                        validator.task_manager._organic_task_storage._gateway_manager._gateway_api,
                    ).results[task.id],
                    AssignedMiner,
                )

    @pytest.mark.asyncio
    async def test_submit_task_too_fast(self, validator: Validator, reset_validation_server: HTTPServer) -> None:
        """
        Tests behavior when a miner submits a task too fast.
        In this case, validation timeout should be applied and only one submission should be accepted.
        """
        validation_lock_duration = 0.1
        submit_count = 10
        validator.config.validation.validation_lock_duration = validation_lock_duration
        pull = await validator.pull_task(create_pull_task(1))
        assert pull.task is not None
        with time_machine.travel(FROZEN_TIME, tick=True):
            init_time = int(time_machine.time())
            # Create 10 parallel submit tasks
            submit_tasks = [validator.submit_results(create_submit_result(1, pull.task)) for _ in range(submit_count)]
            await asyncio.gather(*submit_tasks)
            assert validator.miners[1].cooldown_until >= init_time + TASK_COOLDOWN
            assert validator.miners[1].last_submit_time < init_time + TASK_COOLDOWN + TASK_COOLDOWN_PENALTY
            assert len(validator.miners[1].observations) == 1
            assert validator.miners[1].validation_locked_until >= init_time + validation_lock_duration
