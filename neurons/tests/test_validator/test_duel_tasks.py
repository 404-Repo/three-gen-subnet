import asyncio
import json
from datetime import timedelta
import bittensor as bt

import pybase64
from pytest_httpserver import HTTPServer

import pytest
import time_machine

from tests.test_validator.conftest import FROZEN_TIME, create_pull_task, create_submit_result
from validator.duels.duels_task_storage import GARBAGE_COLLECTION_CYCLE
from validator.validator import Validator


async def pull_and_submit(validator: Validator, miner_uid: int, wallets: list[bt.Wallet]) -> None:
    pull = await validator.pull_task(create_pull_task(miner_uid, wallets))
    assert pull.task is not None
    synapse = create_submit_result(miner_uid, pull.task, wallets)
    await validator.submit_results(synapse)


async def mark_miners_as_active(
    time_travel: time_machine.Coordinates, validator: Validator, miners_uids: list[int], wallets: list[bt.Wallet]
) -> None:
    duel_start = FROZEN_TIME + timedelta(seconds=validator.config.duels.start_delay)
    one_task_before = duel_start - timedelta(seconds=validator.config.generation.task_cooldown + 1)

    time_travel.move_to(one_task_before, tick=False)
    for miner_id in miners_uids:
        await pull_and_submit(validator, miner_id, wallets)


async def pull_and_submit_empty_results(validator: Validator, miner_uid: int, wallets: list[bt.Wallet]) -> None:
    pull = await validator.pull_task(create_pull_task(miner_uid, wallets))
    assert pull.task is not None
    synapse = create_submit_result(miner_uid, pull.task, wallets)
    synapse.results = ""
    await validator.submit_results(synapse)


class TestDuelTasks:

    @pytest.mark.asyncio
    async def test_full_cycle_success(
        self,
        time_travel: time_machine.Coordinates,
        validation_server: HTTPServer,
        duel_save_server: HTTPServer,
        validator: Validator,
        wallets: list[bt.Wallet],
    ) -> None:
        # Notes:
        #   - before the duel starts, miners should pull a task to mark themselves as active.
        #   - frozen time is datetime(year=2025, month=1, day=1), a fixed known value,
        #     duel start delay is fixed known value,
        #     prompt is always "Monkey" in tests;
        #     hash function will always return 251 and 248 miners for the first duel.
        #   - check the final duel results with duel_save_saver.

        await mark_miners_as_active(time_travel, validator, miners_uids=[251, 248], wallets=wallets)

        # Time to start duels
        duel_start = FROZEN_TIME + timedelta(seconds=validator.config.duels.start_delay)
        time_travel.move_to(duel_start, tick=True)

        pull = await validator.pull_task(create_pull_task(251, wallets))
        assert pull.task is not None

        assert len(validator.task_manager._duel_task_storage._pending_duels) == 1
        assert len(validator.task_manager._duel_task_storage._pending_duels_by_miner[251]) == 0
        assert len(validator.task_manager._duel_task_storage._pending_duels_by_miner[248]) == 1
        assert len(validator.task_manager._duel_task_storage._started_duels) == 1

        await validator.submit_results(create_submit_result(251, pull.task, wallets))

        await asyncio.sleep(0)  # sleep to switch to the taskmanager submit

        pull = await validator.pull_task(create_pull_task(248, wallets))
        assert pull.task is not None

        assert len(validator.task_manager._duel_task_storage._pending_duels) == 0
        assert len(validator.task_manager._duel_task_storage._pending_duels_by_miner[248]) == 0
        assert len(validator.task_manager._duel_task_storage._started_duels) == 1

        await validator.submit_results(create_submit_result(248, pull.task, wallets))
        await asyncio.sleep(0)  # sleep to switch to the taskmanager submit

        assert len(validator.task_manager._duel_task_storage._started_duels) == 0
        assert validator.task_manager._duel_task_storage._pending_judgment_queue.qsize() == 1

        await validator.task_manager._duel_task_storage.start_judging_duels()
        await asyncio.sleep(0.1)  # sleep to switch to the judge runner

        assert validator.task_manager._duel_task_storage._pending_judgment_queue.qsize() == 0
        assert len(validator.task_manager._duel_task_storage._started_duels) == 0
        await asyncio.sleep(0.1)  # sleep to switch to the duel publication

        duel_save_server.check_assertions()
        hotkey = wallets[0].hotkey
        message_to_sign = f"duel_saver{1735691400}{hotkey.ss58_address}"
        expected_duel_result = {
            "validator_hotkey": hotkey.ss58_address,
            "signature": "",
            "finish_time": 0,
            "timestamp_nonce": 1735691400,
            "prompt": "Monkey",
            "task_type": "text",
            "winner": 1,
            "explanation": "issues",
            "left": {
                "hotkey": wallets[248].hotkey.ss58_address,
                "coldkey": wallets[248].coldkey.ss58_address,
                "glicko_before": 1500.0,
                "glicko_after": 1668.95123159519,
                "glicko_rd": 296.19808829689504,
                "glicko_vol": 0.2,
            },
            "right": {
                "hotkey": wallets[251].hotkey.ss58_address,
                "coldkey": wallets[251].coldkey.ss58_address,
                "glicko_before": 1500.0,
                "glicko_after": 1331.04876840481,
                "glicko_rd": 296.19808829689504,
                "glicko_vol": 0.2,
            },
        }
        publish_request = [req for req, res in duel_save_server.log if req.path == "/api/save_duel/"][0]
        saved_results = json.loads(publish_request.form["results"])
        saved_results["finish_time"] = 0
        signature = saved_results["signature"]
        saved_results["signature"] = ""
        assert hotkey.verify(message_to_sign, pybase64.b64decode(signature.encode("utf-8")))
        assert saved_results == expected_duel_result

        validator.task_manager._duel_task_storage.stop_judging_duels()

    @pytest.mark.asyncio
    async def test_one_miner_failed(
        self,
        time_travel: time_machine.Coordinates,
        validation_server: HTTPServer,
        duel_save_server: HTTPServer,
        validator: Validator,
        wallets: list[bt.Wallet],
    ) -> None:
        await mark_miners_as_active(time_travel, validator, miners_uids=[251, 248], wallets=wallets)

        # Time to start duels
        duel_start = FROZEN_TIME + timedelta(seconds=validator.config.duels.start_delay)
        time_travel.move_to(duel_start, tick=True)

        await pull_and_submit_empty_results(validator, miner_uid=251, wallets=wallets)
        await asyncio.sleep(0)  # sleep to switch to the taskmanager submit

        pull = await validator.pull_task(create_pull_task(248, wallets))
        assert pull.task is not None
        await validator.submit_results(create_submit_result(248, pull.task, wallets))
        await asyncio.sleep(0)  # sleep to switch to the taskmanager submit

        assert len(validator.task_manager._duel_task_storage._started_duels) == 0
        assert validator.task_manager._duel_task_storage._pending_judgment_queue.qsize() == 0

        await asyncio.sleep(0.1)  # sleep to switch to the duel publication

        duel_save_server.check_assertions()
        hotkey = wallets[0].hotkey
        expected_duel_result = {
            "validator_hotkey": hotkey.ss58_address,
            "signature": "",
            "finish_time": 0,
            "timestamp_nonce": 1735691400,
            "prompt": "Monkey",
            "task_type": "text",
            "winner": 1,
            "explanation": "One or both miners failed to generate",
            "left": {
                "hotkey": wallets[248].hotkey.ss58_address,
                "coldkey": wallets[248].coldkey.ss58_address,
                "glicko_before": 1500.0,
                "glicko_after": 1668.95123159519,
                "glicko_rd": 296.19808829689504,
                "glicko_vol": 0.2,
            },
            "right": {
                "hotkey": wallets[251].hotkey.ss58_address,
                "coldkey": wallets[251].coldkey.ss58_address,
                "glicko_before": 1500.0,
                "glicko_after": 1331.04876840481,
                "glicko_rd": 296.19808829689504,
                "glicko_vol": 0.2,
            },
        }
        publish_request = [req for req, res in duel_save_server.log if req.path == "/api/save_duel/"][0]
        saved_results = json.loads(publish_request.form["results"])
        saved_results["finish_time"] = 0
        saved_results["signature"] = ""
        assert saved_results == expected_duel_result

    @pytest.mark.asyncio
    async def test_both_miner_failed(
        self,
        time_travel: time_machine.Coordinates,
        validation_server: HTTPServer,
        duel_save_server: HTTPServer,
        validator: Validator,
        wallets: list[bt.Wallet],
    ) -> None:
        await mark_miners_as_active(time_travel, validator, miners_uids=[251, 248], wallets=wallets)

        # Time to start duels
        duel_start = FROZEN_TIME + timedelta(seconds=validator.config.duels.start_delay)
        time_travel.move_to(duel_start, tick=True)
        await pull_and_submit_empty_results(validator, miner_uid=251, wallets=wallets)
        await asyncio.sleep(0)  # sleep to switch to the taskmanager submit

        await pull_and_submit_empty_results(validator, miner_uid=248, wallets=wallets)
        await asyncio.sleep(0)  # sleep to switch to the taskmanager submit

        assert len(validator.task_manager._duel_task_storage._started_duels) == 0
        assert validator.task_manager._duel_task_storage._pending_judgment_queue.qsize() == 0

        await asyncio.sleep(0.1)  # sleep to switch to the duel publication

        duel_save_server.check_assertions()
        hotkey = wallets[0].hotkey
        expected_duel_result = {
            "validator_hotkey": hotkey.ss58_address,
            "signature": "",
            "finish_time": 0,
            "timestamp_nonce": 1735691400,
            "prompt": "Monkey",
            "task_type": "text",
            "winner": 0,
            "explanation": "One or both miners failed to generate",
            "left": {
                "hotkey": wallets[248].hotkey.ss58_address,
                "coldkey": wallets[248].coldkey.ss58_address,
                "glicko_before": 1500.0,
                "glicko_after": 1500.0,
                "glicko_rd": 296.17900101404155,
                "glicko_vol": 0.2,
            },
            "right": {
                "hotkey": wallets[251].hotkey.ss58_address,
                "coldkey": wallets[251].coldkey.ss58_address,
                "glicko_before": 1500.0,
                "glicko_after": 1500.0,
                "glicko_rd": 296.17900101404155,
                "glicko_vol": 0.2,
            },
        }
        publish_request = [req for req, res in duel_save_server.log if req.path == "/api/save_duel/"][0]
        saved_results = json.loads(publish_request.form["results"])
        saved_results["finish_time"] = 0
        saved_results["signature"] = ""
        assert saved_results == expected_duel_result

    @pytest.mark.asyncio
    async def test_no_duel_task(
        self,
        time_travel: time_machine.Coordinates,
        validation_server: HTTPServer,
        duel_save_server: HTTPServer,
        validator: Validator,
        wallets: list[bt.Wallet],
    ) -> None:
        duel_start = FROZEN_TIME + timedelta(seconds=validator.config.duels.start_delay)
        time_travel.move_to(duel_start, tick=True)
        pull = await validator.pull_task(create_pull_task(1, wallets))
        assert pull.task is not None
        assert len(validator.task_manager._duel_task_storage._started_duels) == 0

    @pytest.mark.asyncio
    async def test_duel_recycling(
        self,
        time_travel: time_machine.Coordinates,
        validation_server: HTTPServer,
        duel_save_server: HTTPServer,
        validator: Validator,
        wallets: list[bt.Wallet],
    ) -> None:
        await mark_miners_as_active(time_travel, validator, miners_uids=[251, 248], wallets=wallets)

        duel_start = FROZEN_TIME + timedelta(seconds=validator.config.duels.start_delay)
        time_travel.move_to(duel_start, tick=True)
        await pull_and_submit(validator, miner_uid=251, wallets=wallets)
        await asyncio.sleep(0)  # sleep to switch to the taskmanager submit

        assert len(validator.task_manager._duel_task_storage._pending_duels) == 1
        assert len(validator.task_manager._duel_task_storage._pending_duels_by_miner[251]) == 0
        assert len(validator.task_manager._duel_task_storage._pending_duels_by_miner[248]) == 1
        assert len(validator.task_manager._duel_task_storage._started_duels) == 1

        # Miner 248 never finishes his task

        duel_recycle_time = FROZEN_TIME + timedelta(
            seconds=validator.config.duels.start_delay + 2 * GARBAGE_COLLECTION_CYCLE  # Double cycle time
        )
        time_travel.move_to(duel_recycle_time, tick=True)
        await validator.task_manager._duel_task_storage._collect_garbage()

        assert len(validator.task_manager._duel_task_storage._pending_duels) == 0
        assert len(validator.task_manager._duel_task_storage._pending_duels_by_miner[251]) == 0
        assert len(validator.task_manager._duel_task_storage._pending_duels_by_miner[248]) == 0
        assert len(validator.task_manager._duel_task_storage._started_duels) == 0
