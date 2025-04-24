import asyncio
from base64 import b64encode
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import bittensor as bt
import pybase64
import pyspz
import pytest
import time_machine
from bittensor_wallet import Keypair
from bittensor_wallet.mock import get_mock_wallet
from common.miner_license_consent_declaration import MINER_LICENSE_CONSENT_DECLARATION
from common.protocol import PullTask, SubmitResults, Task
from pytest_httpserver import HTTPServer
from validator.config import _build_parser
from validator.gateway.gateway import Gateway
from validator.gateway.gateway_api import GatewayApi, GatewayTask, GetGatewayTasksResult
from validator.gateway.gateway_manager import GatewayManager
from validator.gateway.gateway_scorer import GatewayScorer
from validator.task_manager.task import AssignedMiner, LegacyOrganicTask
from validator.task_manager.task_manager import TaskManager
from validator.task_manager.task_storage.organic_task_storage import OrganicTaskStorage
from validator.task_manager.task_storage.synthetic_asset_storage import SyntheticAssetStorage
from validator.task_manager.task_storage.synthetic_prompt_service import SyntheticPromptService
from validator.task_manager.task_storage.synthetic_task_storage import SyntheticTaskStorage
from validator.validation_service import ValidationService
from validator.validator import Validator

from tests.test_validator.subtensor_mocks import METAGRAPH_INFO, NEURONS, WALLETS


FROZEN_TIME = datetime(year=2025, month=1, day=1)
TASK_THROTTLE_PERIOD = 20
TASK_COOLDOWN = 60
TASK_COOLDOWN_PENALTY = 120
COOLDOWN_VIOLATION_PENALTY = 30
COOLDOWN_VIOLATIONS_THRESHOLD = 10
with Path("tests/resources/robocop with hammer in one hand.spz").open("rb") as f:
    data = f.read()
MINER_RESULT_FULL = str(pybase64.b64encode(pyspz.decompress(data, include_normals=False)).decode(encoding="utf-8"))
MINER_RESULT = "dummy"
VALIDATION_SCORE = 0.99
ASSIGNED_MINERS_COUNT = 4
SECOND_MINER_TIMEOUT = 30
TASK_TIMEOUT = 5 * 60  # 5 minutes
GATEWAY_TASK_QUEUE_SIZE = 5
GATEWAY_TASK_FETCH_INTERVAL = 1
TASK_FETCH_INTERVAL = 2
SYNTHETIC_PROMPT_DELAY = 0
SYNTHETIC_PROMPT_FETCH_INTERVAL = 2
SYNTHETIC_PROMPT_BATCH_SIZE = 10
TASK_QUEUE_SIZE = 100
STRONG_MINER_COUNT = 100
GATEWAY_DOMAIN = "127.0.0.1"
GATEWAY_PORT = 4443
BOOTSTRAP_GATEWAY = f"https://{GATEWAY_DOMAIN}:{GATEWAY_PORT}"
GATEWAY_TASK_COUNT = 100


def create_pull_task(uid: int | None) -> PullTask:
    synapse = PullTask()
    if uid is None:
        synapse.dendrite.hotkey = "unknown"
    else:
        synapse.dendrite.hotkey = WALLETS[uid].hotkey.ss58_address
    synapse.axon.hotkey = WALLETS[0].hotkey.ss58_address
    return synapse


def create_submit_result(uid: int | None, task: Task, full: bool = False) -> SubmitResults:
    if uid is None:
        miner_hotkey = get_mock_wallet().hotkey
    else:
        miner_hotkey = WALLETS[uid].hotkey
    validator_wallet = WALLETS[0]
    signature = b64encode(
        miner_hotkey.sign(
            f"{MINER_LICENSE_CONSENT_DECLARATION}{0}{task.prompt}{validator_wallet.hotkey.ss58_address}{miner_hotkey.ss58_address}"
        )
    )
    if not full:
        synapse = SubmitResults(task=task, results=MINER_RESULT, submit_time=0, signature=signature, compression=2)
    else:
        synapse = SubmitResults(task=task, results=MINER_RESULT_FULL, submit_time=0, signature=signature, compression=0)

    synapse.dendrite.hotkey = miner_hotkey.ss58_address
    synapse.axon.hotkey = WALLETS[0].hotkey.ss58_address
    return synapse


@pytest.fixture
def time_travel() -> Generator[time_machine.travel, None, None]:
    traveller = time_machine.travel(FROZEN_TIME, tick=False)
    traveller.start()
    yield traveller
    traveller.stop()


@pytest.fixture
def synthetic_prompt_server(make_httpserver: HTTPServer) -> HTTPServer:
    make_httpserver.expect_request("/get", method="GET").respond_with_json(
        {"prompts": [f"synthetic_prompt_{idx}" for idx in range(SYNTHETIC_PROMPT_BATCH_SIZE)]}
    )
    return make_httpserver


@pytest.fixture
def storage_server(make_httpserver: HTTPServer) -> HTTPServer:
    make_httpserver.expect_request("/store", method="POST").respond_with_json({"status": "ok"})
    return make_httpserver


@pytest.fixture(scope="function")
def config(validation_server: HTTPServer, synthetic_prompt_server: HTTPServer, storage_server: HTTPServer) -> bt.config:
    parser = _build_parser()
    default_prompts_path = Path(__file__).parent.parent / "resources" / "prompts.txt"
    return bt.config(
        parser,
        args=[
            "--netuid",
            "17",
            "--task.synthetic.default_prompts_path",
            str(default_prompts_path),
            "--generation.task_cooldown",
            f"{TASK_COOLDOWN}",
            "--generation.throttle_period",
            f"{TASK_THROTTLE_PERIOD}",
            "--generation.cooldown_penalty",
            f"{TASK_COOLDOWN_PENALTY}",
            "--generation.cooldown_violation_penalty",
            f"{COOLDOWN_VIOLATION_PENALTY}",
            "--generation.cooldown_violations_threshold",
            f"{COOLDOWN_VIOLATIONS_THRESHOLD}",
            "--public_api.enabled",
            "--validation.endpoints",
            validation_server.url_for(""),
            "--task.synthetic.prompter.endpoint",
            synthetic_prompt_server.url_for(""),
            "--telemetry.disabled",
            "--task.gateway.bootstrap_gateway",
            BOOTSTRAP_GATEWAY,
            "--task.gateway.task_queue_size",
            f"{GATEWAY_TASK_QUEUE_SIZE}",
            "--task.gateway.task_fetch_interval",
            f"{GATEWAY_TASK_FETCH_INTERVAL}",
            "--task.gateway.task_timeout",
            f"{TASK_TIMEOUT}",
            "--task.synthetic.prompter.fetch_interval",
            f"{SYNTHETIC_PROMPT_FETCH_INTERVAL}",
            "--task.synthetic.prompter.delay",
            f"{SYNTHETIC_PROMPT_DELAY}",
            "--task.synthetic.prompter.batch_size",
            f"{SYNTHETIC_PROMPT_BATCH_SIZE}",
            "--public_api.strong_miners_count",
            f"{STRONG_MINER_COUNT}",
            "--task.organic.assigned_miners_count",
            f"{ASSIGNED_MINERS_COUNT}",
            "--storage.enabled",
            "--storage.endpoint_url",
            storage_server.url_for(""),
            "--task.gateway.enabled",
        ],
    )


@pytest.fixture(scope="function")
def validation_server(make_httpserver: HTTPServer) -> HTTPServer:
    make_httpserver.expect_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json(
        {"score": VALIDATION_SCORE}
    )
    return make_httpserver


@pytest.fixture
def reset_validation_server(validation_server: HTTPServer) -> HTTPServer:
    validation_server.clear()
    validation_server.expect_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json(
        {"score": VALIDATION_SCORE}
    )
    return validation_server


# Mock is necessary because GatewayApi uses http3 that is not compatible with pytest-httpserver.
class GatewayApiMock(GatewayApi):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tasks = {str(uuid4()): f"gateway_0_prompt_{idx}" for idx in range(GATEWAY_TASK_COUNT)}
        self.results: dict[str, AssignedMiner | None | str] = {}

    async def get_tasks(self, host: str, validator_hotkey: Keypair, task_count: int) -> GetGatewayTasksResult:
        result: list[GatewayTask] = []
        for _ in range(task_count):
            if not self._tasks:
                break
            task_id, prompt = self._tasks.popitem()
            result.append(GatewayTask(id=task_id, prompt=prompt, gateway_host=host))
        gateways = [
            Gateway(
                node_id=1,
                domain=GATEWAY_DOMAIN,
                ip="5.9.29.227",
                name="node-1-eu",
                http_port=GATEWAY_PORT,
                available_tasks=GATEWAY_TASK_COUNT,
                last_task_acquisition=FROZEN_TIME.timestamp() - 5,
            ),
        ]
        return GetGatewayTasksResult(tasks=result, gateways=gateways)

    async def add_result(
        self,
        *,
        validator_hotkey: Keypair,
        task: GatewayTask,
        score: float | None = None,
        miner_hotkey: str | None = None,
        asset: bytes | None = None,
        error: str | None = None,
    ) -> None:
        if miner_hotkey is not None:
            self.results[task.id] = AssignedMiner(
                hotkey=miner_hotkey,
                score=score if score is not None else VALIDATION_SCORE,
                compressed_result=pybase64.b64encode(asset).decode(encoding="utf-8") if asset is not None else None,
                assign_time=int(FROZEN_TIME.timestamp()),
                submit_time=int(FROZEN_TIME.timestamp()),
                finished=True,
            )
        else:
            self.results[task.id] = error


@pytest.fixture(scope="session")
def subtensor() -> bt.MockSubtensor:
    subtensor_mock = bt.MockSubtensor()
    subtensor_mock.setup()
    subtensor_mock.create_subnet(17)

    subtensor_mock.get_metagraph_info = Mock(return_value=METAGRAPH_INFO)
    subtensor_mock.neurons_lite = Mock(return_value=NEURONS)
    subtensor_mock.query_runtime_api = Mock(return_value=None)
    return subtensor_mock


@pytest.fixture(scope="function")
def task_manager(config: bt.config) -> TaskManager:
    return TaskManager(
        organic_task_storage=OrganicTaskStorage(
            gateway_manager=GatewayManager(
                gateway_scorer=GatewayScorer(),
                gateway_api=GatewayApiMock(),
                gateway_info_server=config.task.gateway.bootstrap_gateway,
            ),
            config=config,
            wallet=WALLETS[0],
        ),
        synthetic_task_storage=SyntheticTaskStorage(
            default_prompts_path=config.task.synthetic.default_prompts_path,
            synthetic_prompt_service=SyntheticPromptService(
                prompt_service_url=config.task.synthetic.prompter.endpoint,
                batch_size=config.task.synthetic.prompter.batch_size,
            ),
            synthetic_asset_storage=SyntheticAssetStorage(
                enabled=config.storage.enabled,
                service_api_key=config.storage.service_api_key,
                endpoint_url=config.storage.endpoint_url,
                validation_score_threshold=config.storage.validation_score_threshold,
            ),
            config=config,
            wallet=WALLETS[0],
        ),
        config=config,
    )


@pytest.fixture
def validator(config: bt.config, subtensor: bt.MockSubtensor, task_manager: TaskManager) -> Validator:
    validation_service = ValidationService(
        endpoints=config.validation.endpoints,
        storage_enabled=config.validation.storage_enabled,
        validation_score_threshold=config.storage.validation_score_threshold,
    )
    return Validator(
        config=config,
        wallet=WALLETS[0],
        subtensor=subtensor,
        task_manager=task_manager,
        validation_service=validation_service,
    )


@asynccontextmanager
async def get_validator_with_available_organic_tasks(
    config: bt.config, subtensor: bt.MockSubtensor, task_manager: TaskManager
) -> AsyncGenerator[Validator, None]:
    """
    Retrurns validator with the same number of organic and legacy tasks.
    Do not use it inside time_machine context with tick=False. Will hand.
    """

    validation_service = ValidationService(
        endpoints=config.validation.endpoints,
        storage_enabled=config.validation.storage_enabled,
        validation_score_threshold=config.storage.validation_score_threshold,
    )

    task_manager = task_manager

    validator = Validator(
        config=config,
        wallet=WALLETS[0],
        subtensor=subtensor,
        task_manager=task_manager,
        validation_service=validation_service,
    )

    # Put legacy organic tasks to the task manager.
    for idx in range(GATEWAY_TASK_QUEUE_SIZE):
        task = LegacyOrganicTask.create_task(
            id=str(uuid4()),
            prompt=f"legacy_organic_task_prompt_{idx}",
        )
        validator.task_manager._organic_task_storage.add_legacy_task(task=task)

    # Put gateway organic tasks to the task manager.
    async_task = asyncio.create_task(validator.task_manager._organic_task_storage.fetch_gateway_tasks_cron())
    while True:
        if validator.task_manager._organic_task_storage._gateway_task_queue:
            async_task.cancel()
            break
        await asyncio.sleep(0.1)

    yield validator
