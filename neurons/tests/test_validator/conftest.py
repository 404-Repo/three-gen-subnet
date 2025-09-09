from base64 import b64encode
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any
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
from validator.duels.duels_task_storage import DuelsTaskStorage
from validator.duels.ratings import DuelRatings
from validator.gateway.gateway import Gateway
from validator.gateway.gateway_api import GatewayApi, GatewayTask, GetGatewayTasksResult
from validator.gateway.gateway_manager import GatewayManager
from validator.gateway.gateway_scorer import GatewayScorer
from validator.task_manager.task_manager import TaskManager
from validator.task_manager.task_storage.organic_task_storage import (
    OrganicTaskStorage,
)
from validator.task_manager.task_storage.organic_task import AssignedMiner, LegacyOrganicTask
from validator.task_manager.task_storage.synthetic_asset_storage import SyntheticAssetStorage
from validator.task_manager.task_storage.synthetic_prompt_service import SyntheticPromptService
from validator.task_manager.task_storage.synthetic_task_storage import SyntheticTaskStorage
from validator.validation_service import ValidationService
from validator.validator import Validator
from tests.test_validator.subtensor_mock import MockSubtensor


FROZEN_TIME = datetime(year=2025, month=1, day=1)
TASK_THROTTLE_PERIOD = 20
TASK_COOLDOWN = 60
TASK_COOLDOWN_PENALTY = 120
COOLDOWN_VIOLATION_PENALTY = 30
COOLDOWN_VIOLATIONS_THRESHOLD = 10
with Path("tests/resources/robocop with hammer in one hand.spz").resolve().open("rb") as f:
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
DUELS_START_DELAY = 1800


@pytest.fixture(scope="session")
def netuid() -> int:
    return 17


def create_pull_task(uid: int | None, wallets: list[bt.Wallet]) -> PullTask:
    synapse = PullTask()
    if uid is None:
        synapse.dendrite.hotkey = "unknown"
    else:
        synapse.dendrite.hotkey = wallets[uid].hotkey.ss58_address
    synapse.axon.hotkey = wallets[0].hotkey.ss58_address
    return synapse


def create_submit_result(uid: int | None, task: Task, wallets: list[bt.Wallet], full: bool = False) -> SubmitResults:
    if uid is None:
        miner_hotkey = get_mock_wallet().hotkey
    else:
        miner_hotkey = wallets[uid].hotkey
    validator_wallet = wallets[0]
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
    synapse.axon.hotkey = wallets[0].hotkey.ss58_address
    return synapse


@pytest.fixture
def time_travel() -> Generator[time_machine.Coordinates, None, None]:
    traveller = time_machine.travel(FROZEN_TIME, tick=False)
    yield traveller.start()
    traveller.stop()


@pytest.fixture(scope="session")
def wallets() -> list[bt.Wallet]:
    return [get_mock_wallet() for _ in range(200)]


@pytest.fixture(scope="function")
def config(
    validation_server: HTTPServer,
    synthetic_prompt_server: HTTPServer,
    storage_server: HTTPServer,
    judge_server: HTTPServer,
    duel_save_server: HTTPServer,
    netuid: int,
) -> bt.config:
    parser = _build_parser()
    default_prompts_path = Path(__file__).parent.parent / "resources" / "prompts.txt"
    return bt.config(
        parser,
        args=[
            "--netuid",
            f"{netuid}",
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
            "--task.gateway.enabled",
            "--duels.start_delay",
            f"{DUELS_START_DELAY}",
            "--duels.judge_endpoint",
            judge_server.url_for("/v1/"),
            "--duels.duel_saver_endpoint",
            duel_save_server.url_for("/api/save_duel/"),
        ],
    )


@pytest.fixture(scope="function")
def validation_server(make_httpserver: HTTPServer) -> HTTPServer:
    make_httpserver.expect_request("/validate_txt_to_3d_ply/", method="POST").respond_with_json(
        {"score": VALIDATION_SCORE, "grid_preview": pybase64.b64encode("grid preview".encode()).decode()}
    )
    make_httpserver.expect_request("/render_duel_view/", method="POST").respond_with_data(b"render.png")
    return make_httpserver


@pytest.fixture(scope="function")
def judge_server(make_httpserver: HTTPServer) -> HTTPServer:
    make_httpserver.expect_request("/v1/chat/completions", method="POST").respond_with_json(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": '{"issues": "issues", "worst": 2}',
                    },
                }
            ]
        }
    )
    return make_httpserver


@pytest.fixture(scope="function")
def duel_save_server(make_httpserver: HTTPServer) -> HTTPServer:
    make_httpserver.expect_request("/api/save_duel/", method="POST").respond_with_json({"status": "ok"})
    return make_httpserver


@pytest.fixture(scope="function")
def synthetic_prompt_server(make_httpserver: HTTPServer) -> HTTPServer:
    make_httpserver.expect_request("/get", method="GET").respond_with_json(
        {"prompts": [f"synthetic_prompt_{idx}" for idx in range(SYNTHETIC_PROMPT_BATCH_SIZE)]}
    )
    return make_httpserver


@pytest.fixture(scope="function")
def storage_server(make_httpserver: HTTPServer) -> HTTPServer:
    make_httpserver.expect_request("/store", method="POST").respond_with_json({"status": "ok"})
    return make_httpserver


@pytest.fixture(scope="function", autouse=True)
def clear_http_server(make_httpserver: HTTPServer) -> None:
    make_httpserver.clear()


# Mock is necessary because GatewayApi uses http3 that is not compatible with pytest-httpserver.
class GatewayApiMock(GatewayApi):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
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
        miner_uid: int | None = None,
        miner_rating: float | None = None,
        asset: bytes | None = None,
        error: str | None = None,
    ) -> None:
        if miner_hotkey is not None:
            assert miner_uid is not None
            assert miner_rating is not None
            self.results[task.id] = AssignedMiner(
                uid=miner_uid,
                hotkey=miner_hotkey,
                rating=miner_rating,
                score=score if score is not None else VALIDATION_SCORE,
                compressed_result=pybase64.b64encode(asset).decode(encoding="utf-8") if asset is not None else None,
                assign_time=int(FROZEN_TIME.timestamp()),
                submit_time=int(FROZEN_TIME.timestamp()),
                finished=True,
            )
        else:
            self.results[task.id] = error


@pytest.fixture(scope="session")
def subtensor(netuid: int, wallets: list[bt.Wallet]) -> MockSubtensor:
    subtensor_mock = MockSubtensor()
    subtensor_mock.setup()
    subtensor_mock.create_subnet(netuid)

    for wallet in wallets:
        # It is important to use float number for stake and balance.
        # int is considered as rao. It is divided internally by 1e9.
        subtensor_mock.force_register_neuron(
            netuid=netuid,
            hotkey=wallet.hotkey.ss58_address,
            coldkey=wallet.coldkey.ss58_address,
            stake=bt.Balance(10000.0),
            balance=bt.Balance(10000.0),
        )

    metagraph = bt.Metagraph(netuid=netuid, network=subtensor_mock.network, sync=False)
    metagraph.sync(subtensor=subtensor_mock)
    return subtensor_mock


@pytest.fixture(scope="function")
def ratings() -> DuelRatings:
    return DuelRatings()


@pytest.fixture(scope="function")
def task_manager(config: bt.config, ratings: DuelRatings, wallets: list[bt.Wallet]) -> TaskManager:
    synthetic_task_storage = SyntheticTaskStorage(
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
        wallet=wallets[0],
    )
    validation_service = ValidationService(
        endpoints=config.validation.endpoints,
        validation_score_threshold=config.storage.validation_score_threshold,
    )
    return TaskManager(
        organic_task_storage=OrganicTaskStorage(
            gateway_manager=GatewayManager(
                gateway_scorer=GatewayScorer(),
                gateway_api=GatewayApiMock(),
                gateway_info_server=config.task.gateway.bootstrap_gateway,
            ),
            config=config,
            wallet=wallets[0],
            ratings=ratings,
        ),
        synthetic_task_storage=synthetic_task_storage,
        duel_task_storage=DuelsTaskStorage(
            config=config,
            wallet=wallets[0],
            synthetic_task_storage=synthetic_task_storage,
            validation_service=validation_service,
            ratings=ratings,
        ),
        config=config,
    )


@pytest.fixture
def validator(
    config: bt.config,
    subtensor: bt.MockSubtensor,
    task_manager: TaskManager,
    ratings: DuelRatings,
    wallets: list[bt.Wallet],
) -> Validator:
    validation_service = ValidationService(
        endpoints=config.validation.endpoints,
        validation_score_threshold=config.storage.validation_score_threshold,
    )
    return Validator(
        config=config,
        wallet=wallets[0],
        subtensor=subtensor,
        task_manager=task_manager,
        validation_service=validation_service,
        ratings=ratings,
    )
