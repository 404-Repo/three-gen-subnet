import json
import time
from enum import Enum
from typing import Self

import bittensor as bt
import pybase64
from bittensor_wallet import Keypair
from pydantic import BaseModel, model_validator

from validator.gateway.gateway import Gateway
from validator.gateway.http3_client.http3_client import Http3Client


class GatewayRoute(Enum):
    """Routes for gateway nodes"""

    GET_TASKS = "/get_tasks"
    ADD_RESULT = "/add_result"


class GatewayTask(BaseModel):
    """Task from gateway"""

    id: str
    prompt: str | None = None
    image: str | None = None
    gateway_host: str

    @model_validator(mode="after")
    def check_prompt_xor_image(self) -> Self:
        if self.prompt is None and self.image is None:
            raise ValueError("Either prompt or image must be provided")
        if self.prompt is not None and self.image is not None:
            raise ValueError("Only one of prompt or image should be provided, not both")
        return self


class GetGatewayTasksResult(BaseModel):
    tasks: list[GatewayTask]
    gateways: list[Gateway]


class GatewayApi:
    """API client for interacting with gateway nodes"""

    _DEFAULT_LATENCY: float = 100.0

    def __init__(self) -> None:  # TODO: make this configurable
        self._http3_client = Http3Client()
        self._host_to_latency: dict[str, float] = {}

    async def get_tasks(self, *, host: str, validator_hotkey: Keypair, task_count: int) -> GetGatewayTasksResult:
        """Gets a list of tasks from the gateway."""
        url = self._construct_url(host=host, route=GatewayRoute.GET_TASKS)
        timestamp = int(time.time())
        msg = f"404_GATEWAY_{timestamp}"
        signature = pybase64.b64encode(validator_hotkey.sign(msg)).decode(encoding="utf-8")

        payload = {
            "validator_hotkey": validator_hotkey.ss58_address,
            "timestamp": msg,
            "signature": signature,
            "requested_task_count": task_count,
        }
        response = await self._http3_client.post(url=url, payload=payload)
        self._calc_exp_avg_latency(host=host, latency=response.latency)
        data = json.loads(response.data)
        tasks: list[GatewayTask] = []
        for task in data["tasks"]:
            task["gateway_host"] = host
            tasks.append(GatewayTask.model_validate(task))
        gateways: list[Gateway] = []
        for gateway in data["gateways"]:
            gateway = Gateway.model_validate(gateway)
            gateway.latency = self._host_to_latency.get(gateway.url, self._DEFAULT_LATENCY)
            gateways.append(gateway)
        return GetGatewayTasksResult(tasks=tasks, gateways=gateways)

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
        """Adds a result to the task."""
        url = self._construct_url(host=task.gateway_host, route=GatewayRoute.ADD_RESULT)
        timestamp = int(time.time())
        msg = f"404_GATEWAY_{timestamp}"
        signature = pybase64.b64encode(validator_hotkey.sign(msg)).decode(encoding="utf-8")
        bt.logging.debug(f"GatewayApi: adding result to {url}")
        payload: dict[str, int | float | str | bytes | None] = {
            "id": task.id,
            "signature": signature,
            "timestamp": msg,
            "validator_hotkey": validator_hotkey.ss58_address,
        }
        if miner_hotkey is not None:
            payload["status"] = "success"
            payload["miner_hotkey"] = miner_hotkey
            payload["asset"] = asset
            payload["score"] = score
            payload["miner_uid"] = miner_uid
            payload["miner_rating"] = miner_rating
        else:
            payload["status"] = "failure"
            payload["reason"] = error
        response = await self._http3_client.post_form_data(url=url, data=payload)
        self._calc_exp_avg_latency(host=task.gateway_host, latency=response.latency)

    def _construct_url(self, *, host: str, route: GatewayRoute) -> str:
        return f"{host}{route.value}"

    def _calc_exp_avg_latency(self, *, host: str, latency: float) -> None:
        if host not in self._host_to_latency:
            self._host_to_latency[host] = latency
        else:
            self._host_to_latency[host] = 0.8 * self._host_to_latency[host] + 0.2 * latency
