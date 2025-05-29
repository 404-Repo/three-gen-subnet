import random as rd

import bittensor as bt
from bittensor_wallet import Keypair

from validator.config import config
from validator.gateway.gateway import Gateway
from validator.gateway.gateway_api import GatewayApi, GatewayTask, GetGatewayTasksResult, gateway_api
from validator.gateway.gateway_scorer import GatewayScorer, gateway_scorer
from validator.task_manager.task import GatewayOrganicTask


class GatewayManager:
    """Manages all stuff related with gateways."""

    def __init__(
        self,
        *,
        gateway_scorer: GatewayScorer,
        gateway_api: GatewayApi,
        gateway_info_server: str,
    ) -> None:
        self._gateways: list[Gateway] = []
        self._gateway_scorer: GatewayScorer = gateway_scorer
        self._gateway_api: GatewayApi = gateway_api
        self._gateway_info_server: str = gateway_info_server

    def get_best_gateway(self) -> Gateway | None:
        """Returns the best gateway.
        If all gateways have minimal possible score, returns the random gateway.
        """
        if not self._gateways:
            return None
        gateway = max(self._gateways, key=lambda x: x.score)
        if gateway.score == GatewayScorer.GATEWAY_MIN_SCORE:
            return rd.choice(self._gateways)  # noqa: S311 # nosec: B311
        return gateway

    def update_gateways(self, *, gateways: list[Gateway]) -> None:
        """Updates the list of gateways."""
        self._gateways = self._gateway_scorer.score(gateways=gateways)
        for gateway in self._gateways:
            bt.logging.trace(f"Gateway updated: {gateway.get_info()}")

    async def get_tasks(
        self, *, gateway_host: str, validator_hotkey: Keypair, task_count: int
    ) -> GetGatewayTasksResult:
        """Fetches tasks from the gateway."""
        tasks = await self._gateway_api.get_tasks(
            host=gateway_host, validator_hotkey=validator_hotkey, task_count=task_count
        )
        return tasks

    async def add_result(
        self,
        *,
        validator_hotkey: Keypair,
        task: GatewayOrganicTask,
        score: float | None = None,
        miner_hotkey: str | None = None,
        asset: bytes | None = None,
        error: str | None = None,
    ) -> None:
        """Adds a result to the task."""
        await self._gateway_api.add_result(
            validator_hotkey=validator_hotkey,
            task=GatewayTask(
                id=task.id,
                prompt=task.prompt,
                gateway_host=task.gateway_url,
            ),
            score=score,
            miner_hotkey=miner_hotkey,
            asset=asset,
            error=error,
        )


gateway_manager = GatewayManager(
    gateway_scorer=gateway_scorer,
    gateway_api=gateway_api,
    gateway_info_server=config.task.gateway.bootstrap_gateway,
)
