import random as rd

import bittensor as bt
from bittensor_wallet import Keypair

from validator.gateway.gateway import Gateway
from validator.gateway.gateway_api import GatewayApi, GatewayTask
from validator.gateway.gateway_scorer import GatewayScorer
from validator.gateway.http3_client.http3_client import Http3Exception
from validator.task_manager.task_storage.organic_task import GatewayOrganicTask


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

    def _update_gateways(self, *, gateways: list[Gateway]) -> None:
        """Updates the list of gateways."""
        self._gateways = self._gateway_scorer.score(gateways=gateways)
        for gateway in self._gateways:
            bt.logging.trace(f"Gateway updated: {gateway.get_info()}")

    async def get_tasks(self, *, url: str, validator_hotkey: Keypair, task_count: int) -> list[GatewayTask]:
        """Fetches tasks from the gateway."""
        tasks: list[GatewayTask] = []
        try:
            # Reset disabled flag after each try to fetch task
            # and set up it again based on the result.
            for gateway in self._gateways:
                gateway.disabled = False
            result = await self._gateway_api.get_tasks(
                host=url, validator_hotkey=validator_hotkey, task_count=task_count
            )
            tasks = result.tasks
            self._gateways = result.gateways
        except Http3Exception as e:
            bt.logging.error(f"Failed fetching gateway tasks: {e}.")

        # Disable gateway if no tasks were fetched.
        # Either because no real tasks or because of network error.
        if not tasks:
            bt.logging.trace(f"Gateway {url} is disabled for the next iteration: no tasks returned.")
            for gateway in self._gateways:
                if gateway.url == url:
                    gateway.disabled = True
                    break
        self._update_gateways(gateways=self._gateways)
        return tasks

    async def add_result(
        self,
        *,
        validator_hotkey: Keypair,
        task: GatewayOrganicTask,
        score: float | None = None,
        miner_hotkey: str | None = None,
        miner_uid: int | None = None,
        miner_rating: float | None = None,
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
            miner_uid=miner_uid,
            miner_rating=miner_rating,
            asset=asset,
            error=error,
        )
