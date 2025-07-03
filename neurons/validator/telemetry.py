import asyncio
import time
import weakref
from typing import Any

import aiohttp
import bittensor as bt
import pybase64

from validator.miner_data import MinerData


class Telemetry:
    def __init__(self, miners: list[MinerData], wallet: bt.wallet, metagraph: bt.metagraph, config: bt.config) -> None:
        """
        Handles collection and submission of telemetry data from validator operations.

        This class manages the gathering and reporting of performance metrics for tasks processed
        by miners in the network. It tracks both per-miner statistics and individual task metrics.

        Metrics are periodically sent to a push gateway server with cryptographic authentication
        using the validator's hotkey for signature verification.
        """
        self.disabled = config.telemetry.disabled
        """Flag that indicates if telemetry is disabled."""
        self.push_gateway = config.telemetry.push_gateway
        """Push gateway server address."""
        self.miners = miners
        """List of the miners."""
        self.wallet = wallet
        """Wallet of the validator."""
        self.validator_hotkey = wallet.hotkey.ss58_address
        """Hotkey of the validator."""
        self.metagraph_ref = weakref.ref(metagraph)
        """Weak reference to the metagraph."""
        self.accumulated_metrics: list[dict[str, Any]] = []
        """List of the accumulated metrics."""

    async def start(self) -> None:
        if self.disabled:
            return
        asyncio.create_task(self._schedule_miners_metrics())
        asyncio.create_task(self._schedule_task_metrics())

    def add_task_metrics(
        self,
        miner_hotkey: str,
        miner_coldkey: str,
        score: float,
        delivery_time: float,
        size: int,
    ) -> None:
        if self.disabled:
            return

        self.accumulated_metrics.append(
            {
                "score": score,
                "delivery_time": delivery_time,
                "size": size,
                "compression": 2,
                "timestamp": int(time.time()),
                "labels": {
                    "miner_hotkey": miner_hotkey,
                    "miner_coldkey": miner_coldkey,
                    "task_type": "text-to-3d",
                },
            }
        )

    async def _schedule_miners_metrics(self) -> None:
        while True:
            await asyncio.sleep(120)
            asyncio.create_task(self._send_miners_metrics())

    async def _schedule_task_metrics(self) -> None:
        while True:
            await asyncio.sleep(5)
            asyncio.create_task(self._send_task_metrics())

    async def _send_miners_metrics(self) -> None:
        metagraph = self.metagraph_ref()
        if metagraph is None:
            return

        metrics = [
            {
                "score": miner.fidelity_score,
                "results_amount": len(miner.observations),
                "labels": {
                    "miner_hotkey": miner.hotkey,
                    "miner_coldkey": metagraph.axons[miner.uid].coldkey,
                    "task_type": "text-to-3d",
                },
            }
            for miner in self.miners
        ]
        await self._send_metrics(metrics)

    async def _send_task_metrics(self) -> None:
        if not self.accumulated_metrics:
            return

        metrics = self.accumulated_metrics
        self.accumulated_metrics = []
        await self._send_metrics(metrics)

    async def _send_metrics(self, metrics: list[dict[str, Any]]) -> None:
        nonce, signature = await self._sign()
        payload = {
            "hotkey": self.validator_hotkey,
            "nonce": nonce,
            "signature": signature,
            "metrics": metrics,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.push_gateway, json=payload, headers={"Content-Type": "application/json"}
            ) as response:
                if not response.ok:
                    bt.logging.warning(f"Failed to send metrics to {self.push_gateway} with: {response.reason}")

    async def _sign(self) -> tuple[int, str]:
        nonce = time.time_ns()
        message = f"{nonce}:{self.validator_hotkey}"
        signature = pybase64.b64encode(self.wallet.hotkey.sign(message)).decode(encoding="utf-8")
        return nonce, signature
