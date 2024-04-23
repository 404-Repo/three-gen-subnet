import time
import typing
import weakref

import bittensor as bt


class MetagraphSynchronizer:
    """Encapsulates metagraph syncs."""

    def __init__(
        self, metagraph: bt.metagraph, subtensor: bt.subtensor, sync_interval: int, log_info_iterval: int
    ) -> None:
        self._metagraph_ref = weakref.ref(metagraph)
        self._subtensor_ref = weakref.ref(subtensor)
        self._sync_interval = sync_interval
        self._log_info_interval = log_info_iterval

        self._last_sync_time = 0.0
        """Last metagraph sync time."""
        self._last_info_time = 0.0
        """Last time neuron info was logged."""

    def sync(self) -> None:
        if self._last_sync_time + self._sync_interval > time.time():
            return

        self._last_sync_time = time.time()

        bt.logging.info("Synchronizing metagraph")
        try:
            typing.cast(bt.metagraph, self._metagraph_ref()).sync(subtensor=self._subtensor_ref())
            bt.logging.info("Metagraph synchronized")
        except Exception as e:
            bt.logging.exception(f"Metagraph synchronization failed with {e}")

    def log_info(self, uid: int) -> None:
        if self._last_info_time + self._log_info_interval > time.time():
            return

        self._last_info_time = time.time()

        metagraph: bt.metagraph = self._metagraph_ref()

        log = (
            "Miner | "
            f"UID:{uid} | "
            f"Block:{metagraph.block.item()} | "
            f"Stake:{metagraph.S[uid]:.4f} | "
            f"Trust:{metagraph.T[uid]:.4f} | "
            f"Incentive:{metagraph.I[uid]:.6f} | "
            f"Emission:{metagraph.E[uid]:.6f}"
        )
        bt.logging.info(log)
