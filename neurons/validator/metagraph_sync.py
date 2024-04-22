import time
import weakref

import bittensor as bt


class MetagraphSynchronizer:
    def __init__(
        self,
        metagraph: bt.metagraph,
        subtensor: bt.subtensor,
        sync_interval: int,
        log_info_iterval: int,
        strong_miners_count: int,
    ) -> None:
        self._metagraph_ref = weakref.ref(metagraph)
        self._subtensor_ref = weakref.ref(subtensor)
        self._sync_interval = sync_interval
        self._log_info_interval = log_info_iterval
        self._strong_miners_count = strong_miners_count

        self._strong_miners: set[int] = set()
        """Top `strong_miners_count` miners by incentive."""

        self._last_sync_time = 0.0
        """Last metagraph sync time."""
        self._last_info_time = 0.0
        """Last time neuron info was logged."""

    def should_sync(self) -> bool:
        return self._last_sync_time + self._sync_interval <= time.time()

    def sync(self) -> None:
        self._last_sync_time = time.time()

        bt.logging.info("Synchronizing metagraph")
        try:
            metagraph: bt.metagraph = self._metagraph_ref()
            metagraph.sync(subtensor=self._subtensor_ref())
            bt.logging.info("Metagraph synchronized")
        except Exception as e:
            bt.logging.exception(f"Metagraph synchronization failed with {e}")
            return

        # Scores are not reset deliberately, so that new miners can start from the lowest emission but not from zero.

        neurons = [(metagraph.I[uid], uid) for uid in range(int(metagraph.n))]
        neurons.sort(reverse=True)
        self._strong_miners = {uid for i, uid in neurons[: self._strong_miners_count]}

    def log_info(self, uid: int) -> None:
        if self._last_info_time + self._log_info_interval > time.time():
            return

        self._last_info_time = time.time()

        metagraph: bt.metagraph = self._metagraph_ref()

        log = (
            "Validator | "
            f"UID:{uid} | "
            f"Block:{metagraph.block.item()} | "
            f"Stake:{metagraph.S[uid]:.4f} | "
            f"VTrust:{metagraph.Tv[uid]:.4f} | "
            f"Dividends:{metagraph.D[uid]:.6f} | "
            f"Emission:{metagraph.E[uid]:.6f}"
        )
        bt.logging.info(log)

    def is_strong_miner(self, uid: int) -> bool:
        return uid in self._strong_miners
