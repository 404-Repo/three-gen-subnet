import asyncio
import copy
import typing
from datetime import datetime
from pathlib import Path

import bittensor as bt

from storage_subnet.models import StoredData
from storage_subnet.protocol import StoreUser


class Storage:
    config: bt.config
    """Copy of the original config."""
    subtensor: bt.subtensor
    """The subtensor is the connection to the Bittensor blockchain."""
    metagraph: bt.metagraph
    """The metagraph holds the state of the network, letting us know about other validators and miners."""
    storage_wallet: bt.wallet
    """The wallet used to access the storage network."""
    queue: asyncio.Queue[StoredData]
    """Async-friendly queue to store data."""
    validator_uid: int | None = None
    """Storage sub-net validator to use for storing."""

    disabled: bool = False
    """If misconfigured, storing will be disabled."""

    def __init__(self, config: bt.config) -> None:
        bt.logging.info("Asset storing enabled. Initializing a storage client.")

        self.config: bt.config = copy.deepcopy(config)

        if self.config.storage.testnet:
            self.subtensor = bt.subtensor(network="test")
        else:
            self.subtensor = bt.subtensor(config=self.config)

        self.metagraph = bt.metagraph(netuid=self.config.storage.netuid, network=self.subtensor.network, sync=False)
        self.metagraph.sync(subtensor=self.subtensor)

        bt.logging.debug("Storage metagraph synced.")

        if (
            config.storage.wallet.name == bt.defaults.wallet.name
            and config.storage.wallet.hotkey == bt.defaults.wallet.hotkey
        ):
            self.storage_wallet = bt.wallet(config=self.config)
        else:
            self.storage_wallet = bt.wallet(
                name=config.storage.wallet.name, hotkey=config.storage.wallet.hotkey, path=config.storage.wallet.path
            )

        bt.logging.info(f"Storage wallet: {self.storage_wallet}")

        self.queue = asyncio.Queue(maxsize=self.config.storage.queue_size)

        if self.config.storage.validator.hotkey not in self.metagraph.hotkeys:
            bt.logging.error("Storage validator is not found on storage subnet. Disabling storing.")
            self.disabled = True
        else:
            self.validator_uid = self.metagraph.hotkeys.index(self.config.storage.validator.hotkey)
            bt.logging.info(f"Storage validator found. UID: {self.validator_uid}.")

        # TODO: add periodic resync in case of validator migration

        asyncio.create_task(self.storing_worker())

    async def store(self, data: StoredData) -> None:
        await self.queue.put(data)
        bt.logging.debug(f"Generated data is queued for storing. Queue size: {self.queue.qsize()}")

    async def storing_worker(self) -> None:
        while True:
            data = await self.queue.get()
            await self._store(data)
            self.queue.task_done()

    async def _store(self, data: StoredData) -> None:
        bt.logging.debug(f"Storing data for prompt: {data.prompt}")

        ttl: int = self.config.storage.ttl

        synapse = StoreUser(
            encrypted_data=data.to_base64(),
            encryption_payload="{}",
            ttl=ttl,
        )

        axon = self.metagraph.axons[self.validator_uid]
        dendrite = bt.dendrite(wallet=self.storage_wallet)

        timeout = 120
        cid = None
        for attempt in range(3):
            response = typing.cast(
                StoreUser,
                await dendrite.call(target_axon=axon, synapse=synapse, timeout=timeout, deserialize=False),
            )
            if response.axon.status_code == 200:
                cid = response.data_hash
                break

            bt.logging.debug(
                f"Failed to save generated assets. Attempt #{attempt}. Repeating attempt with bigger timeout."
                f"Status message: {response.dendrite.status_message}."
            )
            timeout += 120
        else:
            bt.logging.error("Failed to save generated assets to the storage subnet")

        if cid is not None:
            bt.logging.info(f"Generated assets saved to: {self.config.storage.validator.hotkey}. CID: {cid}")
            self._update_stored_catalog(cid, data)

    def _update_stored_catalog(self, cid: str, data: StoredData) -> None:
        try:
            path: Path = self.config.neuron.full_path / "catalog.txt"
            line = f"{datetime.utcnow()} {cid} {data.validator} {data.miner}\n"
            with path.open(mode="a", encoding="utf-8") as file:
                file.write(line)
        except Exception:
            bt.logging.exception("Failed to update catalog of stored files.")
