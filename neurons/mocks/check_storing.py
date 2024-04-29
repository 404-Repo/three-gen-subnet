import argparse
import asyncio
import typing
from pathlib import Path

import bittensor as bt
from storage_subnet.models import StoredData
from storage_subnet.protocol import RetrieveUser, StoreUser


async def main() -> None:
    config = await get_config()
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = bt.metagraph(netuid=config.netuid, network=subtensor.network, sync=False)
    metagraph.sync(subtensor=subtensor)
    dendrite = bt.dendrite(wallet)

    with Path(config.data).open() as f:  # noqa
        data = f.read()

    packet = StoredData(
        assets=data,
        miner="MINER-HOT-KEY",
        validator="VALIDATOR-HOT-KEY",
        prompt=config.prompt,
        submit_time=0,
        signature="b64.encode(sign(f'{submit_time}{prompt}{validator.hotkey}{miner.hotkey}'))",
    )
    # ttl: int = 60 * 60 * 24 * 30
    ttl: int = 60 * 60 * 24

    synapse = StoreUser(
        encrypted_data=packet.to_base64(),
        encryption_payload="{}",
        ttl=ttl,
    )

    timeout = 120
    cid = None
    for attempt in range(3):
        response = typing.cast(
            StoreUser,
            await dendrite.call(target_axon=metagraph.axons[86], synapse=synapse, timeout=timeout, deserialize=False),
        )
        if response.axon.status_code == 200:
            bt.logging.info(f"Saved successfully. CID: {response.data_hash}")
            cid = response.data_hash
            break

        bt.logging.debug(
            f"Attempt #{attempt}. Saving to SN21 failed with: "
            f"{response.axon.status_message}({response.axon.status_code})"
        )
        timeout += 120
    else:
        bt.logging.error("Failed to save generated assets to SN21")

    await asyncio.sleep(30)

    # cid = "bafkreieujw6f76xwuqnwpzfplaheib2tmyo4onuvvzlnvns6jwiaaxfb5q"

    if cid is None:
        return

    dendrite = bt.dendrite(wallet=bt.wallet(name="test.miner"))

    synapse = RetrieveUser(
        data_hash=cid,
    )

    timeout = 300
    response = typing.cast(
        RetrieveUser,
        await dendrite.call(target_axon=metagraph.axons[86], synapse=synapse, timeout=timeout, deserialize=False),
    )

    if response.axon.status_code != 200:
        bt.logging.error(
            f"Retrieval from SN21 failed with: " f"{response.axon.status_message}({response.axon.status_code})"
        )
        return

    if response.encrypted_data is None:
        bt.logging.error("Empty packed retrieved from the storage")
        return

    packet = StoredData.from_base64(response.encrypted_data)
    bt.logging.info(f"Retrieved: {packet}.")


async def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    parser.add_argument("--netuid", type=int, help="Storage subnet netuid", default=22)
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the file with generated results.",
        default="content_pcl.h5",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt used for generation.",
        default="tourmaline tassel earring",
    )
    return bt.config(parser)


if __name__ == "__main__":
    asyncio.run(main())
