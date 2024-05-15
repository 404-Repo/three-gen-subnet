import bittensor as bt

from common.protocol import Version


NEURONS_VERSION = Version(major=0, minor=3, patch=0)


def compare_versions(miner: Version, validator: Version, validator_hotkey: str) -> None:
    if miner.major > validator.major:
        bt.logging.warning(
            f"Validator {validator_hotkey} is running an outdated version. "
            f"Current version: {miner}. Validator version: {validator}"
        )
        return

    if miner.major < validator.major:
        bt.logging.warning(
            f"Your miner is running an outdated version. "
            f"Current version: {validator}. Your version: {validator}."
            f"Please, update your miner"
        )
        return

    if miner.minor > validator.minor:
        bt.logging.warning(
            f"Validator {validator_hotkey} is running an outdated version. "
            f"Current version: {miner}. Validator version: {validator}"
        )
        return

    if miner.minor < validator.minor:
        bt.logging.warning(
            f"Your miner is running an outdated version. "
            f"Current version: {validator}. Your version: {validator}."
            f"Please, update your miner"
        )
        return

    if miner.patch < validator.patch:
        bt.logging.info(
            f"There is a new patch version available. Current version: {validator}. Your version: {validator}."
        )
        return
