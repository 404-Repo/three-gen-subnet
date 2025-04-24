from base64 import b64encode

from bittensor_wallet.mock import get_mock_wallet
from common.miner_license_consent_declaration import MINER_LICENSE_CONSENT_DECLARATION
from common.protocol import PullTask, SubmitResults, Task
from tests.test_validator.conftest import MINER_RESULT, MINER_RESULT_FULL
from tests.test_validator.subtensor_mocks import WALLETS


def create_pull_task(uid: int | None) -> PullTask:
    synapse = PullTask()
    if uid is None:
        synapse.dendrite.hotkey = "unknown"
    else:
        synapse.dendrite.hotkey = WALLETS[uid].hotkey.ss58_address
    synapse.axon.hotkey = WALLETS[0].hotkey.ss58_address
    return synapse


def create_submit_result(uid: int | None, task: Task, full: bool = False) -> SubmitResults:
    if uid is None:
        miner_hotkey = get_mock_wallet().hotkey
    else:
        miner_hotkey = WALLETS[uid].hotkey
    validator_wallet = WALLETS[0]
    signature = b64encode(
        miner_hotkey.sign(
            f"{MINER_LICENSE_CONSENT_DECLARATION}{0}{task.prompt}{validator_wallet.hotkey.ss58_address}{miner_hotkey.ss58_address}"
        )
    )
    if not full:
        synapse = SubmitResults(task=task, results=MINER_RESULT, submit_time=0, signature=signature, compression=2)
    else:
        synapse = SubmitResults(task=task, results=MINER_RESULT_FULL, submit_time=0, signature=signature, compression=2)

    synapse.dendrite.hotkey = miner_hotkey.ss58_address
    synapse.axon.hotkey = WALLETS[0].hotkey.ss58_address
    return synapse
