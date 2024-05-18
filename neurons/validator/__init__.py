import asyncio
import base64
import copy
import time
from typing import Tuple  # noqa: UP035

import bittensor as bt
import torch
from api import Generate, StatusCheck
from auto_updater import AutoUpdater
from bittensor.utils import weight_utils
from common import create_neuron_dir, owner
from common.protocol import Feedback, GetVersion, PullTask, SubmitResults, Task
from common.version import NEURONS_VERSION
from pydantic import BaseModel
from storage_subnet import Storage, StoredData
from substrateinterface import Keypair

from validator import fidelity_check
from validator.dataset import Dataset
from validator.metagraph_sync import MetagraphSynchronizer
from validator.miner_data import MinerData
from validator.rate_limiter import RateLimiter
from validator.task_registry import TaskRegistry
from validator.version import VALIDATOR_VERSION


NEURONS_LIMIT = 256


class Validator:
    uid: int
    """Each validator gets a unique identity (UID) in the network for differentiation."""
    config: bt.config
    """Copy of the original config."""
    wallet: bt.wallet
    """The wallet holds the cryptographic key pairs for the miner."""
    subtensor: bt.subtensor
    """The subtensor is the connection to the Bittensor blockchain."""
    metagraph: bt.metagraph
    """The metagraph holds the state of the network, letting us know about other validators and miners."""
    axon: bt.axon
    """Axon for external connections."""
    dataset: Dataset
    """Prompts for synthetic tasks."""
    task_registry: TaskRegistry
    """Organic tasks registry."""
    miners: list[MinerData]
    """List of neurons."""
    metagraph_sync: MetagraphSynchronizer
    """Simple wrapper to encapsulate metagraph syncs."""
    public_api_limiter: RateLimiter
    """Rate limiter for organic requests."""
    storage: Storage | None
    """Bittensor storage subnet is used to store generated assets (if enabled)."""
    updater: AutoUpdater
    """Simple wrapper to encapsulate neuron auto-updates."""

    def __init__(self, config: bt.config) -> None:
        self.config: bt.config = copy.deepcopy(config)
        create_neuron_dir(self.config)

        bt.logging(config=config, logging_dir=config.full_path)

        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        self._self_check_for_registration()

        self.metagraph = bt.metagraph(
            netuid=self.config.netuid, network=self.subtensor.network, sync=False
        )  # Make sure not to sync without passing subtensor

        self.metagraph_sync = MetagraphSynchronizer(
            self.metagraph,
            self.subtensor,
            self.config.neuron.sync_interval,
            self.config.neuron.log_info_interval,
            self.config.neuron.strong_miners_count,
        )
        self.metagraph_sync.sync()

        bt.logging.info(f"Metagraph: {self.metagraph}")

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet {self.config.netuid} with uid {self.uid} "
            f"using network: {self.subtensor.chain_endpoint}"
        )

        if not self._is_enough_stake_to_set_weights():
            bt.logging.warning(
                f"You need t{self.config.neuron.min_stake_to_set_weights} to set weights. "
                f"You have t{self.metagraph.S[self.uid]}"
            )

        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        self.axon.attach(
            forward_fn=self.pull_task,
            blacklist_fn=self.blacklist_pulling_task,
        ).attach(
            forward_fn=self.submit_results,
            blacklist_fn=self.blacklist_submitting_results,
            priority_fn=self.prioritize_submitting_results,
        ).attach(
            forward_fn=self.get_version,
            blacklist_fn=self.blacklist_getting_version,
        )

        if self.config.public_api.enabled:
            if self.config.public_api.whitelist_disabled:
                bt.logging.warning("Security risk. Whitelist for public API disabled.")

            self.axon.attach(
                forward_fn=self.generate,
                blacklist_fn=self.blacklist_generation,
            ).attach(
                forward_fn=self.check_status,
                blacklist_fn=self.blacklist_checking_status,
            )

        bt.logging.info(f"Axon created: {self.axon}")

        self.dataset = Dataset(
            default_prompts_path=self.config.dataset.default_prompts_path,
            prompter_url=self.config.dataset.prompter.endpoint,
            fetch_prompt_interval=self.config.dataset.prompter.fetch_interval,
            wallet=self.wallet,
        )

        self.task_registry = TaskRegistry(
            copies=self.config.public_api.copies,
            wait_after_first_copy=self.config.public_api.wait_after_first_copy,
            task_timeout=self.config.generation.task_timeout,
            polling_interval=self.config.public_api.polling_interval,
        )

        self.public_api_limiter = RateLimiter(
            max_requests=self.config.public_api.rate_limit.requests, period=self.config.public_api.rate_limit.period
        )

        self.miners = [MinerData(uid=x) for x in range(NEURONS_LIMIT)]
        self.load_state()

        self.updater = AutoUpdater(
            disabled=self.config.neuron.auto_update_disabled,
            interval=self.config.neuron.auto_update_interval,
            local_version=VALIDATOR_VERSION,
        )

        if self.config.storage.enabled:
            self.storage = Storage(config)
        else:
            self.storage = None

    def generate(self, synapse: Generate) -> Generate:
        """Public API. Generation request."""
        bt.logging.debug(
            f"Public API. Generation requested. Requester: {synapse.dendrite.hotkey}. Prompt: {synapse.prompt}"
        )

        synapse.task_id = self.task_registry.add_task(synapse.prompt, synapse.dendrite.hotkey)
        synapse.polling_interval = self.config.public_api.polling_interval
        return synapse

    def blacklist_generation(self, synapse: Generate) -> Tuple[bool, str]:  # noqa: UP006, UP035
        if not self._is_wallet_whitelisted(synapse.dendrite.hotkey):
            return True, "Not whitelisted"

        if not self.public_api_limiter.is_allowed(synapse.dendrite.hotkey):
            bt.logging.info(f"{synapse.dendrite.hotkey} hit the rate limit for public api calls")
            return True, "Rate limit"

        return False, ""

    def _is_wallet_whitelisted(self, hotkey: str) -> bool:
        """Checks whether wallet is whitelisted to use public API."""

        # TODO: move whitelist to the config file
        # TODO: add dynamic watchers to whitelist additional keys without restarting the validator.

        # Temporary solution. Only one wallet that belongs to the subnet owners is whitelisted.
        # Validators are free to modify this list.
        # Once the testing period is over, run-time whitelist modification will be provided.
        whitelist = ["5DwHD8Ja9aWGhB5nbmZqCCs9GNMpzxhBcZjnhcY73Sd75qe7"]
        if not self.config.public_api.whitelist_disabled and hotkey not in whitelist:
            bt.logging.warning(f"{hotkey} is not white-listed for the public API")
            return False

        return True

    def check_status(self, synapse: StatusCheck) -> StatusCheck:
        """Public API. Status check request. Rate limit is applied in the `get_task_status`."""

        synapse.status, synapse.results = self.task_registry.get_task_status(synapse.task_id, synapse.dendrite.hotkey)
        return synapse

    def blacklist_checking_status(self, synapse: StatusCheck) -> Tuple[bool, str]:  # noqa: UP006, UP035
        if not self._is_wallet_whitelisted(synapse.dendrite.hotkey):
            return True, "Not whitelisted"

        return False, ""

    def pull_task(self, synapse: PullTask) -> PullTask:
        """Miner requesting new task from the validator."""

        uid = self._get_neuron_uid(synapse.dendrite.hotkey)
        if uid is None:
            bt.logging.error("Unexpected behaviour, unknown neuron after blacklist")
            return synapse

        miner = self.miners[uid]
        if miner.is_task_expired(self.config.generation.task_timeout):
            bt.logging.debug(f"[{uid}] asked for a new task while having expired task. New task will be assigned")
            miner.reset_task()

        cooldown_violation = True
        if miner.assigned_task is not None:
            bt.logging.debug(
                f"[{uid}] asked for a new task while having assigned task. " f"Resetting the task and setting cooldown"
            )
            cooldown_violation = False
            miner.reset_task(cooldown=self.config.generation.task_cooldown)

        if miner.is_on_cooldown():
            if cooldown_violation:
                miner.cooldown_violations += 1
            bt.logging.debug(
                f"[{uid}] asked for a new task while on a cooldown. " f"Total violations: {miner.cooldown_violations}"
            )
            synapse.cooldown_until = miner.cooldown_until
            synapse.cooldown_violations = miner.cooldown_violations
            return synapse

        is_strong_miner = self.metagraph_sync.is_strong_miner(uid)

        organic_task = self.task_registry.get_next_task(synapse.dendrite.hotkey, is_strong_miner=is_strong_miner)
        if organic_task is not None:
            task = Task(id=organic_task.id, prompt=organic_task.prompt)
            bt.logging.debug(f"[{uid}] pulls organic task ({task.prompt} | {task.id})")
        else:
            task = Task(prompt=self.dataset.get_random_prompt())
            # task = Task(prompt="tourmaline tassel earring")
            bt.logging.debug(f"[{uid}] pulls synthetic task ({task.prompt} | {task.id})")

        miner.assign_task(task)

        synapse.task = task
        synapse.submit_before = int(time.time()) + self.config.generation.task_timeout
        synapse.version = NEURONS_VERSION
        return synapse

    def blacklist_pulling_task(self, synapse: PullTask) -> Tuple[bool, str]:  # noqa: UP006, UP035
        uid = self._get_neuron_uid(synapse.dendrite.hotkey)
        if uid is None:
            return True, f"Unrecognized hotkey {synapse.dendrite.hotkey} ({synapse.dendrite.ip})"

        return False, ""

    async def submit_results(self, synapse: SubmitResults) -> SubmitResults:
        uid = self._get_neuron_uid(synapse.dendrite.hotkey)
        if uid is None:
            bt.logging.error("Unexpected behaviour, unknown neuron after blacklist")
            return synapse

        miner = self.miners[uid]

        if synapse.task is None:
            bt.logging.warning(f"[{uid}] submitted results with no original task")
            return self._add_feedback(synapse, miner)

        if miner.assigned_task != synapse.task:
            bt.logging.warning(f"[{uid}] submitted results for the wrong task")
            return self._add_feedback(synapse, miner)

        if synapse.results == "":
            bt.logging.debug(f"[{uid}] submitted empty results")
            miner.reset_task(cooldown=self.config.generation.task_cooldown)
            return self._add_feedback(synapse, miner)

        if not self._verify_results_signature(synapse):
            bt.logging.warning(f"[{uid}] submitted results with wrong signature")
            return self._add_feedback(synapse, miner)

        validation_score = await fidelity_check.validate(
            self.config.validation.endpoint, synapse.task.prompt, synapse.results
        )
        if validation_score is None:
            miner.reset_task(cooldown=self.config.generation.task_cooldown)
            return self._add_feedback(synapse, miner, validation_failed=True)

        fidelity_score = self._get_fidelity_score(validation_score)

        miner.reset_task(cooldown=self.config.generation.task_cooldown)

        if fidelity_score == 0:
            bt.logging.debug(f"[{uid}] submitted results with low fidelity score. Results not accepted")
            return self._add_feedback(synapse, miner)

        if self.storage is not None:
            asyncio.create_task(self.storage.store(StoredData.from_results(synapse)))

        current_time = int(time.time())
        miner.add_observation(
            task_finish_time=current_time,
            fidelity_score=fidelity_score,
            moving_average_alpha=self.config.neuron.moving_average_alpha,
        )

        self.task_registry.complete_task(synapse.task.id, synapse.dendrite.hotkey, synapse.results, validation_score)

        return self._add_feedback(synapse, miner, current_time=current_time, fidelity_score=fidelity_score)

    def _add_feedback(
        self,
        synapse: SubmitResults,
        miner: MinerData,
        fidelity_score: float = 0.0,
        current_time: int | None = None,
        validation_failed: bool = False,
    ) -> SubmitResults:
        if current_time is None:
            current_time = int(time.time())
        reward = miner.calculate_reward(current_time)
        synapse.feedback = Feedback(
            validation_failed=validation_failed,
            task_fidelity_score=fidelity_score,
            average_fidelity_score=miner.fidelity_score,
            generations_within_8_hours=len(miner.observations),
            current_miner_reward=reward,
        )
        synapse.cooldown_until = current_time + self.config.generation.task_cooldown
        return synapse

    @staticmethod
    def _verify_results_signature(synapse: SubmitResults) -> bool:
        # This security measure is redundant for results validation process, however
        # it's needed for stored results verification.
        if synapse.task is None:
            return False

        keypair = Keypair(ss58_address=synapse.dendrite.hotkey)
        message = f"{synapse.submit_time}{synapse.task.prompt}{synapse.axon.hotkey}{synapse.dendrite.hotkey}"
        return bool(keypair.verify(message, base64.b64decode(synapse.signature.encode(encoding="utf-8"))))

    @staticmethod
    def _get_fidelity_score(validation_score: float) -> float:
        # To avoid any randomness or luck in the validation, threshold approach is used.
        if validation_score >= 0.8:
            return 1.0
        if validation_score >= 0.6:
            return 0.75
        return 0.0

    def blacklist_submitting_results(self, synapse: SubmitResults) -> Tuple[bool, str]:  # noqa: UP006, UP035
        uid = self._get_neuron_uid(synapse.dendrite.hotkey)
        if uid is None:
            return True, f"Unrecognized hotkey {synapse.dendrite.hotkey} ({synapse.dendrite.ip})"

        miner = self.miners[uid]
        if miner.assigned_task is None:
            return True, (
                f"[{uid}] submitted results while having no task assigned. "
                f"It could happen if validator restarts or miner faults"
            )

        return False, ""

    def prioritize_submitting_results(self, synapse: SubmitResults) -> float:
        uid = self._get_neuron_uid(synapse.dendrite.hotkey)
        if uid is None:
            bt.logging.error("Unexpected behaviour, unknown neuron after blacklist")
            return 0.0

        return float(self.metagraph.S[uid])

    async def get_version(self, synapse: GetVersion) -> GetVersion:
        synapse.version = VALIDATOR_VERSION
        synapse.validation_version = await fidelity_check.version(self.config.validation.endpoint)
        return synapse

    def blacklist_getting_version(self, synapse: GetVersion) -> Tuple[bool, str]:  # noqa: UP006, UP035
        uid = self._get_neuron_uid(synapse.dendrite.hotkey)
        if uid is None:
            return True, f"Unrecognized hotkey {synapse.dendrite.hotkey} ({synapse.dendrite.ip})"

        if synapse.dendrite.hotkey != owner.HOTKEY and not self._is_enough_stake_to_set_weights(uid):
            return True, (
                f"Version check allowed for validators only. "
                f"Request from {synapse.dendrite.hotkey} ({synapse.dendrite.ip})"
            )

        return False, ""

    def _self_check_for_registration(self) -> None:
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            raise RuntimeError(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again."
            )

    def _is_enough_stake_to_set_weights(self, uid: int | None = None) -> bool:
        uid_to_check = self.uid if uid is None else uid
        return bool(self.metagraph.S[uid_to_check].item() >= self.config.neuron.min_stake_to_set_weights)

    def _get_neuron_uid(self, hotkey: str) -> int | None:
        for neuron in self.metagraph.neurons:
            if neuron.hotkey == hotkey:
                return int(neuron.uid)

        return None

    async def run(self) -> None:
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        bt.logging.info(
            f"Serving validator axon {self.axon} on network: {self.config.subtensor.chain_endpoint} "
            f"with netuid: {self.config.netuid}"
        )

        bt.logging.debug("Starting the validator.")

        while True:
            await asyncio.sleep(5)
            self.metagraph_sync.log_info(self.uid)

            if self.metagraph_sync.should_sync():
                self.save_state()
                self.metagraph_sync.sync()
                self._set_weights()

            if await self.updater.should_update():
                self.save_state()
                await self.updater.update()
                break

            if self.dataset.should_fetch_fresh_prompts():
                await self.dataset.fetch_fresh_prompts()

    class State(BaseModel):
        miners: list[MinerData]

    def save_state(self) -> None:
        try:
            path = self.config.neuron.full_path / "state.txt"
            with path.open("w") as f:
                f.write(Validator.State(miners=self.miners).json())
        except Exception as e:
            bt.logging.exception(f"Validator state saving failed with {e}")

    def load_state(self) -> None:
        path = self.config.neuron.full_path / "state.txt"
        if not path.exists():
            bt.logging.warning("No saved state found")
            return

        try:
            with path.open("r") as f:
                content = f.read()
            self.miners = Validator.State.parse_raw(content).miners
        except Exception as e:
            bt.logging.exception(f"Failed to load the state: {e}")

        bt.logging.info("Validator state loaded.")

    def _set_weights(self) -> None:
        if not self._is_enough_stake_to_set_weights():
            return

        if self.metagraph.last_update[self.uid] + self.config.neuron.weight_set_interval > self.metagraph.block:
            return

        current_time = int(time.time())
        rewards = torch.tensor([miner.calculate_reward(current_time) for miner in self.miners])

        bt.logging.debug(f"Rewards: {rewards}")

        raw_weights = torch.nn.functional.normalize(rewards, p=1, dim=0)

        # bt.logging.debug(f"Normalized weights: {raw_weights}")

        (
            processed_weight_uids,
            processed_weights,
        ) = weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        (
            converted_uids,
            converted_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )

        self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=converted_uids,
            weights=converted_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=int(NEURONS_VERSION),
        )
