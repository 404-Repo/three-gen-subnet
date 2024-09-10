import asyncio
import base64
import copy
import time
from typing import Tuple  # noqa: UP035

import bittensor as bt
import torch
from auto_updater import AutoUpdater
from bittensor.utils import weight_utils
from common import create_neuron_dir, owner
from common.protocol import Feedback, GetVersion, PullTask, SubmitResults, Task
from common.version import NEURONS_VERSION
from pydantic import BaseModel
from storage_subnet import Storage, StoredData
from substrateinterface import Keypair

from validator import fidelity_check
from validator.api import PublicAPIServer
from validator.api.task_registry import TaskRegistry
from validator.dataset import Dataset
from validator.metagraph_sync import MetagraphSynchronizer
from validator.miner_data import MinerData
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
    miners: list[MinerData]
    """List of neurons."""
    metagraph_sync: MetagraphSynchronizer
    """Simple wrapper to encapsulate metagraph syncs."""
    storage: Storage | None
    """Bittensor storage subnet is used to store generated assets (if enabled)."""
    updater: AutoUpdater
    """Simple wrapper to encapsulate neuron auto-updates."""
    task_registry: TaskRegistry | None
    """Organic tasks registry."""
    public_server: PublicAPIServer | None
    """FastApi server that serves the public API endpoint."""

    def __init__(self, config: bt.config) -> None:
        self.config: bt.config = copy.deepcopy(config)
        create_neuron_dir(self.config)

        bt.logging(config=config, logging_dir=config.full_path)

        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        self._self_check_for_registration()

        self.miners = [MinerData(uid=x) for x in range(NEURONS_LIMIT)]
        self.load_state()

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
        self.metagraph_sync.sync(self.miners)

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
            priority_fn=self.prioritize_getting_version,
        )

        bt.logging.info(f"Axon created: {self.axon}")

        self.dataset = Dataset(
            default_prompts_path=self.config.dataset.default_prompts_path,
            prompter_url=self.config.dataset.prompter.endpoint,
            fetch_prompt_interval=self.config.dataset.prompter.fetch_interval,
            wallet=self.wallet,
        )

        if self.config.public_api.enabled:
            self.task_registry = TaskRegistry(
                copies=self.config.public_api.copies,
                wait_after_first_copy=self.config.public_api.wait_after_first_copy,
                task_timeout=self.config.generation.task_timeout,
            )

            self.public_server = PublicAPIServer(config=self.config, task_registry=self.task_registry)
        else:
            self.task_registry = None
            self.public_server = None

        self.updater = AutoUpdater(
            disabled=self.config.neuron.auto_update_disabled,
            interval=self.config.neuron.auto_update_interval,
            local_version=VALIDATOR_VERSION,
        )

        if self.config.storage.enabled:
            self.storage = Storage(config)
        else:
            self.storage = None

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

        if miner.assigned_task is not None:
            bt.logging.debug(
                f"[{uid}] asked for a new task while having assigned task. Resetting the task and setting cooldown"
            )
            miner.reset_task(cooldown=self.config.generation.task_cooldown)
            return self._add_cooldown_data(synapse, miner)

        if miner.is_on_cooldown():
            miner.cooldown_violations += 1
            bt.logging.debug(
                f"[{uid}] asked for a new task while on a cooldown ({miner.cooldown_left()} sec left). "
                f"Total violations: {miner.cooldown_violations}"
            )
            cooldown_violations_threshold_reached = (
                miner.cooldown_violations > self.config.neuron.cooldown_violations_threshold
            )
            if cooldown_violations_threshold_reached:
                miner.cooldown_until += self.config.neuron.cooldown_violation_penalty
                bt.logging.debug(f"[{uid}] Cooldown penalty added.")

            return self._add_cooldown_data(synapse, miner)

        is_strong_miner = self.metagraph_sync.is_strong_miner(uid)

        if self.task_registry is not None:
            organic_task = self.task_registry.get_next_task(synapse.dendrite.hotkey, is_strong_miner=is_strong_miner)
        else:
            organic_task = None

        if organic_task is not None:
            task = Task(id=organic_task.id, prompt=organic_task.prompt)
            bt.logging.debug(f"[{uid}] pulls organic task ({task.prompt} | {task.id})")
        else:
            task = Task(prompt=self.dataset.get_random_prompt())
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
            return self._add_feedback_and_strip(synapse, miner)

        if miner.assigned_task != synapse.task:
            bt.logging.warning(f"[{uid}] submitted results for the wrong task")
            return self._add_feedback_and_strip(synapse, miner)

        # Reducing cooldown violations to allow some accidental cooldown violations.
        miner.cooldown_violations = max(0, miner.cooldown_violations - 1)

        if synapse.results == "":
            bt.logging.debug(f"[{uid}] submitted empty results")

            self._reset_miner_on_failure(miner=miner, hotkey=synapse.dendrite.hotkey, task_id=synapse.task.id)
            return self._add_feedback_and_strip(synapse, miner)

        if not self._verify_results_signature(synapse):
            bt.logging.warning(f"[{uid}] submitted results with wrong signature")

            self._reset_miner_on_failure(
                miner=miner,
                hotkey=synapse.dendrite.hotkey,
                task_id=synapse.task.id,
                cooldown_penalty=self.config.generation.cooldown_penalty,
            )
            return self._add_feedback_and_strip(synapse, miner)

        validation_score = await fidelity_check.validate(
            self.config.validation.endpoint, synapse.task.prompt, synapse.results, synapse.data_format, synapse.data_ver
        )
        if validation_score is None:
            self._reset_miner_on_failure(miner=miner, hotkey=synapse.dendrite.hotkey, task_id=synapse.task.id)
            return self._add_feedback_and_strip(synapse, miner, validation_failed=True)

        fidelity_score = self._get_fidelity_score(validation_score)

        if fidelity_score == 0:
            bt.logging.debug(f"[{uid}] submitted results with low fidelity score. Results not accepted")

            self._reset_miner_on_failure(
                miner=miner,
                hotkey=synapse.dendrite.hotkey,
                task_id=synapse.task.id,
                cooldown_penalty=self.config.generation.cooldown_penalty,
            )
            return self._add_feedback_and_strip(synapse, miner)

        miner.reset_task(cooldown=self.config.generation.task_cooldown)

        if self.storage is not None:
            asyncio.create_task(self.storage.store(StoredData.from_results(synapse)))

        current_time = int(time.time())
        miner.add_observation(
            task_finish_time=current_time,
            fidelity_score=fidelity_score,
            moving_average_alpha=self.config.neuron.moving_average_alpha,
        )

        if self.task_registry is not None:
            self.task_registry.complete_task(
                synapse.task.id, synapse.dendrite.hotkey, synapse.results, synapse.data_format, validation_score
            )

        return self._add_feedback_and_strip(synapse, miner, current_time=current_time, fidelity_score=fidelity_score)

    def _add_cooldown_data(self, synapse: PullTask, miner: MinerData) -> PullTask:
        synapse.cooldown_violations = miner.cooldown_violations
        synapse.cooldown_until = miner.cooldown_until
        return synapse

    def _reset_miner_on_failure(self, miner: MinerData, hotkey: str, task_id: str, cooldown_penalty: int = 0) -> None:
        miner.reset_task(cooldown=self.config.generation.task_cooldown + cooldown_penalty)
        if self.task_registry is not None:
            self.task_registry.fail_task(task_id, hotkey)

    def _add_feedback_and_strip(
        self,
        synapse: SubmitResults,
        miner: MinerData,
        fidelity_score: float = 0.0,
        current_time: int | None = None,
        validation_failed: bool = False,
    ) -> SubmitResults:
        synapse.results = ""
        synapse.signature = ""

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
        synapse.cooldown_until = miner.cooldown_until
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

    def prioritize_getting_version(self, synapse: GetVersion) -> float:
        return 10000000  # maximizing priority

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

        if self.public_server is not None:
            self.public_server.start()

        bt.logging.debug("Starting the validator.")

        while True:
            await asyncio.sleep(5)
            self.metagraph_sync.log_info(self.uid)

            if self.metagraph_sync.should_sync():
                self.save_state()
                self.metagraph_sync.sync(self.miners)
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
