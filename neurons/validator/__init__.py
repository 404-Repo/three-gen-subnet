import asyncio
import copy
import time
from typing import Tuple  # noqa: UP035

import bittensor as bt
import numpy as np
import pybase64
from auto_updater import AutoUpdater
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from bittensor_wallet import Keypair
from common import create_neuron_dir, owner
from common.miner_license_consent_declaration import MINER_LICENSE_CONSENT_DECLARATION
from common.protocol import Feedback, GetVersion, PullTask, SubmitResults, Task
from numpy.typing import NDArray
from pydantic import BaseModel

from validator import fidelity_check
from validator.api import PublicAPIServer
from validator.api.task_registry import TaskRegistry
from validator.dataset import Dataset
from validator.fidelity_check import ValidationResponse
from validator.metagraph_sync import MetagraphSynchronizer
from validator.miner_data import MinerData
from validator.storage import StorageWrapper
from validator.telemetry import Telemetry
from validator.version import VALIDATOR_VERSION


NEURONS_LIMIT = 256


class Validator:
    uid: int
    """Each validator gets a unique identity (UID) in the network for differentiation."""
    config: bt.config
    """Copy of the original config."""
    wallet: bt.wallet
    """The wallet holds the cryptographic key pairs for the validator."""
    subtensor: bt.subtensor
    """The subtensor is the connection to the Bittensor blockchain."""
    metagraph: bt.metagraph
    """The metagraph holds the state of the network, letting us know about other validators and miners."""
    axon: bt.axon
    """Axon for external connections."""
    dataset: Dataset
    """Prompts for synthetic tasks."""
    miners: list[MinerData]
    """List of miners."""
    metagraph_sync: MetagraphSynchronizer
    """Simple wrapper to encapsulate the assets storage interaction."""
    storage: StorageWrapper
    """Simple wrapper to encapsulate the communication with the storage service (if enabled)."""
    updater: AutoUpdater
    """Simple wrapper to encapsulate validator auto-updates."""
    task_registry: TaskRegistry | None
    """Organic tasks registry."""
    public_server: PublicAPIServer | None
    """FastApi server that serves the public API endpoint."""
    telemetry: Telemetry
    """Responsible for sending validator metrics to the metrics push gateway."""

    def __init__(
        self, config: bt.config, wallet: bt.wallet | None = None, subtensor: bt.subtensor | None = None
    ) -> None:
        self.config: bt.config = copy.deepcopy(config)
        create_neuron_dir(self.config)

        bt.logging.set_config(config=self.config.logging)

        bt.logging.info(f"Starting with config: {config}")

        if wallet is None:
            self.wallet = bt.wallet(config=self.config)
        else:
            self.wallet = wallet
        bt.logging.info(f"Wallet: {self.wallet}")

        if subtensor is None:
            self.subtensor = bt.subtensor(config=self.config)
        else:
            self.subtensor = subtensor
        bt.logging.info(f"Subtensor: {self.subtensor}")

        self._self_check_for_registration()

        self.miners = [MinerData(uid=x) for x in range(NEURONS_LIMIT)]
        self.load_state()

        self._prepare_metagraph()
        self._prepare_axon()
        self._prepare_dataset()
        self._prepare_public_api()
        self._prepare_updater()
        self._prepare_storage()
        self._prepare_telemetry()

        if not self._is_enough_stake_to_set_weights():
            bt.logging.warning(
                f"You need t{self.config.neuron.min_stake_to_set_weights} to set weights. "
                f"You have t{self.metagraph.S[self.uid]}"
            )

    def _self_check_for_registration(self) -> None:
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            raise RuntimeError(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again."
            )

    def _prepare_metagraph(self) -> None:
        self.metagraph = bt.metagraph(
            netuid=self.config.netuid, network=self.subtensor.network, sync=False
        )  # Make sure not to sync without passing subtensor

        self.metagraph_sync = MetagraphSynchronizer(
            self.metagraph,
            self.subtensor,
            self.config.neuron.sync_interval,
            self.config.neuron.log_info_interval,
            self.config.public_api.strong_miners_count,
        )
        self.metagraph_sync.sync(self.miners)

        bt.logging.info(f"Metagraph: {self.metagraph}")

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet {self.config.netuid} with uid {self.uid} "
            f"using network: {self.subtensor.chain_endpoint}"
        )

    def _prepare_axon(self) -> None:
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        self.axon.attach(
            forward_fn=self.pull_task,
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

    def _prepare_dataset(self) -> None:
        self.dataset = Dataset(
            default_prompts_path=self.config.dataset.default_prompts_path,
            prompter_url=self.config.dataset.prompter.endpoint,
            fetch_prompt_interval=self.config.dataset.prompter.fetch_interval,
            wallet=self.wallet,
        )

    def _prepare_public_api(self) -> None:
        if self.config.public_api.enabled:
            bt.logging.info("Starting public API server...")
            self.task_registry = TaskRegistry(
                copies=self.config.public_api.copies,
                wait_after_first_copy=self.config.public_api.wait_after_first_copy,
                task_timeout=self.config.public_api.task_timeout,
            )

            self.public_server = PublicAPIServer(config=self.config, task_registry=self.task_registry)
            bt.logging.info("Public API server started.")
        else:
            bt.logging.info("Public API server disabled.")
            self.task_registry = None
            self.public_server = None

    def _prepare_storage(self) -> None:
        self.storage = StorageWrapper(
            self.config.storage.enabled,
            self.config.storage.service_api_key,
            self.config.storage.endpoint_url,
            self.config.storage.validation_score_threshold,
        )

    def _prepare_updater(self) -> None:
        self.updater = AutoUpdater(
            disabled=self.config.neuron.auto_update_disabled,
            interval=self.config.neuron.auto_update_interval,
            local_version=VALIDATOR_VERSION,
        )

    def _prepare_telemetry(self) -> None:
        self.telemetry = Telemetry(self.miners, self.wallet, self.metagraph, self.config)

    def pull_task(self, synapse: PullTask) -> PullTask:
        """Miner requesting new task from the validator."""

        miner_uid = self._get_neuron_uid(synapse.dendrite.hotkey)
        if miner_uid is None:
            bt.logging.warning(f"Unknown hotkey: {synapse.dendrite.hotkey} ({synapse.dendrite.ip})")
            synapse.cooldown_until = int(time.time()) + 3600
            return synapse

        miner = self.miners[miner_uid]
        cooldown_detected, response_synapse = self._detect_cooldown_violation(synapse, miner, miner_uid)
        if cooldown_detected:
            return response_synapse

        return self._assign_new_task(synapse, miner, miner_uid)

    def _detect_cooldown_violation(self, synapse: PullTask, miner: MinerData, uid: int) -> tuple[bool, PullTask]:
        if miner.assigned_task is not None:
            bt.logging.debug(
                f"[{uid}] asked for a new task while having assigned task. Resetting the task and setting cooldown"
            )
            miner.reset_task(
                throttle_period=self.config.generation.throttle_period, cooldown=self.config.generation.task_cooldown
            )
            return True, self._add_cooldown_data(synapse, miner)

        if miner.is_on_cooldown():
            miner.cooldown_violations += 1
            bt.logging.debug(
                f"[{uid}] asked for a new task while on a cooldown ({miner.cooldown_left()} sec left). "
                f"Total violations: {miner.cooldown_violations}"
            )
            cooldown_violations_threshold_reached = (
                miner.cooldown_violations > self.config.generation.cooldown_violations_threshold
            )
            if cooldown_violations_threshold_reached:
                miner.cooldown_until += self.config.generation.cooldown_violation_penalty
                bt.logging.debug(f"[{uid}] Cooldown penalty added.")

            return True, self._add_cooldown_data(synapse, miner)
        return False, synapse

    def _add_cooldown_data(self, synapse: PullTask, miner: MinerData) -> PullTask:
        synapse.cooldown_violations = miner.cooldown_violations
        synapse.cooldown_until = miner.cooldown_until
        return synapse

    def _assign_new_task(self, synapse: PullTask, miner: MinerData, uid: int) -> PullTask:
        is_strong_miner = self.metagraph_sync.is_strong_miner(uid)
        task = self._get_task(synapse.dendrite.hotkey, is_strong_miner, uid)
        miner.assign_task(task)

        synapse.task = task
        synapse.validation_threshold = self.config.generation.quality_threshold
        synapse.throttle_period = self.config.generation.throttle_period
        return synapse

    def _get_task(self, hotkey: str, is_strong_miner: bool, uid: int) -> Task:
        if self.task_registry is not None:
            organic_task = self.task_registry.get_next_task(hotkey, is_strong_miner=is_strong_miner)
            if organic_task is not None:
                task = Task(id=organic_task.id, prompt=organic_task.prompt)
                bt.logging.debug(f"[{uid}] pulls organic task ({task.prompt[:100]} | {task.id})")
                return task

        task = Task(prompt=self.dataset.get_random_prompt())
        bt.logging.debug(f"[{uid}] pulls synthetic task ({task.prompt[:100]} | {task.id})")
        return task

    async def submit_results(self, synapse: SubmitResults) -> SubmitResults:
        uid = self._get_neuron_uid(synapse.dendrite.hotkey)
        if uid is None:
            bt.logging.error("Unexpected behaviour, unknown neuron after blacklist")
            return synapse

        miner = self.miners[uid]
        miner.cooldown_violations = max(0, miner.cooldown_violations - 1)

        if self._is_wrong_task(synapse, miner):
            return self._add_feedback_and_strip(synapse, miner)

        if self._has_miner_skipped_task(synapse, miner):
            return self._process_task_failure(synapse=synapse, miner=miner, cooldown_penalty=0)

        if self._is_wrong_signature(synapse, miner):
            return self._process_task_failure(
                synapse=synapse, miner=miner, cooldown_penalty=self.config.generation.cooldown_penalty
            )

        validation_endpoint = self.config.validation.endpoints[uid % len(self.config.validation.endpoints)]
        validation_res = await fidelity_check.validate(
            validation_endpoint, synapse, self.storage.enabled, self.storage.validation_score_threshold
        )
        if validation_res is None:
            return self._process_task_failure(synapse=synapse, miner=miner, cooldown_penalty=0, validation_failed=True)

        fidelity_score = self._get_fidelity_score(validation_res.score)
        if fidelity_score == 0:
            bt.logging.debug(f"[{uid}] submitted results with low fidelity score. Prompt: {synapse.task.prompt[:100]}")
            return self._process_task_failure(
                synapse=synapse,
                miner=miner,
                score=validation_res.score,
                cooldown_penalty=self.config.generation.cooldown_penalty,
                validation_failed=False,
            )

        return await self._process_valid_submission(synapse, miner, fidelity_score, validation_res)

    def _is_wrong_task(self, synapse: SubmitResults, miner: MinerData) -> bool:
        if miner.assigned_task != synapse.task:
            bt.logging.warning(f"[{miner.uid}] submitted results for the wrong task")
            return True
        return False

    def _is_wrong_signature(self, synapse: SubmitResults, miner: MinerData) -> bool:
        if not self._verify_results_signature(synapse):
            bt.logging.error(f"[{miner.uid}] submitted results with invalid signature")
            return True
        return False

    def _has_miner_skipped_task(self, synapse: SubmitResults, miner: MinerData) -> bool:
        if synapse.results == "":
            bt.logging.debug(f"[{miner.uid}] submitted empty results ({synapse.task.prompt[:100]})")
            return True
        return False

    def _process_task_failure(
        self,
        synapse: SubmitResults,
        miner: MinerData,
        score: float = 0.0,
        cooldown_penalty: int = 0,
        validation_failed: bool = False,
    ) -> SubmitResults:
        delivery_time = (int(time.time()) - miner.assignment_time) if miner.assignment_time else 0

        miner.reset_task(
            throttle_period=self.config.generation.throttle_period,
            cooldown=self.config.generation.task_cooldown + cooldown_penalty,
        )

        if self.task_registry is not None:
            self.task_registry.fail_task(synapse.task.id, synapse.dendrite.hotkey)

        self.telemetry.add_task_metrics(
            miner_hotkey=synapse.dendrite.hotkey,
            miner_coldkey=self.metagraph.coldkeys[miner.uid],
            score=score,
            delivery_time=delivery_time,
            size=len(synapse.results),
            compression=synapse.compression,
        )

        return self._add_feedback_and_strip(synapse, miner, validation_failed=validation_failed)

    async def _process_valid_submission(
        self, synapse: SubmitResults, miner: MinerData, fidelity_score: float, validation_res: ValidationResponse
    ) -> SubmitResults:
        delivery_time = (int(time.time()) - miner.assignment_time) if miner.assignment_time else 0

        miner.reset_task(
            throttle_period=self.config.generation.throttle_period, cooldown=self.config.generation.task_cooldown
        )

        if self.storage.enabled:
            is_organic = self.task_registry and synapse.task and self.task_registry.is_organic(synapse.task.id)
            if not is_organic:
                asyncio.create_task(
                    self.storage.save_assets(synapse, synapse.results, synapse.signature, validation_res)
                )

        current_time = int(time.time())
        miner.add_observation(
            task_finish_time=current_time,
            fidelity_score=fidelity_score,
            moving_average_alpha=self.config.validation.moving_average_alpha,
        )

        if self.task_registry is not None:
            self.task_registry.complete_task(synapse, validation_res.score)

        self.telemetry.add_task_metrics(
            miner_hotkey=synapse.dendrite.hotkey,
            miner_coldkey=self.metagraph.coldkeys[miner.uid],
            score=validation_res.score,
            delivery_time=delivery_time,
            size=len(synapse.results),
            compression=synapse.compression,
        )

        return self._add_feedback_and_strip(synapse, miner, current_time=current_time, fidelity_score=fidelity_score)

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
            generations_within_the_window=len(miner.observations),
            current_miner_reward=reward,
        )
        synapse.cooldown_until = miner.cooldown_until
        return synapse

    @staticmethod
    def _verify_results_signature(synapse: SubmitResults) -> bool:
        keypair = Keypair(ss58_address=synapse.dendrite.hotkey)
        message = (
            f"{MINER_LICENSE_CONSENT_DECLARATION}"
            f"{synapse.submit_time}{synapse.task.prompt}{synapse.axon.hotkey}{synapse.dendrite.hotkey}"
        )
        encoded_signature = pybase64.b64decode(synapse.signature.encode(encoding="utf-8"), validate=True)
        return bool(keypair.verify(message, encoded_signature))

    def _get_fidelity_score(self, validation_score: float) -> float:
        if validation_score < self.config.generation.quality_threshold:
            return 0.0
        return min(1.0, validation_score)

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
        synapse.validation_version = await fidelity_check.version(self.config.validation.endpoints[0])
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

        await self.telemetry.start()

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
            self.miners = Validator.State.model_validate_json(content).miners
        except Exception as e:
            bt.logging.exception(f"Failed to load the state: {e}")

        bt.logging.info("Validator state loaded.")

    def _set_weights(self) -> None:
        if not self._is_enough_stake_to_set_weights():
            return

        if self.metagraph.last_update[self.uid] + self.config.neuron.weight_set_interval > self.metagraph.block:
            return

        current_time = int(time.time())
        rewards = np.array([miner.calculate_reward(current_time) for miner in self.miners])
        bt.logging.debug(f"Rewards: {rewards}")

        def strip_sigmoid(x: NDArray[np.uint16], n: int) -> NDArray[np.float64]:
            return np.array(0.003 + 0.005 / (1 + np.exp(-0.06 * (x - n / 2))), dtype=np.float64)

        reward_mask = rewards > 5.0
        processed_uids = np.nonzero(reward_mask)[0]
        processed_rewards = rewards[reward_mask]
        n = len(processed_rewards)

        if n == 0:
            bt.logging.warning("No miner is qualified to get incentive")
            return

        sort_idx = np.argsort(processed_rewards)
        probs = strip_sigmoid(np.arange(n, dtype=np.uint16), n)
        final_probs = probs[np.argsort(sort_idx)]
        processed_weights = final_probs / np.sum(final_probs)

        bt.logging.debug(f"Uids: {processed_uids}")
        bt.logging.debug(f"Weights: {processed_weights}")

        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(uids=processed_uids, weights=processed_weights)

        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
        )
        if result:
            bt.logging.info("Weights set on chain successfully!")
        else:
            bt.logging.error(f"Setting weights failed with {msg}")
