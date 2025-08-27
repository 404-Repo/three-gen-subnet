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

from validator.api.public_api_server import PublicAPIServer
from validator.duels.ratings import DuelRatings
from validator.metagraph_sync import MetagraphSynchronizer
from validator.miner_data import MinerData
from validator.task_manager.task_manager import TaskManager
from validator.telemetry import Telemetry
from validator.validation_service import ValidationResponse, ValidationService
from validator.version import VALIDATOR_VERSION


NEURONS_LIMIT = 256


class Validator:
    uid: int
    """Each validator gets a unique identity (UID) in the network for differentiation."""
    config: bt.config
    """Copy of the original config including bittensor config."""
    wallet: bt.wallet
    """The wallet holds the cryptographic key pairs for the validator."""
    subtensor: bt.subtensor
    """The subtensor is the connection to the Bittensor blockchain."""
    metagraph: bt.metagraph
    """The metagraph holds the state of the network, letting us know about other validators and miners."""
    metagraph_sync: MetagraphSynchronizer
    """Synchronizes state of the bittensor network."""
    axon: bt.axon
    """Validator's axon for external connections in bittensor subnet."""
    task_manager: TaskManager
    """Task manager for fetching tasks from the network."""
    miners: list[MinerData]
    """List of miners."""
    ratings: DuelRatings
    """Holds miners duel ratings."""
    updater: AutoUpdater
    """Validator auto-updates."""
    public_server: PublicAPIServer | None
    """FastApi server that serves the public API endpoint for validator.
    Clients send prompts to this server in order to get 3D assets."""
    telemetry: Telemetry
    """Responsible for sending validator metrics to the metrics push gateway."""
    validation_service: ValidationService
    """Service for validating miner's results."""

    def __init__(
        self,
        *,
        config: bt.config,
        task_manager: TaskManager,
        validation_service: ValidationService,
        ratings: DuelRatings,
        wallet: bt.wallet | None = None,
        subtensor: bt.subtensor | None = None,
    ) -> None:
        self.task_manager = task_manager
        self.validation_service = validation_service

        self.config: bt.config = copy.deepcopy(config)
        create_neuron_dir(self.config)
        bt.logging.set_config(config=self.config.logging)
        bt.logging.info(f"Starting validator with config: {config}")

        self.wallet = bt.wallet(config=self.config) if wallet is None else wallet
        bt.logging.info(f"Wallet: {self.wallet}")
        self.subtensor = bt.subtensor(config=self.config) if subtensor is None else subtensor
        bt.logging.info(f"Subtensor: {self.subtensor}")
        self.public_server = PublicAPIServer(config=self.config) if self.config.public_api.enabled else None
        self.updater = AutoUpdater(
            disabled=self.config.neuron.auto_update_disabled,
            interval=self.config.neuron.auto_update_interval,
            local_version=VALIDATOR_VERSION,
        )
        self._check_validator_registered()
        self._init_miners()
        self._init_ratings(ratings=ratings)
        self._init_metagraph()
        self._init_axon()
        self.telemetry = Telemetry(self.miners, self.wallet, self.metagraph, self.config)
        if not self._is_enough_stake_to_set_weights():
            bt.logging.warning(
                f"You need stake "
                f"of t{self.config.neuron.min_stake_to_set_weights} to set weights. "
                f"You have t{self.metagraph.S[self.uid]}"
            )

    def _init_miners(self) -> None:
        # Try to load them from the file if it exists or create new ones.
        # File is used to preserve validator state in between restarts because miner can start its job
        # before restart and finish it after restart.
        self.miners = [MinerData(uid=x) for x in range(NEURONS_LIMIT)]

        path = self.config.neuron.full_path / "state.txt"
        if not path.exists():
            bt.logging.warning("No saved state found")
            return

        try:
            with path.open("r") as f:
                content = f.read()
            self.miners = Validator.State.model_validate_json(content).miners
        except Exception as e:
            bt.logging.exception(f"Failed to load the miners state: {e}")
        bt.logging.info("Miners initialized.")

    def _init_ratings(self, ratings: DuelRatings) -> None:
        self.ratings = ratings
        self.ratings.load_ratings(full_path=self.config.neuron.full_path)

    def _check_validator_registered(self) -> None:
        # Check if the validator is registered on the network.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            raise RuntimeError(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again."
            )
        bt.logging.info("registered on the network.")

    def _init_metagraph(self) -> None:
        # Initialize the metagraph, metagraph synchronizer and synchronizes with it first time.
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
        self.metagraph_sync.sync(self.miners, self.ratings)
        bt.logging.info(f"Metagraph: {self.metagraph}")
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet {self.config.netuid} with uid {self.uid} "
            f"using network: {self.subtensor.chain_endpoint}"
        )

    def _init_axon(self) -> None:
        # Initialize the axon and attaches the forward functions to it.
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

    async def pull_task(self, synapse: PullTask) -> PullTask:
        """Returns a new task to the miner if it is allowed."""

        miner_uid = self._check_miner_registered(synapse=synapse)
        if miner_uid is None:
            return synapse
        miner = self.miners[miner_uid]
        if self._check_miner_on_cooldown(synapse=synapse, miner=miner):
            return synapse
        if self._check_miner_has_assigned_task(miner=miner, synapse=synapse):
            return synapse

        is_strong_miner = self.metagraph_sync.is_strong_miner(miner_uid)

        task = await self.task_manager.get_next_task(
            miner_uid=miner_uid,
            is_strong_miner=is_strong_miner,
            metagraph=self.metagraph,
        )
        synapse.task = Task(id=task.id, prompt=task.prompt)
        miner.assign_task(synapse.task)
        synapse.validation_threshold = self.config.generation.quality_threshold
        synapse.throttle_period = self.config.generation.throttle_period
        return synapse

    async def submit_results(self, synapse: SubmitResults) -> SubmitResults:
        """Callback that is called when a miner submits results (generated 3D asset)."""
        miner_uid = self._check_miner_registered(synapse=synapse)
        if miner_uid is None:
            return synapse
        miner = self.miners[miner_uid]
        if not self._check_miner_signature(synapse=synapse, miner=miner):
            return await self._process_task_failure(
                synapse=synapse, miner=miner, cooldown_penalty=self.config.generation.cooldown_penalty
            )
        miner.cooldown_violations = max(0, miner.cooldown_violations - 1)
        if miner.assigned_task != synapse.task:
            bt.logging.warning(f"[{miner_uid}] submitted results for the wrong task")
            return self._add_feedback_and_strip(synapse, miner)
        if synapse.results == "":
            bt.logging.debug(f"[{miner_uid}] submitted empty results ({synapse.task.id} | {synapse.task.prompt[:100]})")
            return await self._process_task_failure(synapse=synapse, miner=miner, cooldown_penalty=0)

        return await self._validate_results(synapse=synapse, miner=miner, miner_uid=miner_uid)

    async def _validate_results(self, *, synapse: SubmitResults, miner: MinerData, miner_uid: int) -> SubmitResults:
        if miner.validation_locked_until > time.time():
            # Prevents validation spam when miners resubmit due to slow processing
            bt.logging.warning(
                f"[{miner_uid}] submitted results while validation lock. Prompt: ({synapse.task.prompt[:100]})"
            )
            return self._add_feedback_and_strip(synapse, miner)
        else:
            miner.validation_locked_until = int(time.time()) + self.config.validation.validation_lock_duration

        grid_preview_needed = self.task_manager.grid_preview_needed(synapse=synapse)

        validation_res = await self.validation_service.validate(
            synapse=synapse,
            neuron_uid=miner.uid,
            generate_single_preview=self.config.storage.enabled,
            generate_grid_preview=grid_preview_needed,
        )
        if validation_res is None:
            bt.logging.error(f"[{miner_uid}]: validation failed ({synapse.task.prompt[:100]})")
            return await self._process_task_failure(
                synapse=synapse, miner=miner, cooldown_penalty=0, validation_failed=True
            )
        bt.logging.debug(
            f"[{miner_uid}] submitted results with the score: {validation_res.score} ({synapse.task.prompt[:100]})"
        )
        if (
            miner.last_submit_time + self.config.generation.task_cooldown - self.config.generation.throttle_period
            > time.time()
        ):
            # Prevent double rewards when miners resubmit results due to slow validation
            bt.logging.warning(
                f"[{miner_uid}] resubmitted too quickly: {time.time() - miner.last_submit_time:.1f}s "
                f"after last submit. Prompt: {synapse.task.prompt[:100]}"
            )
            return self._add_feedback_and_strip(synapse, miner)

        if validation_res.score < self.config.generation.quality_threshold:
            return await self._process_task_failure(
                synapse=synapse,
                miner=miner,
                score=validation_res.score,
                cooldown_penalty=self.config.generation.cooldown_penalty,
                validation_failed=False,
            )

        return await self._process_valid_result(
            synapse=synapse, miner=miner, validation_res=validation_res, miner_uid=miner_uid
        )

    async def _process_valid_result(
        self, *, synapse: SubmitResults, miner: MinerData, validation_res: ValidationResponse, miner_uid: int
    ) -> SubmitResults:
        # Add miner observation.
        current_time = int(time.time())
        fidelity_score = min(1.0, validation_res.score)
        delivery_time = (int(time.time()) - miner.assignment_time) if miner.assignment_time else 0
        miner.reset_task(
            throttle_period=self.config.generation.throttle_period, cooldown=self.config.generation.task_cooldown
        )
        miner.add_observation(
            task_finish_time=current_time,
            fidelity_score=fidelity_score,
            moving_average_alpha=self.config.validation.moving_average_alpha,
        )
        miner.last_submit_time = int(time.time())

        # Add task metrics.
        self.telemetry.add_task_metrics(
            miner_hotkey=synapse.dendrite.hotkey,
            miner_coldkey=self.metagraph.axons[miner.uid].coldkey,
            score=validation_res.score,
            delivery_time=delivery_time,
            size=len(synapse.results),
        )

        # Submit task results.
        synapse_copy = synapse.model_copy()
        asyncio.create_task(
            self.task_manager.submit_result(
                synapse=synapse_copy, validation_res=validation_res, miner_uid=miner_uid, metagraph=self.metagraph
            )
        )
        return self._add_feedback_and_strip(synapse, miner, current_time=current_time, fidelity_score=fidelity_score)

    async def _process_task_failure(
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

        asyncio.create_task(
            self.task_manager.fail_task(
                task_id=synapse.task.id,
                task_prompt=synapse.task.prompt,
                hotkey=synapse.dendrite.hotkey,
                miner_uid=miner.uid,
                metagraph=self.metagraph,
            )
        )

        self.telemetry.add_task_metrics(
            miner_hotkey=synapse.dendrite.hotkey,
            miner_coldkey=self.metagraph.axons[miner.uid].coldkey,
            score=score,
            delivery_time=delivery_time,
            size=len(synapse.results),
        )

        return self._add_feedback_and_strip(synapse, miner, validation_failed=validation_failed)

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
        current_rating = self.ratings.get_miner_reward_rating(miner.uid)
        reward = miner.calculate_reward(current_time=current_time, rating=current_rating)
        synapse.feedback = Feedback(
            validation_failed=validation_failed,
            task_fidelity_score=fidelity_score,
            average_fidelity_score=miner.fidelity_score,
            generations_within_the_window=len(miner.observations),
            current_duel_rating=current_rating,
            current_miner_reward=reward,
        )
        synapse.cooldown_until = miner.cooldown_until
        return synapse

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
        synapse.validation_version = await self.validation_service.version()
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
                self.ratings.save_ratings(full_path=self.config.neuron.full_path)
                self.metagraph_sync.sync(self.miners, self.ratings)
                self._set_weights()

            if await self.updater.should_update():
                self.save_state()
                self.ratings.save_ratings(full_path=self.config.neuron.full_path)
                await self.updater.update()
                break

    class State(BaseModel):
        miners: list[MinerData]

    def save_state(self) -> None:
        try:
            path = self.config.neuron.full_path / "state.txt"
            with path.open("w") as f:
                f.write(Validator.State(miners=self.miners).model_dump_json())
        except Exception as e:
            bt.logging.exception(f"Validator state saving failed with {e}")

    def _set_weights(self) -> None:
        if not self._is_enough_stake_to_set_weights():
            return

        if self.metagraph.last_update[self.uid] + self.config.neuron.weight_set_interval > self.metagraph.block:
            return

        current_time = int(time.time())
        reward_ratings = self.ratings.get_reward_ratings()
        bt.logging.debug(f"Ratings: {reward_ratings}")
        rewards = np.array(
            [
                miner.calculate_reward(current_time=current_time, rating=rating)
                for miner, rating in zip(self.miners, reward_ratings, strict=False)
            ]
        )
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

    def _check_miner_registered(self, *, synapse: SubmitResults | PullTask) -> int | None:
        miner_uid = self._get_neuron_uid(synapse.dendrite.hotkey)
        if miner_uid is None:
            synapse.cooldown_until = int(time.time()) + 3600
            bt.logging.debug(f"Miner {synapse.dendrite.hotkey} ({synapse.dendrite.ip}) is not registered")
        return miner_uid

    def _check_miner_on_cooldown(self, *, synapse: PullTask, miner: MinerData) -> bool:
        if miner.is_on_cooldown():
            miner.cooldown_violations += 1
            bt.logging.debug(
                f"[{miner.uid}] asked for a new task while on a cooldown "
                f"({miner.cooldown_left()} sec left). "
                f"Total violations: {miner.cooldown_violations}"
            )
            if miner.cooldown_violations > self.config.generation.cooldown_violations_threshold:
                miner.cooldown_until += self.config.generation.cooldown_violation_penalty
                bt.logging.debug(f"[{miner.uid}] Cooldown penalty added.")

            synapse.cooldown_violations = miner.cooldown_violations
            synapse.cooldown_until = miner.cooldown_until
            return True
        return False

    def _check_miner_has_assigned_task(self, *, miner: MinerData, synapse: PullTask) -> bool:
        if miner.assigned_task is not None:
            synapse.task = miner.assigned_task
            synapse.validation_threshold = self.config.generation.quality_threshold
            synapse.throttle_period = self.config.generation.throttle_period
            bt.logging.debug(
                f"[{miner.uid}] asked for a new task while having assigned task ({synapse.task.prompt[:100]})."
            )
            return True
        return False

    def _check_miner_signature(self, *, synapse: SubmitResults, miner: MinerData) -> bool:
        keypair = Keypair(ss58_address=synapse.dendrite.hotkey)
        message = (
            f"{MINER_LICENSE_CONSENT_DECLARATION}"
            f"{synapse.submit_time}{synapse.task.prompt}{synapse.axon.hotkey}{synapse.dendrite.hotkey}"
        )
        encoded_signature = pybase64.b64decode(synapse.signature.encode(encoding="utf-8"), validate=True)
        if not keypair.verify(message, encoded_signature):
            bt.logging.warning(f"[{miner.uid}] submitted results with invalid signature")
            return False
        return True
