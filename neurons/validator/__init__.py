import asyncio
import copy
import time
from typing import Tuple  # noqa: UP035

import bittensor as bt
import torch
from bittensor.utils import weight_utils
from common import create_neuron_dir
from common.protocol import Feedback, Generate, PollResults, PollTask, SubmitResults, Task

from validator.dataset import Dataset
from validator.fidelity_check import validate
from validator.metagraph_sync import MetagraphSynchronizer
from validator.miner_data import MinerData
from validator.rate_limiter import RateLimiter
from validator.task_registry import TaskRegistry
from validator.validator_state import ValidatorState


NEURONS_LIMIT = 256
SCORE_ON_VALIDATION_FAILURE = 0.6


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
    state: ValidatorState
    """Simple wrapper to encapsulate validator state save and load."""
    public_api_limiter: RateLimiter

    def __init__(self, config: bt.config) -> None:
        # TODO: dynamic configuration
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
            self.metagraph, self.subtensor, self.config.neuron.sync_interval, self.config.neuron.log_info_interval
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
            forward_fn=self.poll_task,
            blacklist_fn=self.blacklist_polling_task,
        ).attach(
            forward_fn=self.submit_results,
            blacklist_fn=self.blacklist_submitting_results,
        )

        if self.config.public_api.enabled:
            if self.config.public_api.whitelist_disabled:
                bt.logging.warning("Security risk. Whitelist for public API disabled.")

            self.axon.attach(
                forward_fn=self.generate,
                blacklist_fn=self.blacklist_generation,
            ).attach(
                forward_fn=self.poll_results,
                blacklist_fn=self.blacklist_polling_results,
            )

        bt.logging.info(f"Axon created: {self.axon}")

        self.dataset = Dataset(self.config.dataset.path)

        self.task_registry = TaskRegistry(
            copies=self.config.public_api.copies,
            wait_after_first_copy=self.config.public_api.wait_after_first_copy,
            task_timeout=self.config.generation.task_timeout,
            poll_interval=self.config.public_api.poll_interval,
        )

        self.public_api_limiter = RateLimiter(
            max_requests=self.config.public_api.rate_limit.requests, period=self.config.public_api.rate_limit.period
        )

        self.miners = [MinerData(uid=x) for x in range(NEURONS_LIMIT)]
        self.state = ValidatorState(miners=self.miners)
        self.state.load(self.config.neuron.full_path / "state.pt")

    def generate(self, synapse: Generate) -> Generate:
        synapse.task_id = self.task_registry.add_task(synapse.prompt, synapse.dendrite.hotkey)
        synapse.poll_interval = self.config.public_api.poll_interval
        return synapse

    def blacklist_generation(self, synapse: Generate) -> Tuple[bool, str]:  # noqa: UP006, UP035
        if not self._is_wallet_whitelisted(synapse.dendrite.hotkey):
            return True, "Not whitelisted"

        if not self.public_api_limiter.is_allowed(synapse.dendrite.hotkey):
            bt.logging.info(f"{synapse.dendrite.hotkey} hit the rate limit")
            return True, "Rate limit"

        return False, ""

    def _is_wallet_whitelisted(self, hotkey: str) -> bool:
        # TODO: move whitelist to the config file + http endpoint
        whitelist = ["5DwHD8Ja9aWGhB5nbmZqCCs9GNMpzxhBcZjnhcY73Sd75qe7"]
        if not self.config.public_api.whitelist_disabled and hotkey not in whitelist:
            bt.logging.warning(f"{hotkey} is not white-listed for the public API")
            return False

        return True

    def poll_results(self, synapse: PollResults) -> PollResults:
        synapse.status, synapse.results = self.task_registry.get_task_status(synapse.task_id, synapse.dendrite.hotkey)
        return synapse

    def blacklist_polling_results(self, synapse: PollResults) -> Tuple[bool, str]:  # noqa: UP006, UP035
        if not self._is_wallet_whitelisted(synapse.dendrite.hotkey):
            return True, "Not whitelisted"

        return False, ""

    def poll_task(self, synapse: PollTask) -> PollTask:
        uid = self._get_neuron_uid(synapse.dendrite.hotkey)
        if uid is None:
            bt.logging.error("Unexpected behaviour, unknown neuron after blacklist")
            return synapse

        miner = self.miners[uid]

        # TODO: add strong miners list
        organic_task = self.task_registry.get_next_task(synapse.dendrite.hotkey, is_strong_miner=True)
        if organic_task is not None:
            task = Task(id=organic_task.id, prompt=organic_task.prompt)
            bt.logging.trace(f"[{uid}] pulls organic task ({task.prompt} | {task.id})")
        else:
            task = Task(prompt=self.dataset.get_random_prompt())
            # task = Task(prompt="tourmaline tassel earring")
            bt.logging.trace(f"[{uid}] pulls synthetic task ({task.prompt} | {task.id})")

        miner.assign_task(task)

        synapse.task = task
        return synapse

    def blacklist_polling_task(self, synapse: PollTask) -> Tuple[bool, str]:  # noqa: UP006, UP035
        uid = self._get_neuron_uid(synapse.dendrite.hotkey)
        if uid is None:
            bt.logging.trace(f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"

        miner = self.miners[uid]
        if miner.is_task_expired(self.config.generation.task_timeout):
            bt.logging.debug(f"[{uid}] asked for a new task while having expired task")
            miner.reset_task()

        if miner.assigned_task is not None:
            bt.logging.trace(f"[{uid}] asked for a new task while having assigned task")
            return True, "Waiting for the previous task"

        if miner.is_on_cooldown(self.config.generation.task_cooldown):
            bt.logging.trace(f"[{uid}] asked for a new task while on a cooldown")
            return True, "Cooldown from the previous task"

        return False, ""

    async def submit_results(self, synapse: SubmitResults) -> SubmitResults:
        uid = self._get_neuron_uid(synapse.dendrite.hotkey)
        if uid is None:
            bt.logging.error("Unexpected behaviour, unknown neuron after blacklist")
            return synapse

        miner = self.miners[uid]

        if synapse.task is None:
            bt.logging.warning(f"[{uid}] submitted results with no original task")
            return self._add_feedback(synapse, 0.0, miner.fidelity_score)

        if miner.assigned_task != synapse.task:
            bt.logging.warning(f"[{uid}] submitted results for the wrong task")
            return self._add_feedback(synapse, 0.0, miner.fidelity_score)

        if synapse.results == "":
            bt.logging.warning(f"[{uid}] submitted empty results")
            return self._add_feedback(synapse, 0.0, miner.fidelity_score)

        # TODO: better handle concurrent validation
        validation_score = await validate(self.config.validation.endpoint, synapse.task.prompt, synapse.results)
        if validation_score is None:
            validation_score = SCORE_ON_VALIDATION_FAILURE
        fidelity_score = self._get_fidelity_score(validation_score)

        miner.reset_task()

        if fidelity_score == 0:
            bt.logging.warning(f"[{uid}] submitted results with low fidelity score. Skipping the task")
            return self._add_feedback(synapse, 0.0, miner.fidelity_score)

        miner.add_observation(
            task_finish_time=int(time.time()),
            fidelity_score=fidelity_score,
            moving_average_alpha=self.config.neuron.moving_average_alpha,
        )

        self.task_registry.complete_task(synapse.task.id, synapse.dendrite.hotkey, synapse.results, validation_score)

        return self._add_feedback(synapse, fidelity_score, miner.fidelity_score)

    def _add_feedback(self, synapse: SubmitResults, fidelity_score: float, avg_fidelity_score: float) -> SubmitResults:
        synapse.feedback = Feedback(fidelity_score=fidelity_score, avg_fidelity_score=avg_fidelity_score)
        synapse.cooldown_until = int(time.time()) + self.config.generation.task_cooldown
        return synapse

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
            bt.logging.trace(f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"

        miner = self.miners[uid]
        if miner.assigned_task is None:
            bt.logging.trace(f"[{uid}] submitted results while having no task assigned")
            return True, "No task assigned"

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

    def _is_enough_stake_to_set_weights(self) -> bool:
        return bool(self.metagraph.S[self.uid].item() >= self.config.neuron.min_stake_to_set_weights)

    def _get_neuron_uid(self, hotkey: str) -> int | None:
        for neuron in self.metagraph.neurons:
            if neuron.hotkey == hotkey:
                return int(neuron.uid)

        return None

    async def run(self) -> None:
        # TODO: rollback to threads

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
            self.metagraph_sync.sync()
            self._set_weights()

    def _set_weights(self) -> None:
        if not self._is_enough_stake_to_set_weights():
            return

        if self.metagraph.last_update[self.uid] + self.config.neuron.weight_set_interval > self.metagraph.block:
            return

        current_time = int(time.time())
        rewards = torch.tensor([miner.calculate_reward(current_time) for miner in self.miners])

        bt.logging.debug(f"Rewards: {rewards}")

        raw_weights = torch.nn.functional.normalize(rewards, p=1, dim=0)

        bt.logging.debug(f"Normalized weights: {raw_weights}")

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
            version_key=1,
        )

        self.state.save(self.config.neuron.full_path / "state.pt")
