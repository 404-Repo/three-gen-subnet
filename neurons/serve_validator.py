import asyncio

from validator.api import router
from validator.config import read_config
from validator.duels.duels_task_storage import DuelsTaskStorage
from validator.duels.ratings import DuelRatings
from validator.gateway.gateway_api import GatewayApi
from validator.gateway.gateway_manager import GatewayManager
from validator.gateway.gateway_scorer import GatewayScorer
from validator.task_manager.task_manager import TaskManager
from validator.task_manager.task_storage.organic_task_storage import OrganicTaskStorage
from validator.task_manager.task_storage.synthetic_asset_storage import SyntheticAssetStorage
from validator.task_manager.task_storage.synthetic_prompts_fetcher import SyntheticPromptsFetcher
from validator.task_manager.task_storage.synthetic_task_storage import SyntheticTaskStorage
from validator.validation_service import ValidationService
from validator.validator import Validator


async def main() -> None:
    config = read_config()

    synthetic_prompts_fetcher = SyntheticPromptsFetcher(
        config=config,
    )

    synthetic_asset_storage = SyntheticAssetStorage(
        enabled=config.storage.enabled,
        service_api_key=config.storage.service_api_key,
        endpoint_url=config.storage.endpoint_url,
        validation_score_threshold=config.storage.validation_score_threshold,
    )

    synthetic_task_storage = SyntheticTaskStorage(
        default_text_prompts_path=config.task.synthetic.default_text_prompts_path,
        default_image_prompts_path=config.task.synthetic.default_image_prompts_path,
        synthetic_prompts_fetcher=synthetic_prompts_fetcher,
        synthetic_asset_storage=synthetic_asset_storage,
        config=config,
    )

    validation_service = ValidationService(
        endpoints=config.validation.endpoints,
        validation_score_threshold=config.storage.validation_score_threshold,
    )

    duel_ratings = DuelRatings()

    duel_task_storage = DuelsTaskStorage(
        config=config,
        wallet=None,
        synthetic_task_storage=synthetic_task_storage,
        validation_service=validation_service,
        ratings=duel_ratings,
    )

    gateway_scorer = GatewayScorer()

    gateway_api = GatewayApi()

    gateway_manager = GatewayManager(
        gateway_scorer=gateway_scorer,
        gateway_api=gateway_api,
        gateway_info_server=config.task.gateway.bootstrap_gateway,
    )

    organic_task_storage = OrganicTaskStorage(
        gateway_manager=gateway_manager,
        config=config,
        ratings=duel_ratings,
    )

    task_manager = TaskManager(
        organic_task_storage=organic_task_storage,
        synthetic_task_storage=synthetic_task_storage,
        duel_task_storage=duel_task_storage,
        config=config,
    )

    # TODO: remove
    router.task_manager = task_manager

    neuron = Validator(
        config=config, task_manager=task_manager, validation_service=validation_service, ratings=duel_ratings
    )
    await synthetic_prompts_fetcher.start_fetching_prompts()
    asyncio.create_task(organic_task_storage.fetch_gateway_tasks_cron())
    await duel_task_storage.start_garbage_collection_cron()
    await duel_task_storage.start_judging_duels()
    await organic_task_storage.start_judging_results()
    await neuron.run()


if __name__ == "__main__":
    asyncio.run(main())
