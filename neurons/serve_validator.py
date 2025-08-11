import asyncio

from validator.config import config
from validator.duels.duels_task_storage import duel_task_storage
from validator.duels.ratings import duel_ratings
from validator.task_manager.task_manager import task_manager
from validator.task_manager.task_storage.organic_task_storage import organic_task_storage
from validator.task_manager.task_storage.synthetic_task_storage import synthetic_task_storage
from validator.validation_service import validation_service
from validator.validator import Validator


async def main() -> None:
    neuron = Validator(
        config=config, task_manager=task_manager, validation_service=validation_service, ratings=duel_ratings
    )
    asyncio.create_task(synthetic_task_storage.fetch_synthetic_tasks_cron())
    asyncio.create_task(organic_task_storage.fetch_gateway_tasks_cron())
    await duel_task_storage.start_garbage_collection_cron()
    await duel_task_storage.start_judging_duels()
    await neuron.run()


if __name__ == "__main__":
    asyncio.run(main())
