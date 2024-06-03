import bittensor as bt
import uvicorn
from bittensor.axon import FastAPIThreadedServer
from fastapi import FastAPI

from validator.api.api_key_manager import ApiKeyManager
from validator.api.router import router
from validator.api.task_registry import TaskRegistry


class PublicAPIServer:
    def __init__(self, config: bt.config, task_registry: TaskRegistry) -> None:
        self.port = config.public_api.server_port
        self.sync_api_keys_interval = config.public_api.sync_api_keys_interval

        self.started = False

        self.task_registry = task_registry
        self.api_key_manager = ApiKeyManager(config.neuron.full_path / "api_keys.db")

        self.app = FastAPI()
        self.app.state.task_registry = self.task_registry
        self.app.state.api_key_manager = self.api_key_manager

        log_level = "trace" if bt.logging.__trace_on__ else "critical"
        self.fast_config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.port, log_level=log_level, ws_max_size=104857600  # noqa: S104
        )
        self.fast_server = FastAPIThreadedServer(config=self.fast_config)
        self.app.include_router(router)

        bt.logging.info(f"Public server created. Port: {self.port}")

    def __del__(self) -> None:
        self.stop()

    def start(self) -> None:
        self.fast_server.start()
        self.started = True

        bt.logging.info(f"Public server started. Port: {self.port}")

        self.api_key_manager.start_periodic_sync(self.sync_api_keys_interval)

    def stop(self) -> None:
        self.fast_server.stop()
        self.started = False

        bt.logging.info("Public server stopped.")
