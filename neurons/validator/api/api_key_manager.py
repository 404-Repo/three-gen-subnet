import sqlite3
import threading
import time
from pathlib import Path

import bittensor as bt
from pydantic import BaseModel


class ApiKeyData(BaseModel):
    name: str
    """Descriptive name"""
    api_key: str
    """Unique API-Key to use."""
    max_requests: int
    """The maximum number of requests that are allowed for a single key within the specified time period."""
    period: int
    """The time period, in seconds, for which the request limit (specified by max_requests) applies."""


class ApiKeyManager:
    def __init__(self, db_file: Path) -> None:
        self._db_file: Path = db_file
        self._api_keys: dict[str, ApiKeyData] = {}  # api_key -> ApiKeyData

        self._requests: dict[str, tuple[int, float]] = {}  # api_key -> (request count, first request time)

        self._setup_database()
        self._sync()

    def _setup_database(self) -> None:
        if self._db_file.exists():
            return

        try:
            with sqlite3.connect(self._db_file.as_posix()) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE api_keys (
                        api_key TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        max_requests INTEGER NOT NULL,
                        period INTEGER NOT NULL
                    )
                """
                )
                conn.commit()
        except Exception as e:
            bt.logging.exception(f"Failed to initialize sqlite database for api-key management: {e}")

    def _sync(self) -> None:
        """Synchronously (blocking) updates the api-keys."""

        try:
            bt.logging.info("Synchronizing api-keys.")

            with sqlite3.connect(self._db_file.as_posix()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT api_key, name, max_requests, period FROM api_keys")
                rows = cursor.fetchall()

                self._api_keys = {
                    row[0]: ApiKeyData(name=row[1], api_key=row[0], max_requests=row[2], period=row[3]) for row in rows
                }

            bt.logging.info(f"Api-keys synchronized. {len(self._api_keys)} registered")
        except Exception as e:
            bt.logging.exception(f"Failed to update sqlite database for api-key management: {e}")

    def is_registered(self, api_key: str) -> bool:
        return api_key in self._api_keys

    def is_allowed(self, api_key: str) -> bool:
        if api_key not in self._api_keys:
            return False

        api_data = self._api_keys[api_key]

        request_count, first_request_time = self._requests.get(api_key, (0, 0.0))
        current_time = time.time()

        if current_time - first_request_time > api_data.period:
            self._requests[api_key] = (1, current_time)
            return True

        if request_count >= api_data.max_requests:
            return False

        self._requests[api_key] = (request_count + 1, first_request_time)
        return True

    def get_name(self, api_key: str) -> str:
        return self._api_keys[api_key].name

    def start_periodic_sync(self, interval: int) -> None:
        def sync_job() -> None:
            while True:
                threading.Event().wait(interval)
                self._sync()

        thread = threading.Thread(target=sync_job, daemon=True)
        thread.start()
