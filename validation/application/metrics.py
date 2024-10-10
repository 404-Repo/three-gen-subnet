import time

from loguru import logger
from pydantic import BaseModel, Field


class Metrics(BaseModel):
    validations: int = 0  # Total number of validations.
    accepted: int = 0  # Total number of 0.8+ results.

    last_minute: int = Field(default_factory=lambda: int(time.time() / 60))  # Last minute.

    last_minute_validations: int = 0  # Validations made during the last minute.
    validations_per_minute_ema: float = 180.0  # Average validations per minute.

    last_minute_accepted: int = 0  # 0.8+ results received during the last minute.
    accepted_per_minute_ema: float = 180.0  # Average 0.8+ results received per minute.

    def update(self, score: float) -> None:
        minute = int(time.time() / 60)
        if minute > self.last_minute:
            self.validations_per_minute_ema = self.validations_per_minute_ema * (0.9 ** (minute - self.last_minute))
            self.validations_per_minute_ema += 0.1 * self.last_minute_validations
            self.last_minute_validations = 0

            self.accepted_per_minute_ema = self.accepted_per_minute_ema * (0.9 ** (minute - self.last_minute))
            self.accepted_per_minute_ema += 0.1 * self.last_minute_accepted
            self.last_minute_accepted = 0

            self.last_minute = minute

        accepted = 1 if score > 0.8 else 0

        self.validations += 1
        self.accepted += accepted

        self.last_minute_validations += 1
        self.last_minute_accepted += accepted

        logger.info(
            f"Statistics: "
            f"{self.validations_per_minute_ema:.2f} validations / minute. "
            f"{self.accepted_per_minute_ema:.2f} accepted / minute"
        )
