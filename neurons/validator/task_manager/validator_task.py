from common.protocol import ProtocolTask
from pydantic import BaseModel, Field


class ValidatorTask(BaseModel):
    protocol: ProtocolTask = Field(discriminator="type")

    @property
    def id(self) -> str:
        return self.protocol.id

    @property
    def prompt(self) -> str:
        return self.protocol.prompt

    @property
    def log_id(self) -> str:
        return self.protocol.log_id
