import base64

from common.protocol import SubmitResults
from pydantic import BaseModel


class StoredData(BaseModel):
    assets: str  # Generated 3D assets (base64 encoded).
    miner: str  # Miner hotkey.
    validator: str  # Validator hotkey.
    prompt: str  # Prompt used for generation.

    submit_time: int  # time.time_ns() returned from miner.
    signature: str  # Miner signature: b64encode(sign(f'{submit_time}{prompt}{validator.hotkey}{miner.hotkey}'))

    def to_base64(self) -> str:
        return base64.b64encode(self.json().encode(encoding="utf-8")).decode(encoding="utf-8")

    @staticmethod
    def from_base64(serialized: str) -> "StoredData":
        return StoredData.parse_raw(base64.b64decode(serialized.encode(encoding="utf-8")).decode(encoding="utf-8"))

    @staticmethod
    def from_results(synapse: SubmitResults) -> "StoredData":
        if synapse.results is None or synapse.task is None:
            raise RuntimeError("Unexpected behaviour. Results and task must be set")
        return StoredData(
            assets=synapse.results,
            miner=synapse.dendrite.hotkey,
            validator=synapse.axon.hotkey,
            prompt=synapse.task.prompt,
            submit_time=synapse.submit_time,
            signature=synapse.signature,
        )
