import bittensor as bt
import pydantic


class StoreUser(bt.Synapse):
    encrypted_data: str  # base64 encoded string of encrypted data (bytes)
    encryption_payload: str  # encrypted json serialized bytestring of encryption params

    data_hash: str | None = None  # Miner storage lookup key
    ttl: int | None = None  # time to live (in seconds)

    required_hash_fields: list[str] = pydantic.Field(
        ["encrypted_data", "encryption_payload"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )


class RetrieveUser(bt.Synapse):
    data_hash: str  # Miner storage lookup key

    # Fetched data to return along with AES payload in base64 encoding
    encrypted_data: str | None = None
    encryption_payload: str | None = None

    required_hash_fields: list[str] = pydantic.Field(
        ["data_hash"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )
