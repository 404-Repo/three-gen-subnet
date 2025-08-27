import uuid

from pydantic import BaseModel, Field


class ValidatorTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique id."""
    prompt: str
    """Prompt to use for generation."""
