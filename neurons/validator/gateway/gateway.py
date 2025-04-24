from pydantic import BaseModel


class Gateway(BaseModel):
    """Gateway is a node that is used by validator to get tasks and submit results.
    Clients send prompts to the gateway in order to get 3D-assets as a result."""

    node_id: int
    """Unique identifier of the gateway."""
    domain: str
    """Domain name of the gateway."""
    ip: str
    """IP address of the gateway."""
    name: str
    """Name of the gateway."""
    http_port: int
    """Port number of the gateway."""
    available_tasks: int
    """Number of available tasks."""
    last_task_acquisition: float
    """Timestamp of the last task acquisition by any validator."""
    latency: float | None = 100
    """Latency of the gateway. None means that offline or not calculated yet."""
    score: float = 0.0
    """Score of the gateway."""
    disabled: bool = False
    """Whether the gateway is disabled."""

    @property
    def url(self) -> str:
        """Returns the URL of the gateway."""
        return f"https://{self.domain}:{self.http_port}"

    def get_info(self) -> str:
        return (
            f"{self.url}: Score {self.score} | Disabled {self.disabled} "
            f"| Latency {self.latency} | Available tasks {self.available_tasks} "
            f"| Last task acquisition {self.last_task_acquisition}"
        )
