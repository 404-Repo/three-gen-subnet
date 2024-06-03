from enum import Enum

from pydantic import BaseModel


class Auth(BaseModel):
    api_key: str


class PromptData(BaseModel):
    """
    Attributes:
        prompt (str): The prompt to be used for generating the 3D model.
        send_first_results (bool): Flag to indicate whether the first results
                                   should also be sent, in addition to the best results.
    """

    prompt: str
    send_first_results: bool = False


class TaskStatus(str, Enum):
    """
    Attributes:
        STARTED (str): Indicates the generation process has started.
        FIRST_RESULTS (str): Indicates the first intermediate results have been generated.
        BEST_RESULTS (str): Indicates all miners have completed the job and the best results are available.
        FAILED (str): Indicates the miners failed to provide acceptable results.
    """

    STARTED = "started"
    FIRST_RESULTS = "first_results"
    BEST_RESULTS = "best_results"


class TaskResults(BaseModel):
    """
    Attributes:
        score (float): The validation score of the generated assets.
        assets (str): The base64 encoded 3D assets in HDF5 format.
    """

    score: float
    assets: str | None


class TaskUpdate(BaseModel):
    """
    Attributes:
        status (TaskStatus): The current status of the task.
        results (Optional[str]): Detailed results associated with the current task status.
                                 This can be None if not applicable.
    """

    status: TaskStatus
    results: TaskResults | None = None
