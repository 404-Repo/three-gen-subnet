from abc import ABC, abstractmethod
from io import BytesIO

from engine.data_structures import GaussianSplattingData


class BaseLoader(ABC):
    """
    BaseLoader is an abstract base class for loading data from various sources.
    Subclasses must implement methods to load data from a file or from a buffer.
    """

    @abstractmethod
    def from_file(self, file_name: str, file_path: str) -> GaussianSplattingData:
        """Load data from a file."""
        pass

    @abstractmethod
    def from_buffer(self, buffer: BytesIO) -> GaussianSplattingData:
        """Load data from a buffer"""
        pass
