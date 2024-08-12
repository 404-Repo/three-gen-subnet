from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any

import numpy as np


class BaseLoader(ABC):
    """
    BaseLoader is an abstract base class for loading data from various sources.

    Subclasses must implement methods to load data from a file or from a buffer.
    """

    @abstractmethod
    def from_file(self, file_name: str, file_path: str) -> dict[str, Any]:
        """
        Load data from a file.

        Args:
        file_name (str): The name of the file to load.
        file_path (str): The path to the file.

        Returns:
        Dict: A dictionary containing the loaded data.
        """
        pass

    @abstractmethod
    def from_buffer(self, buffer: BytesIO) -> dict[str, Any]:
        """
        Load data from a buffer.

        Args:
        buffer (io.BytesIO): A buffer containing the data to load.

        Returns:
        Dict: A dictionary containing the loaded data.
        """
        pass


class BaseWriter(ABC):
    @abstractmethod
    def to_buffer(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        features_dc: np.ndarray,
        features_rest: np.ndarray,
        opacities: np.ndarray,
        scale: np.ndarray,
        rotation: np.ndarray,
        sh_degree: int,
    ) -> BytesIO:
        pass

    @abstractmethod
    def to_file(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        features_dc: np.ndarray,
        features_rest: np.ndarray,
        opacities: np.ndarray,
        scale: np.ndarray,
        rotation: np.ndarray,
        sh_degree: int,
        h5file_name: str,
        h5file_path: str,
    ) -> None:
        pass
