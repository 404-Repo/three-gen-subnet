from typing import Any

import numpy as np


def sigmoid(x: np.ndarray) -> Any:
    """
    Apply the sigmoid function element-wise to the input array.

    Args:
    x (np.ndarray): Input array.

    Returns:
    np.ndarray: The output array with the sigmoid function applied element-wise.
    """
    return 1 / (1 + np.exp(-x))
