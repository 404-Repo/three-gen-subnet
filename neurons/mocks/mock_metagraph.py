"""
Mock Metagraph module for testing purposes.

This module provides a MockMetagraph class that wraps around bt.metagraph
and provides additional functionality for mock testing scenarios.
"""

import bittensor as bt


class MockMetagraph:
    """Mock Metagraph wrapper around bt.metagraph with additional properties for testing."""

    def __init__(self, metagraph: bt.metagraph) -> None:
        """
        Initialize MockMetagraph with an existing bt.metagraph instance.
        """
        self._metagraph = metagraph
        self.hotkeys: list[str] = []
        self.coldkeys: list[str] = []
        self.S: list[int] = []
