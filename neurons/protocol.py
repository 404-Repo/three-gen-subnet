import typing

import bittensor as bt


class Task404(bt.Synapse):
    # Required request input, filled by sending dendrite caller.
    dummy_input: int

    # Optional request output, filled by receiving axon.
    dummy_output: typing.Optional[int] = None

    def deserialize(self) -> int:
        return self.dummy_output or 0
