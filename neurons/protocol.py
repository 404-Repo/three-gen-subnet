from __future__ import annotations

import bittensor as bt


class TextTo3D(bt.Synapse):
    prompt_in: str = ""
    mesh_out: bytes | None = None

    def deserialize(self) -> TextTo3D:
        return self
