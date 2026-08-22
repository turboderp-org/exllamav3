from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen = True)
class MTPHotVocabConfig:
    """Configuration for the experimental EXL3 MTP vocabulary-subset fast path.

    ``blocks_path`` loads packed 16-token EXL3 block IDs produced by
    ``util/build_mtp_hot_blocks.py``.
    """

    blocks_path: str | Path | None = None
    embedding_dtype: str = "fp16"
    validate_full_head: bool = False

    def __post_init__(self):
        if self.validate_full_head and not self.enabled:
            raise ValueError("validate_full_head requires a hot-vocabulary selection")
        dtype = self.embedding_dtype.lower()
        if dtype not in ("fp16", "float16", "half", "fp8", "float8", "e4m3"):
            raise ValueError(f"Unsupported MTP hot-embedding dtype: {self.embedding_dtype!r}")

    @property
    def enabled(self) -> bool:
        return bool(self.blocks_path)

    @classmethod
    def from_env(cls) -> MTPHotVocabConfig:
        """Build configuration from environment variables for server integrations."""
        return cls(
            blocks_path = os.environ.get("EXL3_MTP_HOT_BLOCKS"),
            embedding_dtype = os.environ.get("EXL3_MTP_HOT_EMBED_DTYPE", "fp16"),
            validate_full_head = os.environ.get("EXL3_MTP_VALIDATE_SUBHEAD", "0") != "0",
        )
