# finetune_forge/schemas/dataset.py
"""Dataset-format models and schema constants.

``DatasetInfo`` lives in ``state.py`` alongside the pipeline state and is
re-exported here. This module also holds the canonical required-key sets used
by the data processor to detect and validate dataset formats.
"""

from __future__ import annotations

from finetune_forge.schemas.state import DatasetInfo, DatasetFormat

# Canonical column requirements per supported training format.
REQUIRED_SFT_KEYS = frozenset({"instruction", "output"})
REQUIRED_DPO_KEYS = frozenset({"prompt", "chosen", "rejected"})
REQUIRED_REWARD_KEYS = frozenset({"prompt", "chosen", "rejected"})
REQUIRED_PRETRAIN_KEYS = frozenset({"text"})

# Accepted aliases that the processor will normalise into the alpaca schema.
SFT_INSTRUCTION_ALIASES = ("instruction", "question", "input")
SFT_OUTPUT_ALIASES = ("output", "answer")
SFT_CONTEXT_ALIASES = ("context", "system")

__all__ = [
    "DatasetInfo",
    "DatasetFormat",
    "REQUIRED_SFT_KEYS",
    "REQUIRED_DPO_KEYS",
    "REQUIRED_REWARD_KEYS",
    "REQUIRED_PRETRAIN_KEYS",
    "SFT_INSTRUCTION_ALIASES",
    "SFT_OUTPUT_ALIASES",
    "SFT_CONTEXT_ALIASES",
]
