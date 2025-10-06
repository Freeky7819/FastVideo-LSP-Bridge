# fastvideo/models/loader/fsdp_imports.py
# SPDX-License-Identifier: Apache-2.0
"""
Compatibility layer for Torch FSDP API across versions.

Torch >= 2.5 uses *Policy classes (MixedPrecisionPolicy, CPUOffloadPolicy).
Torch <= 2.4 uses MixedPrecision / CPUOffload.
We alias them to stable names that FastVideo code expects.
"""

from __future__ import annotations

try:
    # Torch >= 2.5
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDPModule,
        MixedPrecisionPolicy as _MixedPrecisionPolicy,
        CPUOffloadPolicy as _CPUOffloadPolicy,
        ShardingStrategy,
        BackwardPrefetch,
        StateDictType,
        FullStateDictConfig,
        LocalStateDictConfig,
    )
except Exception:
    # Torch <= 2.4
    from torch.distributed.fsdp import (  # type: ignore
        FullyShardedDataParallel as FSDPModule,
        MixedPrecision as _MixedPrecisionPolicy,
        CPUOffload as _CPUOffloadPolicy,
        ShardingStrategy,
        BackwardPrefetch,
        StateDictType,
        FullStateDictConfig,
        LocalStateDictConfig,
    )

# Public, stable names:
MixedPrecisionPolicy = _MixedPrecisionPolicy
CPUOffloadPolicy = _CPUOffloadPolicy

__all__ = [
    "FSDPModule",
    "MixedPrecisionPolicy",
    "CPUOffloadPolicy",
    "ShardingStrategy",
    "BackwardPrefetch",
    "StateDictType",
    "FullStateDictConfig",
    "LocalStateDictConfig",
]
