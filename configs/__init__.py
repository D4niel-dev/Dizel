"""
configs — Dizel 1.6 Flash and Pro-family configurations.

Flash is the Preview M1 build; Pro is the planned 8-expert, top-4 tier.
Importing this package must stay cheap and must not import model code, so it
cannot create a cycle with ``model.architecture``.
"""

from .v16_flash import (
    DIZEL_16_FLASH_CONFIG,
    V16_FLASH_CONFIG,
    DizelV16FlashConfig,
    ModelConfigV16Flash,
    PretrainConfigV16Flash,
    TokenizerConfigV16Flash,
)
from .v16_pro import (
    DIZEL_16_PRO_CONFIG,
    V16_PRO_CONFIG,
    DizelV16ProConfig,
    ModelConfigV16Pro,
    PretrainConfigV16Pro,
    TokenizerConfigV16Pro,
)

__all__ = [
    "ModelConfigV16Flash",
    "TokenizerConfigV16Flash",
    "PretrainConfigV16Flash",
    "DizelV16FlashConfig",
    "DIZEL_16_FLASH_CONFIG",
    "V16_FLASH_CONFIG",
    "ModelConfigV16Pro",
    "TokenizerConfigV16Pro",
    "PretrainConfigV16Pro",
    "DizelV16ProConfig",
    "DIZEL_16_PRO_CONFIG",
    "V16_PRO_CONFIG",
]
