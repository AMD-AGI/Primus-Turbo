"""Standalone gfx950 FlyDSL forward prototype for D_QK != D_V (DeepSeek-V3 MLA 192/128)."""

from .ops import mla_fwd_sbhd

__all__ = ["mla_fwd_sbhd"]
