import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from impl import asm_attn_bwd  # noqa: E402

__all__ = ["asm_attn_bwd"]
