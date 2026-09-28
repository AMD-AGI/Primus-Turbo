import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from impl import eager_attn_bwd  # noqa: E402

__all__ = ["eager_attn_bwd"]
