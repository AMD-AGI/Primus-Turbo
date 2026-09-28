"""e2e shim (P2 only): the two names the Primus converter imports. FP8 attention is off in the
e2e configs (enable_attention_float8: false), so they are never instantiated; if they are,
E2EAttention refuses fp8_config loudly."""
import enum


class ScalingGranularity(enum.Enum):
    TENSORWISE = "tensorwise"
    ROWWISE = "rowwise"
    BLOCKWISE = "blockwise"
    MX_BLOCKWISE = "mx_blockwise"


class Float8QuantConfig:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
