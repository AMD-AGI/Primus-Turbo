from typing import Any, Dict

from jax.extend.core import Primitive

IMPL_TABLE: Dict[Primitive, Any] = {}
ABSTRACT_EVAL_TABLE: Dict[Primitive, Any] = {}
LOWERING_TABLE: Dict[Primitive, Any] = {}

TRANSPOSE_TABLE: Dict[Primitive, Any] = {}
BATCHING_TABLE: Dict[Primitive, Any] = {}

# Import primitives to register them
from . import grouped_gemm, grouped_gemm_fp8, normalization, quantization  # noqa: F401
