from typing import Any, Dict

from jax.extend.core import Primitive

IMPL_TABLE: Dict[Primitive, Any] = {}
ABSTRACT_EVAL_TABLE: Dict[Primitive, Any] = {}
LOWERING_TABLE: Dict[Primitive, Any] = {}

TRANSPOSE_TABLE: Dict[Primitive, Any] = {}
BATCHING_TABLE: Dict[Primitive, Any] = {}
