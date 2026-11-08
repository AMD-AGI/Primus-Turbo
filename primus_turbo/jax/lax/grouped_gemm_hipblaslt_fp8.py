"""FP8 (e4m3, per-tensor scales) grouped GEMM on hipBLASLt (gfx950) for JAX, with autodiff.

Same contract as flydsl_fp8.grouped_gemm_fp8_flydsl: bf16 in / bf16 out, a [M, K], b [G, K, N], group_lens [G];
operands are quantised per tensor to e4m3 (scale = 448 / amax), the kernels apply the dequantisation scales in
their epilogue. Needs the Primus-Turbo FFI handler `hipblaslt_grouped_gemm_fp8` and a hipBLASLt that has FP8
grouped kernels for gfx950 (see hipblaslt/tensile_gfx950_f8_gg.yaml).
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir, xla

from primus_turbo.jax._C import get_hipblaslt_grouped_gemm_workspace_size

FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.0

hipblaslt_grouped_gemm_fp8_p = Primitive("hipblaslt_grouped_gemm_fp8")
hipblaslt_grouped_gemm_fp8_p.multiple_results = True
hipblaslt_grouped_gemm_fp8_p.def_impl(partial(xla.apply_primitive, hipblaslt_grouped_gemm_fp8_p))


def _abstract_eval(a, b, a_scale, b_scale, group_lens, transA, transB):
    assert a.dtype == FP8 and b.dtype == FP8
    assert not (transA and transB)
    if transA:
        out = ShapedArray((group_lens.shape[0], a.shape[1], b.shape[1]), jnp.bfloat16)
    else:
        n = b.shape[1] if transB else b.shape[2]
        out = ShapedArray((a.shape[0], n), jnp.bfloat16)
    ws = ShapedArray((get_hipblaslt_grouped_gemm_workspace_size(group_lens.shape[0]),), jnp.uint8)
    return (out, ws)


hipblaslt_grouped_gemm_fp8_p.def_abstract_eval(_abstract_eval)
mlir.register_lowering(
    hipblaslt_grouped_gemm_fp8_p, jax.ffi.ffi_lowering("hipblaslt_grouped_gemm_fp8"), platform="rocm"
)


def _bind(a, b, sa, sb, group_lens, transA, transB):
    # kernel choice: PRIMUS_TURBO_HB_GG_TUNE=0 (default for fp8 recipes) uses hipBLASLt's heuristic (pruned library);
    # "subprocess"/"inproc" times the heuristic's candidates once per problem signature and caches the winner
    from primus_turbo.jax.lax.grouped_gemm_hipblaslt import _tune

    _tune(a, b, group_lens, transA, transB)
    out, _ = hipblaslt_grouped_gemm_fp8_p.bind(a, b, sa, sb, group_lens, transA=transA, transB=transB)
    return out


def quantize(x):
    """per-tensor e4m3: returns (x_q, scale_inv f32[1]) with x ~= x_q * scale_inv"""
    amax = jnp.max(jnp.abs(x)).astype(jnp.float32)
    scale = FP8_MAX / jnp.maximum(amax, 1e-12)
    return (x.astype(jnp.float32) * scale).astype(FP8), (1.0 / scale).reshape(1)


def _mask(x, group_lens):
    rows = jax.lax.broadcasted_iota(jnp.int32, (x.shape[0], 1), 0)
    return jnp.where(rows < jnp.sum(group_lens.astype(jnp.int32)), x, jnp.zeros_like(x))


@jax.custom_vjp
def grouped_gemm_fp8_hipblaslt(a, b, group_lens):
    return _fwd(a, b, group_lens)[0]


def _fwd(a, b, group_lens):
    a = _mask(a, group_lens)
    aq, sa = quantize(a)
    bq, sb = quantize(b)
    return _bind(aq, bq, sa, sb, group_lens, False, False), (aq, sa, bq, sb, group_lens)


def _bwd(res, dy):
    aq, sa, bq, sb, group_lens = res
    dyq, sdy = quantize(_mask(dy, group_lens))
    da = _bind(dyq, bq, sdy, sb, group_lens, False, True)      # dy [M, N] x b[g]^T
    db = _bind(aq, dyq, sa, sdy, group_lens, True, False)      # a[g]^T x dy[g] -> [G, K, N]
    return da, db, None


grouped_gemm_fp8_hipblaslt.defvjp(_fwd, _bwd)
